"""
Integration test: wake word detection → VAD recording → RecordingCompleteEvent.

Covers Issue #3 (P0 / CR-01): _consumer_loop must call record_command_with_vad
after a wake word and post RecordingCompleteEvent so the orchestrator's
_handle_recording_complete fires.

Mocking strategy
----------------
- mock_oww_model  (conftest): patches src.audio.ears.Model + src.audio.ears.VAD
- mock_sounddevice (conftest): patches sounddevice.InputStream as a context-manager

Threading model preserved
-------------------------
_consumer_loop runs on a daemon thread; record_command_with_vad blocks that
thread while recording (which prevents wake-detection re-entrancy). The
InputStream context manager stays alive for the duration, so _mic_callback
can continue filling audio_queue.  The test feeds audio into audio_queue from
the main thread to drive record_command_with_vad to completion.
"""

from __future__ import annotations

import queue
import time
import threading
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# CHUNK_SIZE matches ears.py / conftest.py
CHUNK_SIZE: int = 1280


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ears(mock_oww_model: MagicMock, sensitivity: float = 0.5):
    """Construct an Ears instance with mocked OWW model, no filesystem check."""
    with patch("os.path.exists", return_value=False):
        from src.audio.ears import Ears

        return Ears(sensitivity=sensitivity, model_paths=[])


def _speech_chunk() -> np.ndarray:
    """440 Hz sine at 16 kHz, int16 — triggers VAD speech detection."""
    t = np.arange(CHUNK_SIZE, dtype=np.float32) / 16000
    sine = np.sin(2.0 * np.pi * 440.0 * t)
    return (sine * 0.8 * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Integration test: wake → record → RecordingCompleteEvent
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(15)
def test_wake_triggers_record_and_posts_recording_complete_event(
    mock_oww_model: MagicMock,
    mock_sounddevice: MagicMock,
) -> None:
    """_consumer_loop must call record_command_with_vad after a wake and post
    RecordingCompleteEvent.

    RED state (pre-fix): _consumer_loop posts WakeDetectedEvent then breaks
    back to wake-listening without ever calling record_command_with_vad, so
    RecordingCompleteEvent is never posted.

    GREEN state (post-fix): RecordingCompleteEvent appears in the event queue
    within the timeout window.
    """
    from src.core.events import RecordingCompleteEvent, WakeDetectedEvent

    ears = _make_ears(mock_oww_model)
    event_queue: queue.Queue = queue.Queue()

    # --- Configure the OWW model mock ---
    # Use a callable side-effect that fires exactly once then stays silent.
    # A list-based side_effect would exhaust after the _consumer_loop loops
    # back into wake-listening and calls predict() many more times.
    wake_fired = threading.Event()

    def _predict_side_effect(chunk):
        if not wake_fired.is_set():
            wake_fired.set()
            return {"hey_lumi": 0.9}  # wake fires on first call
        return {"hey_lumi": 0.0}     # all subsequent calls stay below threshold

    mock_oww_model.return_value.predict.side_effect = _predict_side_effect

    # --- Configure VAD: speech quickly followed by silence ---
    # record_command_with_vad with silence_limit=1.5 and no initial speech
    # detected falls back to the "no speech detected and >3s elapsed" branch.
    # Set VAD to return silence (0.0) so recording exits quickly on the
    # "no speech within 3s" timeout path — we just need it to complete.
    ears.vad.predict.return_value = 0.0

    # --- Feed audio into audio_queue from a feeder thread ---
    # _consumer_loop's inner loop drains audio_queue for wake-detection.
    # After the wake fires, record_command_with_vad also reads audio_queue.
    # A feeder thread continuously supplies chunks so neither blocks forever.
    stop_feeder = threading.Event()

    def _feeder() -> None:
        chunk = _speech_chunk()
        while not stop_feeder.is_set():
            try:
                ears.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass
            time.sleep(0.005)

    feeder = threading.Thread(target=_feeder, daemon=True, name="TestAudioFeeder")
    feeder.start()

    try:
        ears.start(event_queue)

        # Wait for RecordingCompleteEvent (or WakeDetectedEvent first, then it)
        recording_complete_event = None
        wake_detected_event = None
        deadline = time.monotonic() + 10.0

        while time.monotonic() < deadline:
            try:
                ev = event_queue.get(timeout=0.1)
                if isinstance(ev, WakeDetectedEvent):
                    wake_detected_event = ev
                elif isinstance(ev, RecordingCompleteEvent):
                    recording_complete_event = ev
                    break
            except queue.Empty:
                continue

        # Assert WakeDetectedEvent was posted first
        assert wake_detected_event is not None, (
            "WakeDetectedEvent was never posted — wake word detection is broken"
        )

        # Assert RecordingCompleteEvent was posted (this is the CR-01 regression check)
        assert recording_complete_event is not None, (
            "RecordingCompleteEvent was never posted after wake word detection. "
            "This is the CR-01 bug: _consumer_loop posts WakeDetectedEvent but "
            "never calls record_command_with_vad or posts RecordingCompleteEvent."
        )

        # Verify the audio payload is a numpy int16 array
        assert isinstance(recording_complete_event.audio, np.ndarray), (
            "RecordingCompleteEvent.audio must be a numpy ndarray"
        )
        assert recording_complete_event.audio.dtype == np.int16, (
            f"Expected int16 audio, got {recording_complete_event.audio.dtype}"
        )

    finally:
        stop_feeder.set()
        feeder.join(timeout=2.0)
        ears.stop()


@pytest.mark.integration
@pytest.mark.timeout(15)
def test_wake_triggers_record_command_with_vad_is_called(
    mock_oww_model: MagicMock,
    mock_sounddevice: MagicMock,
) -> None:
    """_consumer_loop must invoke record_command_with_vad after a wake.

    This test spies on record_command_with_vad directly to confirm it is
    called, independent of what it returns.  This complements the event-queue
    test above which verifies the downstream RecordingCompleteEvent.
    """
    from src.core.events import WakeDetectedEvent

    ears = _make_ears(mock_oww_model)
    event_queue: queue.Queue = queue.Queue()

    # Wake fires on first predict call; all subsequent calls return silent.
    # Use callable side_effect to avoid exhaustion when _consumer_loop
    # loops back into wake-listening after recording completes.
    wake_fired2 = threading.Event()

    def _predict_side_effect2(chunk):
        if not wake_fired2.is_set():
            wake_fired2.set()
            return {"hey_lumi": 0.9}
        return {"hey_lumi": 0.0}

    mock_oww_model.return_value.predict.side_effect = _predict_side_effect2

    # VAD: always silence — record_command_with_vad returns in ~3s (no speech)
    ears.vad.predict.return_value = 0.0

    # Track whether record_command_with_vad was called
    record_called = threading.Event()
    record_return_value = np.zeros(1280, dtype=np.int16)

    def _spy_record(*args, **kwargs):
        record_called.set()
        return record_return_value

    ears.record_command_with_vad = _spy_record  # type: ignore[method-assign]

    # Feed chunks so the consumer loop doesn't block on queue.Empty
    stop_feeder = threading.Event()

    def _feeder() -> None:
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16)
        while not stop_feeder.is_set():
            try:
                ears.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass
            time.sleep(0.005)

    feeder = threading.Thread(target=_feeder, daemon=True, name="TestAudioFeeder2")
    feeder.start()

    try:
        ears.start(event_queue)

        # record_command_with_vad must be called within timeout
        was_called = record_called.wait(timeout=8.0)

        assert was_called, (
            "record_command_with_vad was never called after wake word detection. "
            "CR-01: _consumer_loop must invoke it on the wake path."
        )

    finally:
        stop_feeder.set()
        feeder.join(timeout=2.0)
        ears.stop()


# ---------------------------------------------------------------------------
# Shutdown-bound test: stop() while recording must return within 2s
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_stop_while_recording_returns_promptly(
    mock_oww_model: MagicMock,
    mock_sounddevice: MagicMock,
) -> None:
    """stop() called while record_command_with_vad is in progress must unblock
    within a small bound — not block for the full VAD timeout (~10 s).

    This exercises BOTH cooperative-cancel fixes:
      (a) record_command_with_vad exits its loop when self.listening → False.
      (b) thread.join(timeout=2.0) in stop() so the caller is never blocked >2 s.
    """
    from src.core.events import WakeDetectedEvent

    ears = _make_ears(mock_oww_model)
    event_queue: queue.Queue = queue.Queue()

    # Wake fires on first predict, then stays silent.
    wake_fired3 = threading.Event()

    def _predict_side_effect3(chunk):
        if not wake_fired3.is_set():
            wake_fired3.set()
            return {"hey_lumi": 0.9}
        return {"hey_lumi": 0.0}

    mock_oww_model.return_value.predict.side_effect = _predict_side_effect3

    # VAD returns speech score above threshold to keep recording in progress
    # (prevents the "no speech within 3s" early exit, so cooperative cancel
    # via self.listening is the only escape hatch during the test window).
    ears.vad.predict.return_value = 0.6  # above 0.5 threshold → speech_detected=True

    stop_feeder = threading.Event()

    def _feeder() -> None:
        chunk = _speech_chunk()
        while not stop_feeder.is_set():
            try:
                ears.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass
            time.sleep(0.005)

    feeder = threading.Thread(target=_feeder, daemon=True, name="TestAudioFeeder3")
    feeder.start()

    try:
        ears.start(event_queue)

        # Wait for WakeDetectedEvent so we know recording has started
        wake_seen = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                ev = event_queue.get(timeout=0.1)
                if isinstance(ev, WakeDetectedEvent):
                    wake_seen = True
                    break
            except queue.Empty:
                continue

        assert wake_seen, "WakeDetectedEvent never posted — wake detection not working"

        # Give record_command_with_vad a moment to enter its loop
        time.sleep(0.2)

        # Call stop() and assert it returns promptly
        t0 = time.monotonic()
        ears.stop()
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, (
            f"stop() blocked for {elapsed:.2f}s while recording was in progress. "
            "Expected < 2.0s — cooperative cancel or join timeout not working."
        )

    finally:
        stop_feeder.set()
        feeder.join(timeout=2.0)
        # stop() was called above; guard in case thread is still alive
        if hasattr(ears, "thread") and ears.thread.is_alive():
            ears.listening = False
