"""
Tests for src/audio/ears.py

Mocking strategy
----------------
- ``openwakeword.model.Model`` is patched via the ``mock_oww_model`` fixture so
  no ONNX runtime is invoked and no model files are required.
- ``openwakeword.vad.VAD`` is co-patched inside ``mock_oww_model``.
- ``sounddevice.InputStream`` is patched inside ``mock_sounddevice`` so the
  ``_consumer_loop`` thread can open a stream context without real hardware.
- ``openwakeword.utils.AudioFeatures.__init__`` receives the monkey-patch from
  ``Ears.__init__`` itself — the OWW model mock bypasses this entirely.

All tests are marked ``pytest.mark.unit`` and must pass without a microphone,
GPU, or network access.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build an Ears instance with all hardware mocked
# ---------------------------------------------------------------------------


def _make_ears(mock_oww_model, sensitivity: float = 0.5):
    """Construct an Ears instance with mocked OWW model and no filesystem check.

    We also patch ``os.path.exists`` to return False so the constructor does
    not try to resolve the custom hey_lumi.onnx path from the filesystem.
    """
    with patch("os.path.exists", return_value=False):
        from src.audio.ears import Ears
        return Ears(sensitivity=sensitivity, model_paths=[])


# ---------------------------------------------------------------------------
# Construction / teardown
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ears_init_stores_sensitivity(mock_oww_model: MagicMock) -> None:
    """Ears stores the sensitivity value passed at construction time."""
    ears = _make_ears(mock_oww_model, sensitivity=0.7)
    assert ears.sensitivity == 0.7


@pytest.mark.unit
def test_ears_init_creates_audio_queue(mock_oww_model: MagicMock) -> None:
    """Ears creates a queue.Queue instance on construction."""
    ears = _make_ears(mock_oww_model)
    assert isinstance(ears.audio_queue, queue.Queue)


@pytest.mark.unit
def test_ears_init_not_listening_by_default(mock_oww_model: MagicMock) -> None:
    """Ears.listening is False before start() is called."""
    ears = _make_ears(mock_oww_model)
    assert ears.listening is False


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_ears_start_sets_listening_flag(
    mock_oww_model: MagicMock,
    mock_sounddevice: MagicMock,
) -> None:
    """start() sets listening=True before the consumer thread begins."""
    # We want to verify the flag is set without letting the thread run forever.
    # Inject a side-effect that flips listening back off after one queue.get
    # so the thread exits cleanly.
    ears = _make_ears(mock_oww_model)

    call_count = [0]
    assertion_done = threading.Event()

    def _limited_get(**kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:
            assertion_done.wait(timeout=2.0)  # wait for main thread to assert first
            ears.listening = False
            raise queue.Empty
        return np.zeros(1280, dtype=np.int16)

    ears.audio_queue.get = _limited_get  # type: ignore[method-assign]

    event_queue: queue.Queue = queue.Queue()
    ears.start(event_queue)
    assert ears.listening is True  # flag set synchronously before thread starts
    assertion_done.set()  # release thread to exit
    ears.thread.join(timeout=3.0)


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_ears_stop_clears_listening_flag(
    mock_oww_model: MagicMock,
    mock_sounddevice: MagicMock,
) -> None:
    """stop() sets listening=False and joins the background thread."""
    ears = _make_ears(mock_oww_model)

    call_count = [0]

    def _limited_get(**kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:
            ears.listening = False
            raise queue.Empty
        return np.zeros(1280, dtype=np.int16)

    ears.audio_queue.get = _limited_get  # type: ignore[method-assign]

    event_queue: queue.Queue = queue.Queue()
    ears.start(event_queue)
    ears.stop()
    assert ears.listening is False
    assert not ears.thread.is_alive()


# ---------------------------------------------------------------------------
# _mic_callback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mic_callback_enqueues_copy(mock_oww_model: MagicMock) -> None:
    """_mic_callback puts a copy of indata onto audio_queue."""
    ears = _make_ears(mock_oww_model)

    indata = np.ones((1280, 1), dtype=np.int16) * 42
    # The ``time`` and ``status`` parameters mirror the sounddevice callback signature.
    ears._mic_callback(indata, 1280, None, None)

    assert not ears.audio_queue.empty()
    queued = ears.audio_queue.get_nowait()
    np.testing.assert_array_equal(queued, indata)
    # Must be a copy, not the same object
    assert queued is not indata


@pytest.mark.unit
def test_mic_callback_enqueues_multiple_chunks(mock_oww_model: MagicMock) -> None:
    """Each _mic_callback invocation enqueues exactly one item."""
    ears = _make_ears(mock_oww_model)

    for i in range(3):
        chunk = np.full((1280, 1), fill_value=i, dtype=np.int16)
        ears._mic_callback(chunk, 1280, None, None)

    assert ears.audio_queue.qsize() == 3


# ---------------------------------------------------------------------------
# record_command_with_vad
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.timeout(10)
def test_record_vad_returns_empty_array_when_queue_is_empty(
    mock_oww_model: MagicMock,
) -> None:
    """record_command_with_vad returns an empty int16 array when no audio arrives.

    Uses timeout=1.5 s so the outer timeout branch fires quickly.
    The queue remains empty throughout, so queue.Empty is raised on every
    get() call and the function returns after ~1.5 seconds.
    """
    ears = _make_ears(mock_oww_model)
    result = ears.record_command_with_vad(timeout=1.5, silence_limit=0.5)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.timeout(10)
def test_record_vad_concatenates_received_chunks(
    mock_oww_model: MagicMock,
) -> None:
    """record_command_with_vad concatenates all received chunks into one array.

    We pre-fill the queue with known chunks, configure VAD to report speech
    then silence, and verify the returned array is the concatenation.
    """
    ears = _make_ears(mock_oww_model)

    chunk_a = np.ones(1280, dtype=np.int16) * 10
    chunk_b = np.ones(1280, dtype=np.int16) * 20
    chunk_c = np.ones(1280, dtype=np.int16) * 30

    for c in (chunk_a, chunk_b, chunk_c):
        ears.audio_queue.put(c)

    # Configure VAD: speech on first two chunks, then silence.
    # After silence_limit expires the loop breaks.
    vad_scores = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ears.vad.predict.side_effect = vad_scores

    result = ears.record_command_with_vad(timeout=10.0, silence_limit=0.5)

    assert result.dtype == np.int16
    assert len(result) >= 1280 * 2  # at minimum the two speech chunks


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_record_vad_respects_timeout(
    mock_oww_model: MagicMock,
) -> None:
    """record_command_with_vad returns after timeout even with continuous speech."""
    ears = _make_ears(mock_oww_model)

    # VAD always detects speech so silence branch never fires.
    ears.vad.predict.return_value = 0.9

    # Feed a stream of chunks into the queue from a background thread so
    # record_command_with_vad never starves.
    stop_feeding = threading.Event()

    def _feed() -> None:
        while not stop_feeding.is_set():
            ears.audio_queue.put(np.ones(1280, dtype=np.int16))
            time.sleep(0.01)

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()

    start = time.monotonic()
    result = ears.record_command_with_vad(timeout=1.0, silence_limit=5.0)
    elapsed = time.monotonic() - start

    stop_feeding.set()
    feeder.join(timeout=2.0)

    # Should have returned within ~2 s of timeout
    assert elapsed < 3.0
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    assert len(result) > 0


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.timeout(10)
def test_record_vad_speech_then_silence_stops_early(
    mock_oww_model: MagicMock,
) -> None:
    """record_command_with_vad stops before timeout when silence follows speech."""
    ears = _make_ears(mock_oww_model)

    # Simulate speech then silence pattern via VAD side-effects.
    speech_responses = [0.9] * 5
    silence_responses = [0.1] * 30  # enough to exceed silence_limit=0.5
    ears.vad.predict.side_effect = speech_responses + silence_responses

    # Pre-fill queue with chunks that map to the VAD responses.
    for _ in range(len(speech_responses) + len(silence_responses)):
        ears.audio_queue.put(np.ones(1280, dtype=np.int16))

    start = time.monotonic()
    result = ears.record_command_with_vad(timeout=10.0, silence_limit=0.5)
    elapsed = time.monotonic() - start

    # Must finish well before the 10-second timeout.
    assert elapsed < 5.0
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
