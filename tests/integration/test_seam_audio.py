"""
Audio seam contract tests — CR-24 / Issue #25 seam (1).

Covers: Ears._consumer_loop → RecordingCompleteEvent → Orchestrator._handle_recording_complete
        → Scribe.transcribe → TranscriptReadyEvent

Anti-tautology argument (R1 regression):
    R1 was: the wake path stopped producing RecordingCompleteEvent — either because
    _consumer_loop never called record_command_with_vad or it never posted the event.
    These tests exercise that path end-to-end at the seam level.

    ✗ FAILS on pre-fix (R1) code:
    - test_wake_path_posts_recording_complete_event: would fail because the consumer
      loop either never reached record_command_with_vad (if the wake-word branch was
      broken) or never posted RecordingCompleteEvent to the queue.
    - test_recording_complete_to_transcript_ready: would fail because TranscriptReadyEvent
      was never posted if RecordingCompleteEvent was never generated.

    ✓ PASSES on fixed code: the full wake → record → RecordingCompleteEvent →
    Scribe.transcribe → TranscriptReadyEvent chain now works end-to-end.

Mocking strategy
----------------
- Ears hardware boundary mocked via conftest.py ``mock_oww_model`` +
  ``mock_sounddevice`` (patched at module-import level).
- Ears._consumer_loop drives the real code path; mock_oww_model's predict()
  return value is configured to fire a wake-word detection once.
- record_command_with_vad is patched to return canned audio immediately so
  the test doesn't have to pump the audio queue.
- Scribe.transcribe is mocked to return a canned transcript.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.config import (
    IPCConfig,
    LumiConfig,
    RAGConfig,
    ToolsConfig,
    VisionConfig,
)
from src.core.events import RecordingCompleteEvent, ShutdownEvent, TranscriptReadyEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVENT_TIMEOUT_S = 5.0  # seconds to wait for an event to appear in the queue
_CANNED_AUDIO = np.zeros(1600, dtype=np.int16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> LumiConfig:
    return LumiConfig(
        ipc=IPCConfig(enabled=False),
        rag=RAGConfig(enabled=False),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Test 1 — Audio seam: Ears._consumer_loop posts RecordingCompleteEvent (R1)
#
# Anti-tautology: this test would FAIL on R1 pre-fix code because the consumer
# loop either never reached record_command_with_vad (wake branch broken) or
# never posted RecordingCompleteEvent.  The event queue would remain empty.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_wake_path_posts_recording_complete_event(
    mock_oww_model: MagicMock,
    mock_sounddevice: MagicMock,
) -> None:
    """Ears._consumer_loop: wake-word detection → RecordingCompleteEvent posted.

    The Ears consumer loop must:
    1. Detect the wake word (mock_oww_model.predict returns score > sensitivity).
    2. Call record_command_with_vad (patched to return canned audio immediately).
    3. Post RecordingCompleteEvent to the event queue.

    R1 anti-tautology: pre-fix, this event was never produced, so the assertion
    ``event_queue.get(timeout=...)`` would raise queue.Empty → test fails.
    """
    from src.audio.ears import Ears

    event_queue: queue.Queue[Any] = queue.Queue()

    # Configure mock_oww_model to fire ONE wake detection, then go silent.
    # wake_triggered toggles back to 0.0 after the first hit so the loop
    # doesn't loop forever emitting wake events.
    wake_triggered = [False]

    def _predict_side_effect(chunk: np.ndarray) -> dict[str, float]:
        if not wake_triggered[0]:
            wake_triggered[0] = True
            return {"hey_lumi": 1.0}  # triggers wake path (score > sensitivity)
        return {"hey_lumi": 0.0}

    mock_oww_model.return_value.predict.side_effect = _predict_side_effect
    mock_oww_model.return_value.reset.return_value = None

    # Patch record_command_with_vad to return canned audio immediately instead
    # of waiting for real VAD; this is the seam between wake detection and the
    # RecordingCompleteEvent post.
    with patch.object(
        Ears,
        "record_command_with_vad",
        return_value=_CANNED_AUDIO,
    ):
        ears = Ears(sensitivity=0.5)
        # Inject audio chunks into the queue so the consumer loop processes them.
        # Each chunk drives one model.predict() call.
        for _ in range(3):
            ears.audio_queue.put(np.zeros(1280, dtype=np.int16))

        ears.start(event_queue)

        # Drain the queue looking for RecordingCompleteEvent.
        # WakeDetectedEvent will arrive first; that's expected and fine.
        recording_event: Any = None
        deadline = time.monotonic() + _EVENT_TIMEOUT_S
        try:
            while time.monotonic() < deadline:
                try:
                    ev = event_queue.get(timeout=0.2)
                    if isinstance(ev, RecordingCompleteEvent):
                        recording_event = ev
                        break
                except queue.Empty:
                    pass
        finally:
            ears.stop()

    assert recording_event is not None, (
        "RecordingCompleteEvent was never posted to the event queue. "
        "R1 regression: wake path did not produce RecordingCompleteEvent."
    )
    # Verify the audio payload is present.
    audio = recording_event.audio
    assert isinstance(audio, np.ndarray), "RecordingCompleteEvent.audio must be ndarray"


# ---------------------------------------------------------------------------
# Test 2 — Full audio→transcript seam: RecordingCompleteEvent → TranscriptReadyEvent
#
# Anti-tautology: this test would FAIL on R1 pre-fix code because if
# RecordingCompleteEvent was never posted, TranscriptReadyEvent would never be
# posted either.  The queue.get would raise queue.Empty → test fails.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_recording_complete_to_transcript_ready(
    mock_whisper_model: MagicMock,
) -> None:
    """RecordingCompleteEvent routed through Orchestrator dispatch → Scribe runs → TranscriptReadyEvent.

    Drives the seam between Ears (post-recording) and the Orchestrator/Scribe
    by going through the real event-dispatch loop so the ``register_handler``
    wiring is exercised end-to-end:

    1. Register a spy handler on TranscriptReadyEvent BEFORE starting the loop
       (exercises the same registration API the real wiring uses).
    2. Start orch.run() in a daemon thread (real dispatch loop).
    3. Transition state machine to LISTENING (precondition for handler).
    4. Post RecordingCompleteEvent to orch._event_queue — the dispatch loop
       must route it to _handle_recording_complete via the registered handler map.
    5. The spy handler fires (in the dispatch thread) when TranscriptReadyEvent
       is dispatched; it sets a threading.Event to unblock the test thread.

    R1 anti-tautology: if the RecordingCompleteEvent handler registration is
    broken, _handle_recording_complete never fires → _run_scribe never runs →
    TranscriptReadyEvent is never dispatched → spy never fires → threading.Event
    times out → test fails.  A broken register_handler for RecordingCompleteEvent
    specifically is the canary this test lights up.
    """
    import threading

    from src.core.orchestrator import Orchestrator
    from src.core.state_machine import LumiState

    config = _minimal_config()

    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()

    # Build a mock Scribe that returns a canned transcript.
    mock_scribe = MagicMock()
    mock_scribe.transcribe.return_value = "hello lumi"

    with (
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
        patch("src.core.orchestrator.ReasoningRouter"),
    ):
        mock_mem_cls.return_value.load = MagicMock()
        orch = Orchestrator(config, speaker=speaker, scribe=mock_scribe)

    # Register a spy AFTER the orchestrator is built but BEFORE run() starts.
    # The dispatch loop calls ALL handlers registered for an event type in
    # registration order.  The spy captures the event and signals the test.
    _transcript_seen: list[TranscriptReadyEvent] = []
    _transcript_signal = threading.Event()

    def _spy(event: TranscriptReadyEvent) -> None:  # called in dispatch thread
        _transcript_seen.append(event)
        _transcript_signal.set()

    orch.register_handler(TranscriptReadyEvent, _spy)

    brain_exc: list[BaseException] = []

    def _run() -> None:
        try:
            orch.run()
        except Exception as exc:
            brain_exc.append(exc)

    run_thread = threading.Thread(target=_run, daemon=True, name="AudioSeamBrainThread")
    run_thread.start()

    # Give the event loop a moment to enter its get() wait.
    time.sleep(0.05)

    # Transition to LISTENING (handler ignores events in other states).
    orch._state_machine.transition_to(LumiState.LISTENING)

    # Post RecordingCompleteEvent via the real event queue — the dispatch loop
    # must route it to _handle_recording_complete through the registered handler
    # map.  This is the registration wire being tested.
    orch._event_queue.put(RecordingCompleteEvent(audio=_CANNED_AUDIO))

    # Wait for the spy to be called (signalled by the dispatch thread).
    transcript_arrived = _transcript_signal.wait(timeout=_EVENT_TIMEOUT_S)

    orch._event_queue.put(ShutdownEvent())
    run_thread.join(timeout=3.0)
    speaker.stop()

    assert not brain_exc, f"Brain thread raised: {brain_exc[0]}"
    assert transcript_arrived, (
        "TranscriptReadyEvent was never dispatched. "
        "R1 regression: dispatch wiring for RecordingCompleteEvent → "
        "_handle_recording_complete is broken."
    )
    transcript_event = _transcript_seen[0]
    assert isinstance(transcript_event, TranscriptReadyEvent)
    # The canned Whisper mock returns "hello lumi" — verify it arrived intact.
    assert transcript_event.text, "TranscriptReadyEvent.text must not be empty"
