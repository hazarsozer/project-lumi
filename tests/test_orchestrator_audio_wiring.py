"""
Tests for Wave E1 — audio-in pipeline wiring.

Covers:
- WakeDetectedEvent in IDLE → transitions to LISTENING
- WakeDetectedEvent when not IDLE → state unchanged
- RecordingCompleteEvent in LISTENING → Scribe invoked, TranscriptReadyEvent posted
- RecordingCompleteEvent when not LISTENING → no downstream event
- WakeDetectedEvent in SPEAKING → posts InterruptEvent (or at minimum does not crash)
- Ears.start() is called when orchestrator starts
- Ears.stop() is called on ShutdownEvent
- Scribe transcription runs in a daemon thread (not the dispatch thread)
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.core.config import (
    IPCConfig,
    LumiConfig,
    RAGConfig,
    ToolsConfig,
    VisionConfig,
)
from src.core.events import (
    EarsErrorEvent,
    InterruptEvent,
    RecordingCompleteEvent,
    ShutdownEvent,
    TranscriptReadyEvent,
    WakeDetectedEvent,
)
from src.core.state_machine import LumiState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> LumiConfig:
    """Return a LumiConfig with all heavy subsystems disabled."""
    return LumiConfig(
        ipc=IPCConfig(enabled=False),
        rag=RAGConfig(enabled=False),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
    )


def _make_orchestrator(*, ears=None, scribe=None):
    """Build an Orchestrator with all hardware subsystems mocked.

    Args:
        ears: Optional mock Ears instance to inject.
        scribe: Optional mock Scribe instance to inject.

    Returns:
        An Orchestrator ready for testing.
    """
    from src.core.orchestrator import Orchestrator

    config = _minimal_config()

    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()

    with (
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
        patch("src.core.orchestrator.ReasoningRouter"),
    ):
        mock_mem_cls.return_value.load = MagicMock()
        orch = Orchestrator(config, speaker=speaker, ears=ears, scribe=scribe)
    return orch


def _sample_audio() -> np.ndarray:
    """Return a minimal int16 numpy array simulating recorded audio."""
    return np.zeros(1600, dtype=np.int16)


# ---------------------------------------------------------------------------
# 1. WakeDetectedEvent in IDLE → transitions to LISTENING
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wake_detected_transitions_to_listening():
    """Posting WakeDetectedEvent from IDLE moves the state machine to LISTENING."""
    orch = _make_orchestrator()
    assert orch._state_machine.current_state == LumiState.IDLE

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=1.0))

    assert orch._state_machine.current_state == LumiState.LISTENING


# ---------------------------------------------------------------------------
# 2. WakeDetectedEvent when not IDLE → state unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wake_detected_ignored_when_processing():
    """WakeDetectedEvent arriving while PROCESSING does not change the state."""
    orch = _make_orchestrator()
    # Manually drive to PROCESSING (IDLE → LISTENING → PROCESSING)
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=2.0))

    assert orch._state_machine.current_state == LumiState.PROCESSING


@pytest.mark.unit
def test_wake_detected_ignored_when_speaking():
    """WakeDetectedEvent arriving while SPEAKING does not raise and leaves state machine untouched."""
    orch = _make_orchestrator()
    # Drive to SPEAKING
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)
    orch._state_machine.transition_to(LumiState.SPEAKING)

    # Should not raise; state should remain SPEAKING (or IDLE if interrupt was fired)
    # Minimum contract: does not crash
    orch._handle_wake_detected(WakeDetectedEvent(timestamp=3.0))

    # State should be SPEAKING or IDLE (interrupt path transitions to IDLE)
    assert orch._state_machine.current_state in (LumiState.SPEAKING, LumiState.IDLE)


@pytest.mark.unit
def test_wake_detected_ignored_when_listening():
    """WakeDetectedEvent arriving while already LISTENING does not re-transition."""
    orch = _make_orchestrator()
    orch._state_machine.transition_to(LumiState.LISTENING)

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=4.0))

    assert orch._state_machine.current_state == LumiState.LISTENING


# ---------------------------------------------------------------------------
# 3. RecordingCompleteEvent in LISTENING → Scribe invoked, TranscriptReadyEvent posted
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_recording_complete_dispatches_scribe():
    """RecordingCompleteEvent in LISTENING triggers Scribe.transcribe() and
    eventually posts TranscriptReadyEvent to the event queue."""
    mock_scribe = MagicMock()
    mock_scribe.transcribe.return_value = "hello lumi"

    orch = _make_orchestrator(scribe=mock_scribe)
    orch._state_machine.transition_to(LumiState.LISTENING)

    audio = _sample_audio()
    orch._handle_recording_complete(RecordingCompleteEvent(audio=audio))

    # Give the daemon thread time to complete
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if not orch._event_queue.empty():
            break
        time.sleep(0.02)

    assert not orch._event_queue.empty(), "TranscriptReadyEvent was never posted"
    event = orch._event_queue.get_nowait()
    assert isinstance(event, TranscriptReadyEvent)
    assert event.text == "hello lumi"

    mock_scribe.transcribe.assert_called_once()


# ---------------------------------------------------------------------------
# 4. RecordingCompleteEvent when not LISTENING → no downstream event
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_recording_complete_ignored_when_idle():
    """RecordingCompleteEvent received while IDLE is silently dropped."""
    mock_scribe = MagicMock()

    orch = _make_orchestrator(scribe=mock_scribe)
    # State is IDLE (default)
    assert orch._state_machine.current_state == LumiState.IDLE

    audio = _sample_audio()
    orch._handle_recording_complete(RecordingCompleteEvent(audio=audio))

    time.sleep(0.15)

    assert orch._event_queue.empty(), "No event should have been posted"
    mock_scribe.transcribe.assert_not_called()


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_recording_complete_ignored_when_processing():
    """RecordingCompleteEvent received while PROCESSING is silently dropped."""
    mock_scribe = MagicMock()

    orch = _make_orchestrator(scribe=mock_scribe)
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)

    audio = _sample_audio()
    orch._handle_recording_complete(RecordingCompleteEvent(audio=audio))

    time.sleep(0.15)

    assert orch._event_queue.empty(), "No event should have been posted"
    mock_scribe.transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# 5. WakeDetectedEvent in SPEAKING → at minimum does not crash
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wake_while_speaking_does_not_crash():
    """Posting WakeDetectedEvent in SPEAKING state must not raise an exception.

    Wave E4 will add the interrupt path; for now we ensure the handler is
    robust and the orchestrator remains in a valid state.
    """
    orch = _make_orchestrator()
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)
    orch._state_machine.transition_to(LumiState.SPEAKING)

    # Must not raise
    orch._handle_wake_detected(WakeDetectedEvent(timestamp=5.0))

    # Orchestrator must be in a valid state
    assert orch._state_machine.current_state in (
        LumiState.SPEAKING,
        LumiState.IDLE,
        LumiState.LISTENING,
    )


# ---------------------------------------------------------------------------
# 6. Ears.start() is called when orchestrator starts
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_ears_started_on_orchestrator_start():
    """When Orchestrator.run() is entered, Ears.start() has been called with
    the orchestrator's event queue."""
    mock_ears = MagicMock()
    mock_ears.start = MagicMock()

    orch = _make_orchestrator(ears=mock_ears)

    # Post ShutdownEvent immediately so run() exits
    orch.post_event(ShutdownEvent())
    orch.run()

    mock_ears.start.assert_called_once_with(orch._event_queue)


# ---------------------------------------------------------------------------
# 7. Ears.stop() is called on ShutdownEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_ears_stopped_on_shutdown():
    """Ears.stop() is called when ShutdownEvent is handled."""
    mock_ears = MagicMock()
    mock_ears.stop = MagicMock()

    orch = _make_orchestrator(ears=mock_ears)
    orch.post_event(ShutdownEvent())
    orch.run()

    mock_ears.stop.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Scribe runs in a daemon thread (not blocking the dispatch loop)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_scribe_runs_in_daemon_thread():
    """Scribe.transcribe() must execute in a daemon thread, not in the
    orchestrator's dispatch thread. We verify this by checking that the
    calling thread inside transcribe() is not the test's main thread and
    is marked as a daemon thread."""
    transcription_thread: list[threading.Thread | None] = [None]
    transcription_event = threading.Event()

    def _slow_transcribe(audio):
        transcription_thread[0] = threading.current_thread()
        transcription_event.set()
        return "threaded result"

    mock_scribe = MagicMock()
    mock_scribe.transcribe.side_effect = _slow_transcribe

    orch = _make_orchestrator(scribe=mock_scribe)
    orch._state_machine.transition_to(LumiState.LISTENING)

    audio = _sample_audio()
    orch._handle_recording_complete(RecordingCompleteEvent(audio=audio))

    # Wait for transcribe() to be entered
    assert transcription_event.wait(timeout=4.0), "transcribe() was never called"

    worker = transcription_thread[0]
    assert worker is not None
    assert worker is not threading.main_thread(), (
        "Scribe.transcribe() ran on the main thread — it must run in a daemon thread"
    )
    assert worker.daemon, "Scribe worker thread must be a daemon thread"


# ---------------------------------------------------------------------------
# 9. Scribe failure falls back to IDLE
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_scribe_failure_returns_to_idle():
    """When Scribe.transcribe() raises, the orchestrator returns to IDLE
    and does not post a TranscriptReadyEvent."""
    mock_scribe = MagicMock()
    mock_scribe.transcribe.side_effect = RuntimeError("STT exploded")

    orch = _make_orchestrator(scribe=mock_scribe)
    orch._state_machine.transition_to(LumiState.LISTENING)

    audio = _sample_audio()
    orch._handle_recording_complete(RecordingCompleteEvent(audio=audio))

    # Give daemon thread time to finish
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if orch._state_machine.current_state == LumiState.IDLE:
            break
        time.sleep(0.02)

    assert orch._state_machine.current_state == LumiState.IDLE
    assert orch._event_queue.empty(), "No TranscriptReadyEvent should be posted on error"


# ---------------------------------------------------------------------------
# 10. No Ears → start/stop calls do not crash (text-only mode)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_no_ears_text_only_mode_runs_without_crash():
    """When ears=None, orchestrator.run() completes normally.

    This preserves the text-only mode used by ZMQ/UserTextEvent path.
    """
    orch = _make_orchestrator(ears=None)
    orch.post_event(ShutdownEvent())
    # Must not raise
    orch.run()
