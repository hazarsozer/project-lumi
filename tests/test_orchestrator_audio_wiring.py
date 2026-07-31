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

    # State should be LISTENING (Wave E4 interrupt path: SPEAKING → IDLE → LISTENING).
    # SPEAKING and IDLE remain accepted for forward-compatibility should the
    # implementation change again, but the expected outcome is now LISTENING.
    assert orch._state_machine.current_state in (LumiState.SPEAKING, LumiState.IDLE, LumiState.LISTENING)


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

    # Poll until TranscriptReadyEvent appears in the queue.
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if not orch._event_queue.empty():
            break
        time.sleep(0.005)

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

    # Legitimate negative-test window: wait briefly to confirm no event is posted.
    # The _handle_recording_complete in IDLE is synchronous (no thread dispatched),
    # so any erroneous async action would appear within this window.
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

    # Legitimate negative-test window: no thread is dispatched in PROCESSING
    # state, but we wait briefly to confirm nothing posts unexpectedly.
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
# Wave E4 — wake-while-speaking interrupt tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wake_while_speaking_posts_interrupt_event():
    """WakeDetectedEvent in SPEAKING state must post an InterruptEvent to the
    event queue so the TTS playback is cancelled before re-entering LISTENING."""
    orch = _make_orchestrator()
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)
    orch._state_machine.transition_to(LumiState.SPEAKING)

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=10.0))

    # Drain the queue and look for an InterruptEvent.
    events: list = []
    try:
        while True:
            events.append(orch._event_queue.get_nowait())
    except Exception:
        pass

    interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupt_events) == 1, (
        f"Expected exactly one InterruptEvent; got: {events}"
    )
    assert interrupt_events[0].source == "wake_word"


@pytest.mark.unit
def test_wake_while_speaking_transitions_to_listening():
    """After posting InterruptEvent, the state machine must be in LISTENING
    (not SPEAKING or IDLE) so that recording starts immediately."""
    orch = _make_orchestrator()
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)
    orch._state_machine.transition_to(LumiState.SPEAKING)

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=11.0))

    assert orch._state_machine.current_state == LumiState.LISTENING


@pytest.mark.unit
def test_wake_while_not_speaking_no_interrupt():
    """WakeDetectedEvent from IDLE must NOT post an InterruptEvent — only the
    standard IDLE → LISTENING transition should happen."""
    orch = _make_orchestrator()
    assert orch._state_machine.current_state == LumiState.IDLE

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=12.0))

    # Queue must be empty — no InterruptEvent should have been posted.
    events: list = []
    try:
        while True:
            events.append(orch._event_queue.get_nowait())
    except Exception:
        pass

    interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupt_events) == 0, (
        f"InterruptEvent must not be posted from IDLE; got: {events}"
    )
    assert orch._state_machine.current_state == LumiState.LISTENING


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

    # Poll until state recovers to IDLE after the STT error.
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if orch._state_machine.current_state == LumiState.IDLE:
            break
        time.sleep(0.005)

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


# ---------------------------------------------------------------------------
# R4 regression — orchestrator must build a real TTS in the production path.
# main.py constructs the Orchestrator without a `tts=` argument; before this
# fix the orchestrator did not auto-build one, so _tts stayed None, every reply
# hit the `tts is None` fast-path, and the Brain shipped permanently silent
# regardless of config.tts.enabled.  Caught only by the DoD §2 live-test gate.
# ---------------------------------------------------------------------------


def test_orchestrator_builds_tts_when_enabled_and_audio_out_not_injected() -> None:
    """No speaker and no tts injected (the main.py path) → orchestrator builds a
    real KokoroTTS, wired to the auto-created SpeakerThread, so it can speak."""
    from src.audio.mouth import KokoroTTS
    from src.core.orchestrator import Orchestrator

    config = _minimal_config()  # tts defaults to TTSConfig(enabled=True)
    assert config.tts.enabled is True  # guard: this is the scenario under test

    with (
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
        patch("src.core.orchestrator.ReasoningRouter"),
        patch("src.audio.speaker.sd.OutputStream"),  # no real audio device
        patch.object(KokoroTTS, "_load_model", lambda self: None),  # skip 89MB load
    ):
        mock_mem_cls.return_value.load = MagicMock()
        orch = Orchestrator(config)  # no speaker, no tts → production path
        try:
            assert isinstance(orch._tts, KokoroTTS), (
                "Orchestrator must auto-build KokoroTTS when config.tts.enabled "
                "and no audio-out is injected (R4: Brain shipped silent)."
            )
            assert orch._tts._speaker is orch._speaker  # wired to the real speaker
        finally:
            orch._speaker.stop()


def test_orchestrator_no_tts_when_speaker_injected() -> None:
    """Injecting a speaker (the test path) must NOT auto-build a real TTS — the
    89MB model stays unloaded and existing fixtures keep their None behaviour."""
    orch = _make_orchestrator()
    assert orch._tts is None
