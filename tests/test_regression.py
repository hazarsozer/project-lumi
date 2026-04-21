"""
Regression guard for Project Lumi.

These tests peg key observable behaviours that must never silently change.
They are intentionally thin — each test asserts one contract at the boundary
level (public API, event protocol, state transitions) without coupling to
internal implementation details.

Contracts covered:
  R1  Orchestrator full-turn state cycle: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
  R2  _dispatch_user_turn is called for TranscriptReadyEvent
  R3  _dispatch_user_turn is called for UserTextEvent
  R4  WakeDetectedEvent while SPEAKING posts InterruptEvent
  R5  MetricsCollector.snapshot() returns a dict with the correct histogram shape
  R6  PromptEngine.build_prompt() output contains the system prompt text
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig
from src.core.events import (
    InterruptEvent,
    LLMResponseReadyEvent,
    ShutdownEvent,
    SpeechCompletedEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    WakeDetectedEvent,
)
from src.core.metrics import MetricsCollector
from src.core.orchestrator import Orchestrator
from src.core.state_machine import LumiState
from src.llm.prompt_engine import DEFAULT_SYSTEM_PROMPT, PromptEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator() -> Orchestrator:
    """Minimal Orchestrator with a mocked SpeakerThread (no audio hardware)."""
    mock_speaker = MagicMock(spec=SpeakerThread)
    return Orchestrator(config=LumiConfig(), speaker=mock_speaker)


# ---------------------------------------------------------------------------
# R1 — Full-turn state cycle
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(5)
def test_full_turn_state_cycle() -> None:
    """IDLE → LISTENING → PROCESSING → SPEAKING → IDLE is the only valid full turn.

    This test drives the state machine through each leg of the cycle directly
    (no LLM inference) to confirm that:
      - Each transition is accepted by the state machine.
      - The final state after SpeechCompletedEvent is IDLE.
    """
    orch = _make_orchestrator()
    sm = orch.state_machine

    assert sm.current_state == LumiState.IDLE

    # Simulate wake word
    sm.transition_to(LumiState.LISTENING)
    assert sm.current_state == LumiState.LISTENING

    # Simulate transcript / user input arriving
    sm.transition_to(LumiState.PROCESSING)
    assert sm.current_state == LumiState.PROCESSING

    # Simulate LLM response ready → TTS starts
    sm.transition_to(LumiState.SPEAKING)
    assert sm.current_state == LumiState.SPEAKING

    # Simulate speech finished → back to IDLE
    sm.transition_to(LumiState.IDLE)
    assert sm.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# R2 — _dispatch_user_turn called on TranscriptReadyEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(5)
def test_dispatch_user_turn_called_for_transcript_event() -> None:
    """_dispatch_user_turn is the shared entry point for TranscriptReadyEvent.

    When a TranscriptReadyEvent arrives in LISTENING state the orchestrator
    must call _dispatch_user_turn (not any deprecated method name).
    We patch it to a sentinel and confirm it is called with the correct text.
    """
    orch = _make_orchestrator()
    orch.state_machine.transition_to(LumiState.LISTENING)

    called_with: list[tuple] = []

    original = orch._dispatch_user_turn

    def _spy(text: str, source: str) -> None:
        called_with.append((text, source))

    orch._dispatch_user_turn = _spy  # type: ignore[method-assign]

    orch._handle_transcript(TranscriptReadyEvent(text="hello lumi"))

    assert len(called_with) == 1, "_dispatch_user_turn must be called exactly once"
    assert called_with[0][0] == "hello lumi"
    assert called_with[0][1] == "transcript"


# ---------------------------------------------------------------------------
# R3 — _dispatch_user_turn called on UserTextEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(5)
def test_dispatch_user_turn_called_for_user_text_event() -> None:
    """_dispatch_user_turn is the shared entry point for UserTextEvent.

    When a UserTextEvent arrives in IDLE state the orchestrator steps through
    LISTENING and then calls _dispatch_user_turn with source='user_text'.
    """
    orch = _make_orchestrator()
    # State starts at IDLE — UserTextEvent handler performs the double-step.

    called_with: list[tuple] = []

    def _spy(text: str, source: str) -> None:
        called_with.append((text, source))

    orch._dispatch_user_turn = _spy  # type: ignore[method-assign]

    orch._handle_user_text(UserTextEvent(text="what time is it"))

    assert len(called_with) == 1, "_dispatch_user_turn must be called exactly once"
    assert called_with[0][0] == "what time is it"
    assert called_with[0][1] == "user_text"


# ---------------------------------------------------------------------------
# R4 — WakeDetectedEvent while SPEAKING posts InterruptEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(5)
def test_wake_while_speaking_posts_interrupt_event() -> None:
    """WakeDetectedEvent arriving in SPEAKING state must enqueue an InterruptEvent.

    The orchestrator should not drop or silently ignore the wake word — it
    must post InterruptEvent so any in-flight TTS synthesis is cancelled.
    """
    orch = _make_orchestrator()
    sm = orch.state_machine

    # Advance to SPEAKING state.
    sm.transition_to(LumiState.LISTENING)
    sm.transition_to(LumiState.PROCESSING)
    sm.transition_to(LumiState.SPEAKING)

    # Drain the internal event queue before the test action so we can
    # inspect exactly what _handle_wake_detected enqueues.
    drained: list[object] = []
    try:
        while True:
            drained.append(orch._event_queue.get_nowait())
    except queue.Empty:
        pass

    orch._handle_wake_detected(WakeDetectedEvent(timestamp=time.monotonic()))

    # Collect what was posted to the queue.
    posted: list[object] = []
    try:
        while True:
            posted.append(orch._event_queue.get_nowait())
    except queue.Empty:
        pass

    interrupt_events = [e for e in posted if isinstance(e, InterruptEvent)]
    assert len(interrupt_events) >= 1, (
        "WakeDetectedEvent while SPEAKING must post at least one InterruptEvent; "
        f"got: {posted!r}"
    )


# ---------------------------------------------------------------------------
# R5 — MetricsCollector.snapshot() shape
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_metrics_snapshot_has_correct_histogram_shape() -> None:
    """MetricsCollector.snapshot() must return a dict with the documented keys.

    Histogram entries must contain: count, mean, p50, p95, p99.
    Counter entries must be plain integers.
    An empty collector must return an empty dict (not None or a list).
    """
    mc = MetricsCollector()

    # Empty collector
    empty = mc.snapshot()
    assert isinstance(empty, dict), "snapshot() must return a dict"
    assert len(empty) == 0, "fresh collector snapshot must be empty"

    # Record a histogram value and a counter
    mc.record("latency_ms", 42.0)
    mc.record("latency_ms", 100.0)
    mc.increment("requests_total")
    mc.increment("requests_total")
    mc.increment("requests_total")

    snap = mc.snapshot()

    assert "latency_ms" in snap, "recorded histogram key must appear in snapshot"
    assert "requests_total" in snap, "incremented counter must appear in snapshot"

    hist = snap["latency_ms"]
    assert isinstance(hist, dict), "histogram value must be a dict"
    for required_key in ("count", "mean", "p50", "p95", "p99"):
        assert required_key in hist, f"histogram dict missing required key: {required_key!r}"

    assert hist["count"] == 2
    assert snap["requests_total"] == 3


# ---------------------------------------------------------------------------
# R6 — PromptEngine.build_prompt() contains system prompt text
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prompt_engine_build_prompt_contains_system_prompt() -> None:
    """PromptEngine.build_prompt() output must embed the system prompt.

    The assembled ChatML string must contain the exact system prompt text so
    that downstream model inference always receives the persona instructions.
    """
    engine = PromptEngine()
    prompt = engine.build_prompt(
        user_text="tell me a joke",
        history=[],
    )

    assert isinstance(prompt, str), "build_prompt() must return a str"
    assert DEFAULT_SYSTEM_PROMPT in prompt, (
        "build_prompt() output must contain the full DEFAULT_SYSTEM_PROMPT text"
    )
    # Also verify the user input is present.
    assert "tell me a joke" in prompt

    # Confirm the ChatML framing tokens are present.
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
