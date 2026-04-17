"""
RED-phase tests for graceful runtime error recovery in the Orchestrator.

These tests verify that the state machine always returns to IDLE when
unexpected exceptions occur during inference — regardless of which call
inside _run_inference raises.

Failure modes covered:
1. ReasoningRouter.generate() raises RuntimeError       → must reach IDLE
2. memory.save() raises OSError (inside _llm_state_lock) → must reach IDLE
3. Same as #1 but via _handle_user_text                  → must reach IDLE
4. Post-crash IDLE state still accepts a new interrupt correctly (no-op)
5. A bad handler that raises must not crash the event loop

Tests 1–3 are expected RED (failing) until the implementation guard is added
around self._memory.save() in the success path of _run_inference.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig
from src.core.events import (
    InterruptEvent,
    LLMResponseReadyEvent,
    ShutdownEvent,
    TranscriptReadyEvent,
    UserTextEvent,
)
from src.core.orchestrator import Orchestrator
from src.core.state_machine import LumiState
from src.llm.reasoning_router import ReasoningRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator() -> Orchestrator:
    """Create an Orchestrator with a mock SpeakerThread so no audio device is needed."""
    mock_speaker = MagicMock(spec=SpeakerThread)
    return Orchestrator(config=LumiConfig(), speaker=mock_speaker)


def _wait_for_idle(orch: Orchestrator, timeout: float = 5.0) -> bool:
    """Block until the orchestrator's state machine reaches IDLE, or timeout.

    Registers a one-shot observer on the state machine and returns True when
    IDLE is reached, False on timeout.  The observer is set before any event
    is posted so transitions that happen quickly are still captured.
    """
    reached_idle = threading.Event()

    def _on_transition(old: LumiState, new: LumiState) -> None:
        if new == LumiState.IDLE:
            reached_idle.set()

    orch.state_machine.register_observer(_on_transition)
    # If already IDLE (transition happened before we registered), set immediately.
    if orch.state_machine.current_state == LumiState.IDLE:
        reached_idle.set()

    return reached_idle.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Test 1 — ReasoningRouter.generate() crashes via _handle_transcript
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(8)
def test_transcript_llm_crash_returns_to_idle() -> None:
    """RuntimeError from ReasoningRouter.generate() must transition state to IDLE.

    The existing except-Exception block in _run_inference (transcript path)
    already handles this case — the test documents and locks the behaviour so
    any future refactor cannot silently break it.

    RED trigger: if the except-Exception block is ever removed or narrowed,
    this test will fail.
    """
    orch = _make_orchestrator()

    reached_idle = threading.Event()

    def _on_transition(old: LumiState, new: LumiState) -> None:
        if new == LumiState.IDLE:
            reached_idle.set()

    orch.state_machine.register_observer(_on_transition)

    def _raise_runtime(text: str, cancel_flag: threading.Event, **kwargs: object) -> str:
        raise RuntimeError("GPU exploded")

    with patch.object(orch._reflex_router, "route", return_value=None), \
         patch.object(orch._reasoning_router, "generate", side_effect=_raise_runtime):
        orch.state_machine.transition_to(LumiState.LISTENING)

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.post_event(TranscriptReadyEvent(text="crash me"))

        assert reached_idle.wait(timeout=5.0), (
            "State machine never reached IDLE after RuntimeError from generate(); "
            "state is still: " + orch.state_machine.current_state.value
        )

        orch.post_event(ShutdownEvent())
        loop_thread.join(timeout=3.0)

    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Test 2 — memory.save() crashes in the SUCCESS path (inside _llm_state_lock)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(8)
def test_transcript_memory_save_crash_returns_to_idle() -> None:
    """OSError from memory.save() inside the _llm_state_lock success block must
    still transition the state machine to IDLE instead of leaving it stuck in
    PROCESSING.

    This is the primary unhandled failure mode: the save() call at line ~361 in
    orchestrator.py is NOT inside any try/except.  An exception there causes
    the daemon thread to exit silently, leaving state == PROCESSING forever.

    RED: this test will FAIL with the current implementation because the
    recovery guard is missing around self._memory.save() in the success path.
    """
    orch = _make_orchestrator()

    reached_idle = threading.Event()

    def _on_transition(old: LumiState, new: LumiState) -> None:
        if new == LumiState.IDLE:
            reached_idle.set()

    orch.state_machine.register_observer(_on_transition)

    def _ok_generate(text: str, cancel_flag: threading.Event, **kwargs: object) -> str:
        # Returns successfully — the crash happens later in memory.save()
        return "a fine response"

    with patch.object(orch._reflex_router, "route", return_value=None), \
         patch.object(orch._reasoning_router, "generate", side_effect=_ok_generate), \
         patch.object(orch._memory, "save", side_effect=OSError("disk full")):
        orch.state_machine.transition_to(LumiState.LISTENING)

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.post_event(TranscriptReadyEvent(text="store this"))

        assert reached_idle.wait(timeout=5.0), (
            "State machine never reached IDLE after OSError from memory.save(); "
            "state is still: " + orch.state_machine.current_state.value
        )

        orch.post_event(ShutdownEvent())
        loop_thread.join(timeout=3.0)

    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Test 3 — ReasoningRouter.generate() crashes via _handle_user_text
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(8)
def test_user_text_llm_crash_returns_to_idle() -> None:
    """RuntimeError from generate() in the _handle_user_text path must return
    to IDLE, mirroring the behaviour required for _handle_transcript.

    The existing except-Exception block in _handle_user_text._run_inference
    already handles this — the test locks the behaviour symmetrically with
    test_transcript_llm_crash_returns_to_idle.
    """
    orch = _make_orchestrator()

    reached_idle = threading.Event()

    def _on_transition(old: LumiState, new: LumiState) -> None:
        if new == LumiState.IDLE:
            reached_idle.set()

    orch.state_machine.register_observer(_on_transition)

    def _raise_runtime(text: str, cancel_flag: threading.Event, **kwargs: object) -> str:
        raise RuntimeError("model out of memory")

    with patch.object(orch._reflex_router, "route", return_value=None), \
         patch.object(orch._reasoning_router, "generate", side_effect=_raise_runtime):
        # _handle_user_text accepts events from IDLE; it performs the
        # IDLE→LISTENING→PROCESSING double-step internally.
        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.post_event(UserTextEvent(text="typed text crash"))

        assert reached_idle.wait(timeout=5.0), (
            "State machine never reached IDLE after RuntimeError from generate() "
            "via _handle_user_text; state is still: "
            + orch.state_machine.current_state.value
        )

        orch.post_event(ShutdownEvent())
        loop_thread.join(timeout=3.0)

    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Test 4 — post-crash IDLE still handles a new interrupt correctly (no-op)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(10)
def test_interrupt_during_idle_is_noop_after_crash_recovery() -> None:
    """After a crash-recovery cycle (state lands back in IDLE), an InterruptEvent
    received in IDLE is a no-op — state stays IDLE, no exception is raised.

    This test exercises the full recovery → new-event pipeline to confirm the
    orchestrator is not left in a broken state after handling a crash.
    """
    orch = _make_orchestrator()

    # --- Phase 1: trigger a crash and wait for IDLE recovery ---
    reached_idle_after_crash = threading.Event()

    def _on_transition(old: LumiState, new: LumiState) -> None:
        if new == LumiState.IDLE:
            reached_idle_after_crash.set()

    orch.state_machine.register_observer(_on_transition)

    def _ok_generate(text: str, cancel_flag: threading.Event, **kwargs: object) -> str:
        return "ok"

    with patch.object(orch._reflex_router, "route", return_value=None), \
         patch.object(orch._reasoning_router, "generate", side_effect=_ok_generate), \
         patch.object(orch._memory, "save", side_effect=OSError("disk full")):
        orch.state_machine.transition_to(LumiState.LISTENING)

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.post_event(TranscriptReadyEvent(text="crash trigger"))

        # Wait for state to recover to IDLE before sending the interrupt.
        assert reached_idle_after_crash.wait(timeout=5.0), (
            "Crash-recovery did not reach IDLE in time"
        )

    # --- Phase 2: post an InterruptEvent in IDLE, verify it is a no-op ---
    orch.post_event(InterruptEvent(source="test"))
    # Give the event loop time to process it.
    time.sleep(0.05)

    # State must still be IDLE (interrupt in IDLE is ignored, not an error).
    assert orch.state_machine.current_state == LumiState.IDLE

    orch.post_event(ShutdownEvent())
    loop_thread.join(timeout=3.0)

    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Test 5 — a bad handler that raises must not crash the event loop
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(5)
def test_dispatch_handler_exception_does_not_crash_loop() -> None:
    """A registered handler that raises an exception must not kill the event
    loop — subsequent events must still be dispatched.

    Verifies the existing try/except in _dispatch() catches per-handler
    exceptions and continues to the next handler / event.
    """
    orch = _make_orchestrator()

    processed_after_bad_handler: list[str] = []

    def _bad_handler(event: TranscriptReadyEvent) -> None:
        raise ValueError("intentional handler explosion")

    def _good_handler(event: UserTextEvent) -> None:
        processed_after_bad_handler.append(event.text)
        orch.post_event(ShutdownEvent())

    orch.register_handler(TranscriptReadyEvent, _bad_handler)
    orch.register_handler(UserTextEvent, _good_handler)

    # Post both events; the bad handler fires for the first, the loop must
    # survive and dispatch UserTextEvent next.
    orch.post_event(TranscriptReadyEvent(text="explode"))
    orch.post_event(UserTextEvent(text="survive"))

    # run() must return (triggered by ShutdownEvent posted from _good_handler).
    orch.run()

    assert processed_after_bad_handler == ["survive"], (
        "Event loop did not survive the bad handler or did not dispatch "
        "the subsequent UserTextEvent"
    )
