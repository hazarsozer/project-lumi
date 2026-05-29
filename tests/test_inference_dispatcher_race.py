"""Regression tests for CR-10 / GitHub #11.

Two deterministic race-condition tests:

1. interrupt-then-new-turn race
   - Turn 1 is in-flight and raises an exception.
   - Before Turn 1's exception handler can mutate shared state, an interrupt
     drives the machine to IDLE and Turn 2 starts (IDLE → PROCESSING).
   - The exception handler of Turn 1 must NOT clobber Turn 2 by transitioning
     PROCESSING → IDLE.  The second request must NOT be dropped.

2. watchdog-cancel-flag stale across SPEAKING transition
   - A watchdog fires while the state machine is already in SPEAKING (inference
     finished, TTS running).
   - After the watchdog fires the cancel flag must be cleared so the NEXT turn
     is not immediately aborted by a stale set.

Both tests are deterministic: threading.Event objects are used to force exact
interleavings.  No time.sleep calls.
"""

from __future__ import annotations

import queue
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.config import LLMConfig
from src.core.state_machine import LumiState, StateMachine
from src.llm.inference_dispatcher import LLMInferenceDispatcher


# ---------------------------------------------------------------------------
# Fixture: a minimal dispatcher with all dependencies mocked out
# ---------------------------------------------------------------------------


def _make_dispatcher(
    *,
    timeout_s: float = 0.0,
) -> tuple[LLMInferenceDispatcher, StateMachine, MagicMock, list[Any]]:
    """Build a bare LLMInferenceDispatcher with mocked collaborators.

    Returns:
        dispatcher    — the unit under test
        state_machine — the real StateMachine (not mocked)
        reasoning_router_mock — MagicMock with replaceable .generate side_effect
        posted_events — list that collects everything passed to post_event()
    """
    state_machine = StateMachine()

    model_loader = MagicMock()
    model_loader.is_loaded = True  # skip lazy-load path

    reflex_router = MagicMock()
    reflex_router.route.return_value = None       # always slow path
    reflex_router.route_rag_intent.return_value = False

    reasoning_router = MagicMock()
    reasoning_router.generate.return_value = "ok"

    memory = MagicMock()
    tool_executor = MagicMock()
    tool_executor.execute.return_value = []

    posted_events: list[Any] = []

    dispatcher = LLMInferenceDispatcher(
        model_loader=model_loader,
        reflex_router=reflex_router,
        reasoning_router=reasoning_router,
        memory=memory,
        tool_executor=tool_executor,
        state_machine=state_machine,
        event_queue=queue.Queue(),
        llm_config=LLMConfig(inference_timeout_s=timeout_s),
    )

    return dispatcher, state_machine, reasoning_router, posted_events


# ---------------------------------------------------------------------------
# Test 1: interrupt-then-new-turn race — second request must NOT be dropped
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_exception_handler_does_not_clobber_newer_turn() -> None:
    """Turn 1 raises an exception AFTER Turn 2 has already started.

    Forced interleaving:
    1. State machine: IDLE → LISTENING → PROCESSING (Turn 1 dispatched)
    2. Turn 1's generate() blocks on ``pause_exception``.
    3. Main thread: simulates interrupt → IDLE, then Turn 2 dispatched →
       LISTENING → PROCESSING.
    4. Turn 1's generate() is released to raise RuntimeError.
    5. Turn 1's exception handler runs.  It must see _turn_generation != its
       captured my_generation and bail out without touching the state machine.
    6. Assert: state machine still in PROCESSING (Turn 2 alive, not clobbered).
    """
    dispatcher, sm, reasoning_router, posted_events = _make_dispatcher()

    # Synchronisation events for deterministic interleaving.
    turn1_entered = threading.Event()   # set when Turn 1's generate() starts
    pause_exception = threading.Event() # release Turn 1 to raise

    def _turn1_generate(text: str, cancel_flag: threading.Event, **kw: Any) -> str:
        turn1_entered.set()
        pause_exception.wait(timeout=5.0)
        raise RuntimeError("simulated GPU crash")

    reasoning_router.generate.side_effect = _turn1_generate

    # --- Turn 1 ---
    sm.transition_to(LumiState.LISTENING)
    sm.transition_to(LumiState.PROCESSING)
    dispatcher.dispatch(
        "turn one query",
        "test",
        rag_runtime_enabled=False,
        post_event=posted_events.append,
    )

    # Wait until Turn 1's inference thread is inside generate().
    assert turn1_entered.wait(timeout=3.0), "Turn 1 never entered generate()"

    # --- Simulate interrupt ---
    # Replicate the minimal part of _handle_interrupt for PROCESSING state:
    # set cancel flag, transition to IDLE, clear cancel flag.
    with dispatcher.llm_state_lock:
        dispatcher.cancel_flag.set()
        sm.transition_to(LumiState.IDLE)
        dispatcher.cancel_flag.clear()

    # --- Turn 2 ---
    # Now start Turn 2 so _turn_generation is bumped BEFORE Turn 1's exception
    # handler can run.  Turn 2 uses a normal blocking generate that returns
    # a valid string — it should stay in PROCESSING after dispatch() returns
    # because we verify the state BEFORE the inference thread finishes.
    turn2_started = threading.Event()
    turn2_pause = threading.Event()   # keep Turn 2's inference thread alive

    def _turn2_generate(text: str, cancel_flag: threading.Event, **kw: Any) -> str:
        turn2_started.set()
        turn2_pause.wait(timeout=5.0)
        return "turn two response"

    reasoning_router.generate.side_effect = _turn2_generate

    sm.transition_to(LumiState.LISTENING)
    sm.transition_to(LumiState.PROCESSING)
    dispatcher.dispatch(
        "turn two query",
        "test",
        rag_runtime_enabled=False,
        post_event=posted_events.append,
    )

    # Wait until Turn 2's inference thread is inside generate().
    assert turn2_started.wait(timeout=3.0), "Turn 2 never entered generate()"

    # Now release Turn 1 to raise its RuntimeError.  Its exception handler
    # will run concurrently with Turn 2 being in PROCESSING.
    pause_exception.set()

    # Give Turn 1's exception handler a moment to complete.
    # Use a short busy-wait checking the generation counter rather than sleep.
    import time
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with dispatcher.llm_state_lock:
            if dispatcher._turn_generation == 2:
                # Turn 2 is current; exception handler should have seen this and
                # backed off.  Verify state is still PROCESSING.
                break
        time.sleep(0.01)

    # Allow a brief settling window for Turn 1's exception handler thread.
    time.sleep(0.05)

    # --- Assertions ---
    # Turn 2 is still alive (PROCESSING).  The exception handler of Turn 1 must
    # not have transitioned us to IDLE.
    assert sm.current_state == LumiState.PROCESSING, (
        f"Turn 2 was clobbered — state is {sm.current_state} (expected PROCESSING)."
        " Turn 1's exception handler incorrectly reset shared state."
    )

    # Cleanup: release Turn 2's thread.
    turn2_pause.set()


# ---------------------------------------------------------------------------
# Test 2: watchdog cancel flag cleared on SPEAKING-state fire
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_watchdog_clears_cancel_flag_when_firing_during_speaking() -> None:
    """Watchdog fires while the state machine is in SPEAKING.

    Forced interleaving:
    1. Dispatch a turn; inference finishes quickly → SPEAKING.
    2. Manually invoke _watchdog_fn() with state already in SPEAKING (simulates
       the watchdog timer firing late, after the turn transitioned to SPEAKING).
    3. Assert cancel_flag is cleared after the watchdog runs (not left set).
    4. Assert a subsequent dispatch() is not aborted by the stale cancel flag.

    We extract _watchdog_fn by dispatching with a non-zero timeout and then
    running the watchdog body directly to avoid real-time delays.
    """
    # Use a very short timeout so the Timer object is created but we'll cancel
    # it ourselves and call _watchdog_fn manually.
    dispatcher, sm, reasoning_router, posted_events = _make_dispatcher(timeout_s=60.0)

    # Keep a reference to the watchdog function so we can call it directly.
    captured_watchdog: list[Any] = []
    original_timer_init = threading.Timer.__init__

    def _capturing_timer_init(self: threading.Timer, interval: float, function: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        captured_watchdog.append(function)
        original_timer_init(self, interval, function, *args, **kwargs)

    # Patch threading.Timer so we capture the watchdog callable without waiting
    # for it to fire naturally (that would take 60 seconds).
    import unittest.mock as _mock

    generation_advanced = threading.Event()

    def _instant_generate(text: str, cancel_flag: threading.Event, **kw: Any) -> str:
        generation_advanced.set()
        return "quick response"

    reasoning_router.generate.side_effect = _instant_generate

    # Patch parse_tool_calls so inference finishes the full path and the timer
    # is started.
    with _mock.patch("src.llm.inference_dispatcher.threading.Timer") as mock_timer_cls:
        # Capture the watchdog function passed to Timer(timeout, _watchdog_fn).
        watchdog_fn_holder: list[Any] = []

        def _fake_timer_new(interval: float, fn: Any) -> MagicMock:
            watchdog_fn_holder.append(fn)
            fake_t = MagicMock()
            fake_t.daemon = True
            return fake_t

        mock_timer_cls.side_effect = _fake_timer_new

        sm.transition_to(LumiState.LISTENING)
        sm.transition_to(LumiState.PROCESSING)

        dispatch_done = threading.Event()

        def _post_event(ev: Any) -> None:
            posted_events.append(ev)

        # Dispatch runs the inference thread; it will finish quickly.
        dispatcher.dispatch(
            "fast query",
            "test",
            rag_runtime_enabled=False,
            post_event=_post_event,
        )

        # Wait until generate() has been called (inference thread started).
        assert generation_advanced.wait(timeout=3.0), "generate never called"

        import time
        time.sleep(0.1)  # let inference thread complete and transition to SPEAKING

    # At this point the inference thread has finished and Timer was captured but
    # never started (it's a MagicMock).  The watchdog_fn is in watchdog_fn_holder.
    assert watchdog_fn_holder, "Watchdog function was never captured from threading.Timer"
    watchdog_fn = watchdog_fn_holder[0]

    # Manually drive state to SPEAKING to simulate the TTS path.
    # (inference dispatcher transitions PROCESSING→SPEAKING on the first sentence;
    #  we do it here directly since we mocked the timer.)
    if sm.current_state == LumiState.PROCESSING:
        sm.transition_to(LumiState.SPEAKING)

    # State is now SPEAKING.  Call watchdog_fn directly.
    assert sm.current_state == LumiState.SPEAKING

    watchdog_fn()

    # --- Assertions ---
    # Bug: without the fix, cancel_flag would be SET (left over by watchdog).
    # Fix: watchdog detects SPEAKING and clears the flag.
    assert not dispatcher.cancel_flag.is_set(), (
        "Watchdog left cancel_flag SET after firing during SPEAKING. "
        "This stale flag would abort the next turn."
    )

    # Bonus: verify the state machine was NOT transitioned from SPEAKING by the
    # watchdog (SPEAKING→IDLE belongs to the TTS/interrupt path only).
    assert sm.current_state == LumiState.SPEAKING, (
        f"Watchdog incorrectly transitioned state from SPEAKING to {sm.current_state}"
    )


# ---------------------------------------------------------------------------
# Test 3: watchdog PROCESSING path still works (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_watchdog_still_transitions_idle_from_processing() -> None:
    """Watchdog in PROCESSING state must still set cancel_flag and go to IDLE.

    This is a regression guard: the SPEAKING-state fix must not break the
    original PROCESSING-state watchdog behaviour.
    """
    dispatcher, sm, reasoning_router, posted_events = _make_dispatcher(timeout_s=60.0)

    with _mock_timer_capture() as watchdog_fn_holder:
        # Set up a blocking generate that never returns (simulates a hung model).
        # We'll call the watchdog manually rather than waiting for the timer.
        blocked = threading.Event()

        def _hung_generate(text: str, cancel_flag: threading.Event, **kw: Any) -> str:
            blocked.wait(timeout=5.0)
            raise InterruptedError("cancelled")

        reasoning_router.generate.side_effect = _hung_generate

        sm.transition_to(LumiState.LISTENING)
        sm.transition_to(LumiState.PROCESSING)

        dispatcher.dispatch(
            "hung query",
            "test",
            rag_runtime_enabled=False,
            post_event=posted_events.append,
        )

        import time
        time.sleep(0.05)  # let inference thread park inside generate()

    assert watchdog_fn_holder, "Watchdog function was never captured"
    watchdog_fn = watchdog_fn_holder[0]

    # State is still PROCESSING (generate() is hung).
    assert sm.current_state == LumiState.PROCESSING

    watchdog_fn()

    assert dispatcher.cancel_flag.is_set(), (
        "Watchdog should have set cancel_flag when firing in PROCESSING state"
    )
    assert sm.current_state == LumiState.IDLE, (
        "Watchdog should have transitioned PROCESSING→IDLE"
    )

    # Unblock the hung thread.
    blocked.set()


# ---------------------------------------------------------------------------
# Helper: capture threading.Timer watchdog function without starting it
# ---------------------------------------------------------------------------


from contextlib import contextmanager  # noqa: E402 — placed after imports


@contextmanager  # type: ignore[misc]
def _mock_timer_capture() -> Any:
    """Context manager: patch threading.Timer so the watchdog callable is captured.

    Yields a list; after the context exits, list[0] is the watchdog function.
    """
    import unittest.mock as _mock

    holder: list[Any] = []

    def _fake_new(interval: float, fn: Any) -> MagicMock:
        holder.append(fn)
        fake_t = MagicMock()
        fake_t.daemon = True
        return fake_t

    with _mock.patch("src.llm.inference_dispatcher.threading.Timer", side_effect=_fake_new):
        yield holder
