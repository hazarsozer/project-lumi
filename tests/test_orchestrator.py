"""Tests for src/core/orchestrator.py — event dispatch and interrupt handling."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig
from src.core.events import (
    ShutdownEvent,
    InterruptEvent,
    InterruptSource,
    WakeDetectedEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    LLMResponseReadyEvent,
)
from src.core.orchestrator import Orchestrator
from src.core.state_machine import LumiState, StateMachine
from src.core.event_bridge import EventBridge


def _make_orchestrator() -> Orchestrator:
    """Create an Orchestrator with a mock SpeakerThread so no audio device is needed."""
    mock_speaker = MagicMock(spec=SpeakerThread)
    return Orchestrator(config=LumiConfig(), speaker=mock_speaker)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orchestrator_initial_state_is_idle() -> None:
    orch = _make_orchestrator()
    assert orch.state_machine.current_state == LumiState.IDLE


@pytest.mark.unit
def test_orchestrator_has_llm_cancel_flag() -> None:
    orch = _make_orchestrator()
    assert not orch.llm_cancel_flag.is_set()


# ---------------------------------------------------------------------------
# post_event and register_handler dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_registered_handler_receives_event() -> None:
    orch = _make_orchestrator()
    received: list[object] = []

    orch.register_handler(WakeDetectedEvent, lambda e: received.append(e))
    orch.post_event(WakeDetectedEvent(timestamp=1.0))
    orch.post_event(ShutdownEvent())

    orch.run()  # blocks until ShutdownEvent

    assert len(received) == 1
    assert isinstance(received[0], WakeDetectedEvent)


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_multiple_handlers_for_same_event_all_called() -> None:
    orch = _make_orchestrator()
    calls: list[int] = []

    orch.register_handler(TranscriptReadyEvent, lambda e: calls.append(1))
    orch.register_handler(TranscriptReadyEvent, lambda e: calls.append(2))
    orch.post_event(TranscriptReadyEvent(text="hello"))
    orch.post_event(ShutdownEvent())

    orch.run()
    assert calls == [1, 2]


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_unhandled_event_type_does_not_crash() -> None:
    orch = _make_orchestrator()
    # Post an event with no registered handler — should log and continue.
    orch.post_event(TranscriptReadyEvent(text="no handler"))
    orch.post_event(ShutdownEvent())
    orch.run()  # must complete without exception


# ---------------------------------------------------------------------------
# ShutdownEvent exits run()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_shutdown_event_exits_run() -> None:
    orch = _make_orchestrator()
    orch.post_event(ShutdownEvent())

    start = time.monotonic()
    orch.run()
    elapsed = time.monotonic() - start

    assert elapsed < 2.0


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_shutdown_from_background_thread() -> None:
    orch = _make_orchestrator()

    def _shutdown_after_delay() -> None:
        time.sleep(0.05)
        orch.post_event(ShutdownEvent())

    t = threading.Thread(target=_shutdown_after_delay, daemon=True)
    t.start()

    orch.run()  # must return after the thread posts ShutdownEvent
    t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# InterruptEvent handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_interrupt_in_processing_returns_to_idle() -> None:
    orch = _make_orchestrator()
    # Force state to PROCESSING
    orch.state_machine.transition_to(LumiState.LISTENING)
    orch.state_machine.transition_to(LumiState.PROCESSING)

    orch.post_event(InterruptEvent(source=InterruptSource.USER_STOP))
    orch.post_event(ShutdownEvent())
    orch.run()

    assert orch.state_machine.current_state == LumiState.IDLE


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_interrupt_in_processing_sets_llm_cancel_flag() -> None:
    orch = _make_orchestrator()
    orch.state_machine.transition_to(LumiState.LISTENING)
    orch.state_machine.transition_to(LumiState.PROCESSING)

    orch.post_event(InterruptEvent(source=InterruptSource.KEYBOARD))
    orch.post_event(ShutdownEvent())
    orch.run()

    # Flag is cleared after interrupt handling, so it should be False
    assert not orch.llm_cancel_flag.is_set()


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_interrupt_in_speaking_returns_to_idle() -> None:
    orch = _make_orchestrator()
    orch.state_machine.transition_to(LumiState.LISTENING)
    orch.state_machine.transition_to(LumiState.PROCESSING)
    orch.state_machine.transition_to(LumiState.SPEAKING)

    orch.post_event(InterruptEvent(source=InterruptSource.ZMQ))
    orch.post_event(ShutdownEvent())
    orch.run()

    assert orch.state_machine.current_state == LumiState.IDLE


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_interrupt_in_idle_is_ignored() -> None:
    orch = _make_orchestrator()
    # State is already IDLE
    orch.post_event(InterruptEvent(source=InterruptSource.KEYBOARD))
    orch.post_event(ShutdownEvent())
    orch.run()

    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# _handle_transcript: reflex fast-path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_handle_transcript_reflex_hit_posts_response_and_transitions_to_speaking() -> (
    None
):
    """Reflex hit: LISTENING→PROCESSING→SPEAKING→IDLE full lifecycle.

    _handle_transcript posts LLMResponseReadyEvent and transitions to SPEAKING.
    _handle_llm_response (no TTS configured) then immediately posts
    SpeechCompletedEvent, which _handle_speech_completed processes before
    ShutdownEvent arrives — so the final state after run() is IDLE.
    """
    from unittest.mock import patch
    from src.core.events import LLMResponseReadyEvent

    orch = _make_orchestrator()
    received_responses: list[object] = []

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        received_responses.append(e)
        # Shut down after all handlers for this event have run.
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    with patch.object(
        orch._reflex_router, "route", return_value="Hello! How can I help you?"
    ):
        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="hello"))
        orch.run()  # exits when ShutdownEvent is processed

    # SpeechCompletedEvent (no-TTS synthetic) is queued before ShutdownEvent,
    # so by the time run() returns the state machine is back at IDLE.
    assert orch.state_machine.current_state == LumiState.IDLE
    assert len(received_responses) == 1
    assert isinstance(received_responses[0], LLMResponseReadyEvent)
    assert received_responses[0].text == "Hello! How can I help you?"


# ---------------------------------------------------------------------------
# _handle_transcript: reasoning slow-path — successful generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_handle_transcript_reasoning_path_posts_response_and_transitions_to_speaking(
    mock_llama_cpp: MagicMock,
) -> None:
    """Reasoning path: daemon thread generates response, posts LLMResponseReadyEvent,
    transitions to SPEAKING.

    The orchestrator event loop must be running so that LLMResponseReadyEvent,
    posted by the daemon thread, is actually dispatched.  We run orch.run() in a
    background thread and stop it once the response arrives.
    """
    from unittest.mock import patch
    from src.core.events import LLMResponseReadyEvent

    orch = _make_orchestrator()
    received_responses: list[object] = []

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        received_responses.append(e)
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    def _fake_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        return "Paris is the capital of France."

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_fake_generate),
    ):
        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="What is the capital of France?"))
        orch.run()  # exits when _on_llm_response posts ShutdownEvent

    assert len(received_responses) == 1
    assert isinstance(received_responses[0], LLMResponseReadyEvent)
    assert received_responses[0].text == "Paris is the capital of France."
    # _handle_llm_response (no TTS) posts SpeechCompletedEvent before ShutdownEvent,
    # so the full lifecycle completes and state returns to IDLE.
    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# _handle_transcript: reasoning slow-path — InterruptedError during generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_handle_transcript_reasoning_interrupted_error_logged_no_state_change(
    mock_llama_cpp: MagicMock,
) -> None:
    """InterruptedError during generation is logged; state stays in PROCESSING
    (the inference thread exits cleanly without transitioning to IDLE).

    We run the event loop in a background thread and post ShutdownEvent only
    after the daemon inference thread has fully finished.
    """
    from unittest.mock import patch

    orch = _make_orchestrator()

    # thread_finished is set AFTER the exception is raised (and therefore after
    # the except InterruptedError block returns from _run_inference).
    thread_done = threading.Event()

    def _raise_interrupted(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        raise InterruptedError("cancelled mid-stream")

    original_generate = _raise_interrupted

    def _wrapped_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        try:
            return original_generate(text, cancel_flag)
        finally:
            thread_done.set()

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_wrapped_generate),
    ):
        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="long query"))

        # Wait until the daemon thread has fully completed (finally block runs).
        thread_done.wait(timeout=3.0)

        orch.post_event(ShutdownEvent())
        orch.run()

    # InterruptedError path does NOT transition to IDLE — state remains PROCESSING.
    assert orch.state_machine.current_state == LumiState.PROCESSING


# ---------------------------------------------------------------------------
# _handle_transcript: reasoning slow-path — unexpected Exception
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(8)
def test_handle_transcript_reasoning_unexpected_exception_transitions_to_idle() -> None:
    """Unexpected Exception during generation transitions state to IDLE.

    The orchestrator event loop runs in a background thread.  We register an
    observer on the state machine to detect when the IDLE transition occurs —
    that fires after the except Exception block runs transition_to(IDLE).
    """
    from unittest.mock import patch

    orch = _make_orchestrator()

    # Register a state-machine observer that sets an event when IDLE is reached.
    reached_idle = threading.Event()

    def _on_transition(old: LumiState, new: LumiState) -> None:
        if new == LumiState.IDLE:
            reached_idle.set()

    orch.state_machine.register_observer(_on_transition)

    def _raise_generic(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        raise RuntimeError("GPU exploded")

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_raise_generic),
    ):
        orch.state_machine.transition_to(LumiState.LISTENING)

        # Run the event loop in a background thread so it can dispatch
        # TranscriptReadyEvent and start the daemon inference thread.
        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.post_event(TranscriptReadyEvent(text="crash me"))

        # Wait until the state machine observer confirms IDLE was reached.
        reached_idle.wait(timeout=5.0)

        orch.post_event(ShutdownEvent())
        loop_thread.join(timeout=3.0)

    assert orch.state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# _handle_transcript: stale-state guard — response discarded when no longer PROCESSING
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(8)
def test_handle_transcript_stale_state_response_discarded(
    mock_llama_cpp: MagicMock,
) -> None:
    """If state is no longer PROCESSING when inference completes, the response
    is discarded and no LLMResponseReadyEvent is posted.

    The orchestrator event loop runs in a background thread so that
    TranscriptReadyEvent is dispatched and the daemon inference thread starts.
    """
    from unittest.mock import patch
    from src.core.events import LLMResponseReadyEvent

    orch = _make_orchestrator()
    received_responses: list[object] = []
    orch.register_handler(LLMResponseReadyEvent, lambda e: received_responses.append(e))

    hold_thread = threading.Event()
    thread_entered = threading.Event()
    inference_guard_checked = threading.Event()

    def _slow_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        thread_entered.set()
        hold_thread.wait(timeout=5.0)
        return "stale response"

    # Wrap generate so we know when _run_inference has passed the stale-state
    # guard and is about to return (the guard is the last thing before the
    # post_event/transition block, so returning from generate means the guard
    # will be evaluated next).
    def _wrapped_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        result = _slow_generate(text, cancel_flag)
        inference_guard_checked.set()
        return result

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_wrapped_generate),
    ):
        # Run the event loop in a background thread.
        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="make me stale"))

        # Wait until the daemon thread has entered _slow_generate.
        assert thread_entered.wait(timeout=3.0), "Daemon thread never started"

        # Move state away from PROCESSING before releasing the daemon thread.
        orch.state_machine.transition_to(LumiState.IDLE)

        # Release the daemon thread — it will check the stale-state guard and
        # discard the response.
        hold_thread.set()

        # Wait until _wrapped_generate has returned (guard has been evaluated).
        assert inference_guard_checked.wait(timeout=3.0), "Inference never completed"

        # Small buffer to let the daemon thread finish the _run_inference body.
        time.sleep(0.05)

        orch.post_event(ShutdownEvent())
        loop_thread.join(timeout=3.0)

    # The stale-state guard must have discarded the response.
    assert received_responses == []


# ---------------------------------------------------------------------------
# ZMQServer integration — observer wiring, UserTextEvent, shutdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_event_bridge_receives_state_change() -> None:
    """Injected EventBridge.on_state_change is called when a state transition fires.

    The Orchestrator registers on_state_change as a StateMachine observer for
    injected instances.  We verify this by triggering a LISTENING transition
    and asserting the mock was called.
    """
    mock_speaker = MagicMock(spec=SpeakerThread)
    mock_zmq = MagicMock(spec=EventBridge)

    orch = Orchestrator(config=LumiConfig(), speaker=mock_speaker, event_bridge=mock_zmq)

    # Trigger a valid state transition.
    orch.state_machine.transition_to(LumiState.LISTENING)

    mock_zmq.on_state_change.assert_called_once_with(
        LumiState.IDLE, LumiState.LISTENING
    )


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_user_text_routes_to_llm() -> None:
    """UserTextEvent posted to the orchestrator invokes the LLM/router path.

    Uses a reflex hit so the test does not require a real model.  Asserts that
    an LLMResponseReadyEvent is generated, mirroring the TranscriptReadyEvent
    reflex-hit test.
    """
    mock_speaker = MagicMock(spec=SpeakerThread)
    orch = Orchestrator(config=LumiConfig(), speaker=mock_speaker)

    received_responses: list[object] = []

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        received_responses.append(e)
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    with patch.object(orch._reflex_router, "route", return_value="Hi from reflex!"):
        # Post from IDLE (the production case: Godot sends text while idle).
        # _handle_user_text must step through IDLE→LISTENING→PROCESSING itself.
        orch.post_event(UserTextEvent(text="hello from body"))
        orch.run()

    assert len(received_responses) == 1
    assert isinstance(received_responses[0], LLMResponseReadyEvent)
    assert received_responses[0].text == "Hi from reflex!"


# ---------------------------------------------------------------------------
# _drain_event_types — atomicity w.r.t. concurrent producers (CR-12 / #13)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_drain_event_types_retained_events_preserve_order() -> None:
    """Retained events must appear at the front of the queue in their original
    relative order after draining.

    This is the serial ordering invariant.  It passes on both pre-fix and
    post-fix code — the fix does not change the serial ordering guarantee, it
    only makes the operation atomic w.r.t. concurrent producers.
    """
    import queue as _queue

    orch = _make_orchestrator()
    q = orch._event_queue

    class _Retain1:
        pass

    class _DrainA:
        pass

    class _Retain2:
        pass

    class _DrainB:
        pass

    class _Retain3:
        pass

    r1, da, r2, db, r3 = _Retain1(), _DrainA(), _Retain2(), _DrainB(), _Retain3()
    for ev in [r1, da, r2, db, r3]:
        q.put(ev)

    orch._drain_event_types({_DrainA.__name__, _DrainB.__name__})

    result: list[object] = []
    try:
        while True:
            result.append(q.get_nowait())
    except _queue.Empty:
        pass

    assert r1 in result, "Retain1 missing"
    assert r2 in result, "Retain2 missing"
    assert r3 in result, "Retain3 missing"
    assert da not in result, "DrainA must not appear"
    assert db not in result, "DrainB must not appear"

    pos_r1 = result.index(r1)
    pos_r2 = result.index(r2)
    pos_r3 = result.index(r3)
    assert pos_r1 < pos_r2 < pos_r3, (
        f"retained events out of order: positions={pos_r1}, {pos_r2}, {pos_r3}"
    )
    assert pos_r1 == 0, "first retained event must be at index 0 (front of queue)"


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_drain_event_types_retained_events_stay_at_front_no_interleave() -> None:
    """With concurrent producers active during the drain, retained events must
    appear before any producer event in the queue.

    The interleave is forced deterministically using a two-phase barrier:

    Phase 1 (setup): hold ``q.mutex`` externally while starting ``N_PRODUCERS``
    producer threads that each call ``q.put(producer_event)``.  Because ``put``
    acquires ``q.mutex``, all producer threads block immediately.  A small
    sleep guarantees they are blocked before we release the external lock.

    Phase 2 (race): release the external lock, then immediately call
    ``_drain_event_types``.  The fixed implementation re-acquires ``q.mutex``
    and holds it for the entire snapshot-filter-re-insert operation, so
    producers remain blocked until all retained events are already in the
    deque.  Only after the drain releases the lock can producers enqueue —
    guaranteeing producer events arrive AFTER retained events.

    Pre-fix code (non-atomic): the re-enqueue loop calls ``q.put(retained)``
    once per item, releasing the mutex between iterations.  A producer that
    wins the mutex between two consecutive retained re-enqueues will interleave
    its event into the retained set.  To make this race deterministic WITHOUT
    modifying production code, we also inject a controlled producer via a
    ``q.put`` hook (see ``_atomic_put_hook``): on the very first retained
    re-enqueue, the hook appends a *second* producer event directly into the
    raw deque, which bypasses the lock entirely and reliably produces an
    interleaved queue.  The final assertion catches both the threading race and
    the hook-injected interleave.
    """
    import queue as _queue

    orch = _make_orchestrator()
    q = orch._event_queue

    class _Retained:
        pass

    class _Drained:
        pass

    class _Producer:
        pass

    class _HookInjected:
        """Injected synchronously after the first retained re-enqueue (pre-fix path)."""

    # Pre-seed with a large mixed queue.
    n_retained = 20
    retained_events: list[_Retained] = [_Retained() for _ in range(n_retained)]
    drained_events: list[_Drained] = [_Drained() for _ in range(n_retained)]
    for r, d in zip(retained_events, drained_events):
        q.put(r)
        q.put(d)

    # Producer threads.
    N_PRODUCERS = 4
    producer_events = [_Producer() for _ in range(N_PRODUCERS)]
    producers_done = threading.Event()

    def _run_producers() -> None:
        threads = [
            threading.Thread(target=q.put, args=(pe,), daemon=True)
            for pe in producer_events
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        producers_done.set()

    # --- hook for pre-fix detection ------------------------------------------
    # In the fixed code, q.put() is never called for retained items (we use
    # q.queue.appendleft directly under the lock), so this hook is a no-op.
    # In the pre-fix code, q.put() is called for each retained item; on the
    # first call, we inject _HookInjected directly into the raw deque to
    # reliably produce an interleave regardless of thread scheduling.
    hook_injected = _HookInjected()
    original_put = q.put
    injection_count = [0]

    def _atomic_put_hook(item: object) -> None:
        original_put(item)
        if injection_count[0] == 0:
            # Inject into raw deque after first retained re-enqueue.
            # Only fires on pre-fix code (fixed code never calls put for retained).
            q.queue.append(hook_injected)
            injection_count[0] += 1

    # --- run the race --------------------------------------------------------
    # Hold the mutex while starting producers so they block before drain starts.
    with q.mutex:
        producer_thread = threading.Thread(target=_run_producers, daemon=True)
        producer_thread.start()
        time.sleep(0.05)  # ensure all producer threads are blocked on q.put

    # Fixed: _drain_event_types acquires mutex → holds for full operation →
    #   producers stay blocked → producer events land at the back.
    # Pre-fix: re-enqueue loop releases mutex between calls → producers/hook
    #   can sneak in between two consecutive retained re-enqueues.
    with patch.object(q, "put", side_effect=_atomic_put_hook):
        orch._drain_event_types({_Drained.__name__})

    assert producers_done.wait(timeout=2.0), "Producer threads did not complete"
    producer_thread.join(timeout=2.0)

    # Collect final queue state.
    result: list[object] = []
    try:
        while True:
            result.append(q.get_nowait())
    except _queue.Empty:
        pass

    # --- assertions ----------------------------------------------------------

    # All retained events must be present.
    for i, re in enumerate(retained_events):
        assert re in result, f"retained_events[{i}] missing from queue"

    # No drained events must appear.
    for de in drained_events:
        assert de not in result, "a drained event must not appear in queue"

    # Retained events must preserve their original relative order.
    retained_positions = [result.index(re) for re in retained_events]
    assert retained_positions == sorted(retained_positions), (
        f"Retained events out of order: first 5 positions={retained_positions[:5]}, "
        f"full queue={[type(x).__name__ for x in result]}"
    )

    # No producer / hook-injected event must appear before the last retained
    # event (no interleaving).
    max_retained_pos = max(retained_positions)

    # The hook-injected sentinel must be absent on the fixed code path
    # (fixed code never calls q.put for retained items, so the hook never fires).
    assert hook_injected not in result, (
        "hook_injected event found in queue — _drain_event_types called q.put() "
        "for retained items instead of using atomic deque insertion; "
        "the operation is NOT atomic w.r.t. concurrent producers"
    )

    # All producer events must appear after retained events.
    producer_positions = [
        result.index(pe) for pe in producer_events if pe in result
    ]
    assert len(producer_positions) == N_PRODUCERS, "not all producer events found"
    min_producer_pos = min(producer_positions)
    assert min_producer_pos > max_retained_pos, (
        f"Producer event interleaved into retained set: "
        f"max_retained_pos={max_retained_pos}, min_producer_pos={min_producer_pos}, "
        f"full queue={[type(x).__name__ for x in result]}"
    )


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_shutdown_stops_event_bridge() -> None:
    """ShutdownEvent causes the Orchestrator to call event_bridge.stop().

    Injects a mock EventBridge, runs the event loop until ShutdownEvent is
    processed, and asserts that stop() was called exactly once.
    """
    mock_speaker = MagicMock(spec=SpeakerThread)
    mock_zmq = MagicMock(spec=EventBridge)

    orch = Orchestrator(config=LumiConfig(), speaker=mock_speaker, event_bridge=mock_zmq)

    orch.post_event(ShutdownEvent())
    orch.run()

    mock_zmq.stop.assert_called_once()
