"""Tests for src/core/state_machine.py — state transitions and observer hooks."""

import threading
import pytest

from src.core.state_machine import StateMachine, LumiState, InvalidTransitionError


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_initial_state_is_idle() -> None:
    sm = StateMachine()
    assert sm.current_state == LumiState.IDLE


@pytest.mark.unit
def test_custom_initial_state() -> None:
    sm = StateMachine(initial_state=LumiState.LISTENING)
    assert sm.current_state == LumiState.LISTENING


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("from_state,to_state", [
    (LumiState.IDLE,       LumiState.LISTENING),
    (LumiState.LISTENING,  LumiState.PROCESSING),
    (LumiState.LISTENING,  LumiState.IDLE),
    (LumiState.PROCESSING, LumiState.SPEAKING),
    (LumiState.PROCESSING, LumiState.IDLE),
    (LumiState.SPEAKING,   LumiState.IDLE),
])
def test_valid_transitions(from_state: LumiState, to_state: LumiState) -> None:
    sm = StateMachine(initial_state=from_state)
    sm.transition_to(to_state)
    assert sm.current_state == to_state


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("from_state,to_state", [
    (LumiState.IDLE,       LumiState.PROCESSING),
    (LumiState.IDLE,       LumiState.SPEAKING),
    (LumiState.IDLE,       LumiState.IDLE),
    (LumiState.LISTENING,  LumiState.SPEAKING),
    (LumiState.LISTENING,  LumiState.LISTENING),
    (LumiState.PROCESSING, LumiState.LISTENING),
    (LumiState.PROCESSING, LumiState.PROCESSING),
    (LumiState.SPEAKING,   LumiState.LISTENING),
    (LumiState.SPEAKING,   LumiState.PROCESSING),
    (LumiState.SPEAKING,   LumiState.SPEAKING),
])
def test_invalid_transitions_raise(from_state: LumiState, to_state: LumiState) -> None:
    sm = StateMachine(initial_state=from_state)
    with pytest.raises(InvalidTransitionError):
        sm.transition_to(to_state)


@pytest.mark.unit
def test_invalid_transition_does_not_change_state() -> None:
    sm = StateMachine()
    with pytest.raises(InvalidTransitionError):
        sm.transition_to(LumiState.PROCESSING)
    assert sm.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Observer hooks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_observer_fires_on_transition() -> None:
    sm = StateMachine()
    calls: list[tuple[LumiState, LumiState]] = []

    sm.register_observer(lambda old, new: calls.append((old, new)))
    sm.transition_to(LumiState.LISTENING)

    assert calls == [(LumiState.IDLE, LumiState.LISTENING)]


@pytest.mark.unit
def test_observer_fires_with_correct_states() -> None:
    sm = StateMachine()
    observed: list[tuple[LumiState, LumiState]] = []
    sm.register_observer(lambda o, n: observed.append((o, n)))

    sm.transition_to(LumiState.LISTENING)
    sm.transition_to(LumiState.PROCESSING)
    sm.transition_to(LumiState.IDLE)

    assert observed == [
        (LumiState.IDLE, LumiState.LISTENING),
        (LumiState.LISTENING, LumiState.PROCESSING),
        (LumiState.PROCESSING, LumiState.IDLE),
    ]


@pytest.mark.unit
def test_multiple_observers_all_fire() -> None:
    sm = StateMachine()
    results: list[int] = []

    sm.register_observer(lambda o, n: results.append(1))
    sm.register_observer(lambda o, n: results.append(2))
    sm.transition_to(LumiState.LISTENING)

    assert results == [1, 2]


@pytest.mark.unit
def test_observer_exception_does_not_break_transition() -> None:
    sm = StateMachine()

    def bad_observer(old: LumiState, new: LumiState) -> None:
        raise RuntimeError("observer error")

    sm.register_observer(bad_observer)
    sm.transition_to(LumiState.LISTENING)  # must not raise
    assert sm.current_state == LumiState.LISTENING


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.timeout(5)
def test_concurrent_reads_are_safe() -> None:
    sm = StateMachine()
    errors: list[Exception] = []

    def reader() -> None:
        for _ in range(1000):
            try:
                _ = sm.current_state
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
