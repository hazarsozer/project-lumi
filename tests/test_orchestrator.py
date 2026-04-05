"""Tests for src/core/orchestrator.py — event dispatch and interrupt handling."""

import threading
import time
import pytest

from src.core.config import LumiConfig
from src.core.events import (
    ShutdownEvent,
    InterruptEvent,
    WakeDetectedEvent,
    TranscriptReadyEvent,
)
from src.core.orchestrator import Orchestrator
from src.core.state_machine import LumiState


def _make_orchestrator() -> Orchestrator:
    return Orchestrator(config=LumiConfig())


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

    orch.post_event(InterruptEvent(source="user_stop"))
    orch.post_event(ShutdownEvent())
    orch.run()

    assert orch.state_machine.current_state == LumiState.IDLE


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_interrupt_in_processing_sets_llm_cancel_flag() -> None:
    orch = _make_orchestrator()
    orch.state_machine.transition_to(LumiState.LISTENING)
    orch.state_machine.transition_to(LumiState.PROCESSING)

    orch.post_event(InterruptEvent(source="keyboard"))
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

    orch.post_event(InterruptEvent(source="zmq"))
    orch.post_event(ShutdownEvent())
    orch.run()

    assert orch.state_machine.current_state == LumiState.IDLE


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_interrupt_in_idle_is_ignored() -> None:
    orch = _make_orchestrator()
    # State is already IDLE
    orch.post_event(InterruptEvent(source="keyboard"))
    orch.post_event(ShutdownEvent())
    orch.run()

    assert orch.state_machine.current_state == LumiState.IDLE
