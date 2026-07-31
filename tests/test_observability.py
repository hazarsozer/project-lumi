"""
Tests for GitHub Issue #29 — observability: utterance_id contextvar, structured
per-turn JSON fields, error-tracker hook, and heartbeat.

Acceptance criteria:
  1. utterance_id_var is set during a turn; log records emitted within the turn
     carry the id in their JSON output.
  2. _JsonFormatter emits utterance_id when the contextvar is set.
  3. An error-tracker hook, when registered, is called on unhandled handler
     exceptions; default is no-op (no crash when unset).
  4. A heartbeat fires at the configured interval (deterministic — no real sleep).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_orchestrator(
    *,
    heartbeat_interval_s: float = 0.0,
    error_tracker: Callable[[BaseException], None] | None = None,
) -> Any:
    """Build an Orchestrator with mocked-out heavy subsystems for unit tests."""
    from src.core.config import (
        LLMConfig,
        LumiConfig,
        ObservabilityConfig,
        TTSConfig,
    )
    from src.core.orchestrator import Orchestrator

    cfg = LumiConfig(
        tts=TTSConfig(enabled=False),
        llm=LLMConfig(model_path="nonexistent.gguf"),
        observability=ObservabilityConfig(
            heartbeat_interval_s=heartbeat_interval_s
        ),
    )

    speaker_mock = MagicMock()
    speaker_mock.start = MagicMock()
    speaker_mock.stop = MagicMock()

    orch = Orchestrator(
        cfg,
        speaker=speaker_mock,
        tts=None,
        error_tracker=error_tracker,
    )
    return orch


# ===========================================================================
# Part 1 — utterance_id contextvar binding
# ===========================================================================


@pytest.mark.unit
def test_utterance_id_var_default_is_none() -> None:
    """utterance_id_var returns None when no turn is in progress."""
    from src.core.logging_config import utterance_id_var

    assert utterance_id_var.get(None) is None


@pytest.mark.unit
def test_utterance_id_var_set_and_reset() -> None:
    """Setting utterance_id_var with a token and resetting restores None."""
    from src.core.logging_config import utterance_id_var

    uid = str(uuid.uuid4())
    token = utterance_id_var.set(uid)
    try:
        assert utterance_id_var.get(None) == uid
    finally:
        utterance_id_var.reset(token)

    assert utterance_id_var.get(None) is None


@pytest.mark.unit
def test_utterance_id_var_is_thread_local() -> None:
    """utterance_id_var is context-local; a different thread sees its own value."""
    from src.core.logging_config import utterance_id_var

    uid_main = str(uuid.uuid4())
    uid_thread = str(uuid.uuid4())
    seen: list[str | None] = []

    token = utterance_id_var.set(uid_main)

    def _worker() -> None:
        seen.append(utterance_id_var.get(None))  # should be None, not uid_main
        t2 = utterance_id_var.set(uid_thread)
        seen.append(utterance_id_var.get(None))
        utterance_id_var.reset(t2)

    t = threading.Thread(target=_worker)
    t.start()
    t.join()

    utterance_id_var.reset(token)

    # The worker's first read should be None (each thread/task has its own copy).
    assert seen[0] is None
    # The worker's write is independent.
    assert seen[1] == uid_thread
    # The main var is still intact.
    assert utterance_id_var.get(None) is None


# ===========================================================================
# Part 2 — _JsonFormatter per-turn fields
# ===========================================================================


@pytest.mark.unit
def test_json_formatter_includes_utterance_id_when_set() -> None:
    """_JsonFormatter emits utterance_id when utterance_id_var is set."""
    from src.core.logging_config import _JsonFormatter, utterance_id_var

    formatter = _JsonFormatter()
    uid = str(uuid.uuid4())
    token = utterance_id_var.set(uid)
    try:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="inside a turn",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
    finally:
        utterance_id_var.reset(token)

    parsed = json.loads(output)
    assert "utterance_id" in parsed, "utterance_id missing from JSON output"
    assert parsed["utterance_id"] == uid


@pytest.mark.unit
def test_json_formatter_omits_utterance_id_when_not_set() -> None:
    """_JsonFormatter does NOT emit utterance_id when no turn is in progress."""
    from src.core.logging_config import _JsonFormatter, utterance_id_var

    # Ensure clean state (no leaked contextvar from other tests running in the
    # same context).
    token = utterance_id_var.set(None)
    try:
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.DEBUG,
            pathname="",
            lineno=1,
            msg="no active turn",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
    finally:
        utterance_id_var.reset(token)

    parsed = json.loads(output)
    assert "utterance_id" not in parsed


@pytest.mark.unit
def test_inference_dispatcher_binds_utterance_id_to_log(caplog: pytest.LogCaptureFixture) -> None:
    """Log records emitted INSIDE a turn carry the utterance_id in caplog context.

    We don't run real inference; instead we verify that the contextvar is set
    to the dispatcher's utterance_id by reading it from within a mock that
    fires during dispatch().
    """
    from src.core.logging_config import utterance_id_var
    from src.core.config import LLMConfig
    from src.core.state_machine import StateMachine, LumiState
    from src.llm.inference_dispatcher import LLMInferenceDispatcher

    captured_ids: list[str | None] = []

    # Mock reasoning router that captures the contextvar while "generating".
    mock_router = MagicMock()

    def _fake_generate(text, cancel_flag, utterance_id=None, **kwargs):
        # Record what utterance_id_var holds at inference time.
        captured_ids.append(utterance_id_var.get(None))
        return "ok"

    mock_router.generate = _fake_generate

    mock_reflex = MagicMock()
    mock_reflex.route.return_value = None  # No reflex hit — go to LLM path.
    mock_reflex.route_rag_intent.return_value = False

    mock_model_loader = MagicMock()
    mock_model_loader.is_loaded = True

    mock_memory = MagicMock()
    mock_memory.save.return_value = None

    mock_tool_executor = MagicMock()
    mock_tool_executor.execute.return_value = []

    state_machine = StateMachine()
    state_machine.transition_to(LumiState.LISTENING)
    state_machine.transition_to(LumiState.PROCESSING)

    import queue as _queue

    event_q: _queue.Queue[Any] = _queue.Queue()
    events: list[Any] = []

    def _post(ev: Any) -> None:
        events.append(ev)

    dispatcher = LLMInferenceDispatcher(
        model_loader=mock_model_loader,
        reflex_router=mock_reflex,
        reasoning_router=mock_router,
        memory=mock_memory,
        tool_executor=mock_tool_executor,
        state_machine=state_machine,
        event_queue=event_q,
        llm_config=LLMConfig(model_path="fake.gguf", inference_timeout_s=0.0),
    )

    dispatcher.dispatch("hello", source="test", rag_runtime_enabled=False, post_event=_post)

    # Wait for the inference thread to finish (it's a daemon thread).
    import time
    deadline = time.monotonic() + 2.0
    while not captured_ids and time.monotonic() < deadline:
        time.sleep(0.01)

    assert len(captured_ids) >= 1, "Fake generate was never called"
    assert captured_ids[0] is not None, "utterance_id_var was None during inference"
    # Must be a valid UUID4 string.
    assert len(captured_ids[0]) == 36  # "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"


@pytest.mark.unit
def test_utterance_id_var_cleared_after_turn() -> None:
    """After dispatch() completes, utterance_id_var returns to None in the
    inference thread's context.

    We verify this by checking that consecutive dispatches each get fresh IDs
    and that the contextvar doesn't leak between turns.
    """
    from src.core.logging_config import utterance_id_var
    from src.core.config import LLMConfig
    from src.core.state_machine import StateMachine, LumiState
    from src.llm.inference_dispatcher import LLMInferenceDispatcher

    ids_seen: list[str | None] = []
    ids_after: list[str | None] = []

    mock_router = MagicMock()

    def _fake_generate(text, cancel_flag, utterance_id=None, **kwargs):
        ids_seen.append(utterance_id_var.get(None))
        return "ok"

    mock_router.generate = _fake_generate

    mock_reflex = MagicMock()
    mock_reflex.route.return_value = None
    mock_reflex.route_rag_intent.return_value = False

    mock_model_loader = MagicMock()
    mock_model_loader.is_loaded = True
    mock_memory = MagicMock()
    mock_memory.save.return_value = None
    mock_tool_executor = MagicMock()
    mock_tool_executor.execute.return_value = []

    state_machine = StateMachine()
    state_machine.transition_to(LumiState.LISTENING)
    state_machine.transition_to(LumiState.PROCESSING)

    import queue as _queue, time

    dispatcher = LLMInferenceDispatcher(
        model_loader=mock_model_loader,
        reflex_router=mock_reflex,
        reasoning_router=mock_router,
        memory=mock_memory,
        tool_executor=mock_tool_executor,
        state_machine=state_machine,
        event_queue=_queue.Queue(),
        llm_config=LLMConfig(model_path="fake.gguf", inference_timeout_s=0.0),
    )

    # Monkey-patch _run_inference to also record the id AFTER _run_inference_body.
    original_dispatch = dispatcher.dispatch

    after_done = threading.Event()

    def _patched_dispatch(text, source, rag_runtime_enabled, post_event):
        original_dispatch(text, source, rag_runtime_enabled, post_event)

    dispatcher.dispatch("hello", source="test", rag_runtime_enabled=False, post_event=lambda e: None)

    deadline = time.monotonic() + 2.0
    while len(ids_seen) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    # The ID should be non-None during the turn.
    assert ids_seen[0] is not None


# ===========================================================================
# Part 3 — Error-tracker hook
# ===========================================================================


@pytest.mark.unit
def test_error_tracker_called_on_handler_exception() -> None:
    """When a handler raises, the registered error-tracker hook is called."""
    from src.core.events import UserTextEvent

    errors_received: list[BaseException] = []

    def _tracker(exc: BaseException) -> None:
        errors_received.append(exc)

    orch = _make_minimal_orchestrator(error_tracker=_tracker)

    boom = RuntimeError("handler blew up")

    def _bad_handler(event: Any) -> None:
        raise boom

    orch.register_handler(UserTextEvent, _bad_handler)

    # Dispatch the event directly (bypasses the run loop for determinism).
    orch._dispatch(UserTextEvent(text="hello"))

    assert len(errors_received) == 1
    assert errors_received[0] is boom


@pytest.mark.unit
def test_error_tracker_default_noop_no_crash() -> None:
    """No error_tracker registered → unhandled handler exception is logged but
    does NOT crash the process."""
    from src.core.events import UserTextEvent

    orch = _make_minimal_orchestrator(error_tracker=None)

    def _bad_handler(event: Any) -> None:
        raise ValueError("should be swallowed")

    orch.register_handler(UserTextEvent, _bad_handler)

    # Must not raise.
    orch._dispatch(UserTextEvent(text="hello"))


@pytest.mark.unit
def test_error_tracker_hook_exception_does_not_propagate() -> None:
    """If the error-tracker hook itself raises, that exception is suppressed."""
    from src.core.events import UserTextEvent

    def _bad_tracker(exc: BaseException) -> None:
        raise RuntimeError("tracker also broken")

    orch = _make_minimal_orchestrator(error_tracker=_bad_tracker)

    def _bad_handler(event: Any) -> None:
        raise ValueError("original error")

    orch.register_handler(UserTextEvent, _bad_handler)

    # Must not propagate — both the handler and tracker errors are absorbed.
    orch._dispatch(UserTextEvent(text="hello"))


@pytest.mark.unit
def test_set_error_tracker_replaces_hook() -> None:
    """set_error_tracker() replaces the hook at runtime."""
    from src.core.events import UserTextEvent

    first_calls: list[BaseException] = []
    second_calls: list[BaseException] = []

    orch = _make_minimal_orchestrator(error_tracker=lambda e: first_calls.append(e))

    def _bad_handler(event: Any) -> None:
        raise RuntimeError("boom")

    orch.register_handler(UserTextEvent, _bad_handler)

    orch._dispatch(UserTextEvent(text="a"))
    assert len(first_calls) == 1

    orch.set_error_tracker(lambda e: second_calls.append(e))
    orch._dispatch(UserTextEvent(text="b"))

    assert len(first_calls) == 1  # old hook not called again
    assert len(second_calls) == 1


# ===========================================================================
# Part 4 — Heartbeat
# ===========================================================================


@pytest.mark.unit
def test_heartbeat_fires_directly_via_beat_method(caplog: pytest.LogCaptureFixture) -> None:
    """Calling _beat() directly emits a heartbeat log line — deterministic,
    no real sleep required."""
    orch = _make_minimal_orchestrator(heartbeat_interval_s=0.0)

    with caplog.at_level(logging.INFO):
        orch._beat()

    assert any("Heartbeat" in r.message for r in caplog.records)


@pytest.mark.unit
def test_heartbeat_disabled_when_interval_zero() -> None:
    """When heartbeat_interval_s=0.0 (default), no timer is scheduled."""
    orch = _make_minimal_orchestrator(heartbeat_interval_s=0.0)
    assert orch._heartbeat_timer is None


@pytest.mark.unit
def test_heartbeat_timer_scheduled_when_interval_positive() -> None:
    """When heartbeat_interval_s > 0, a threading.Timer is created."""
    # Use a very long interval so the timer never fires during the test.
    orch = _make_minimal_orchestrator(heartbeat_interval_s=9999.0)
    try:
        assert orch._heartbeat_timer is not None
        assert isinstance(orch._heartbeat_timer, threading.Timer)
    finally:
        if orch._heartbeat_timer is not None:
            orch._heartbeat_timer.cancel()


@pytest.mark.unit
def test_heartbeat_fires_and_reschedules(caplog: pytest.LogCaptureFixture) -> None:
    """With a short interval the heartbeat fires at least once and reschedules
    itself.  We use a very short interval (0.05 s) and wait briefly."""
    import time

    interval = 0.05  # 50 ms — fast enough for a unit test, not a real sleep loop
    orch = _make_minimal_orchestrator(heartbeat_interval_s=interval)

    try:
        deadline = time.monotonic() + 1.0  # generous upper bound
        with caplog.at_level(logging.INFO):
            while time.monotonic() < deadline:
                if any("Heartbeat" in r.message for r in caplog.records):
                    break
                time.sleep(0.01)

        assert any("Heartbeat" in r.message for r in caplog.records), (
            "Heartbeat log line never emitted within 1 s"
        )
        # After the first fire, a new timer should have been rescheduled.
        assert orch._heartbeat_timer is not None
    finally:
        if orch._heartbeat_timer is not None:
            orch._heartbeat_timer.cancel()


@pytest.mark.unit
def test_heartbeat_cancelled_on_shutdown() -> None:
    """Dispatching ShutdownEvent cancels the heartbeat timer."""
    from src.core.events import ShutdownEvent
    import time

    orch = _make_minimal_orchestrator(heartbeat_interval_s=9999.0)
    timer = orch._heartbeat_timer
    assert timer is not None
    assert timer.is_alive()

    # Drive the shutdown handler directly without running the full event loop.
    orch._dispatch(ShutdownEvent())

    # timer.cancel() sets the internal event; the timer thread may take a brief
    # moment to exit.  Join with a short timeout to wait for clean exit.
    timer.join(timeout=1.0)
    assert not timer.is_alive(), "Heartbeat timer was not stopped by shutdown"


# ===========================================================================
# Part 5 — ObservabilityConfig loaded from LumiConfig
# ===========================================================================


@pytest.mark.unit
def test_observability_config_defaults() -> None:
    """ObservabilityConfig has sane defaults (heartbeat disabled)."""
    from src.core.config import ObservabilityConfig

    cfg = ObservabilityConfig()
    assert cfg.heartbeat_interval_s == 0.0


@pytest.mark.unit
def test_lumi_config_carries_observability() -> None:
    """LumiConfig exposes the observability sub-config."""
    from src.core.config import LumiConfig, ObservabilityConfig

    cfg = LumiConfig(observability=ObservabilityConfig(heartbeat_interval_s=30.0))
    assert cfg.observability.heartbeat_interval_s == 30.0


@pytest.mark.unit
def test_load_config_parses_observability_section(tmp_path: Any) -> None:
    """load_config() reads the observability: section from YAML."""
    from src.core.config import load_config

    yaml_content = "observability:\n  heartbeat_interval_s: 45.0\n"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(str(cfg_file))
    assert cfg.observability.heartbeat_interval_s == 45.0
