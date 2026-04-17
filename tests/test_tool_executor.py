"""
Unit tests for src.tools.executor.ToolExecutor.

All tests are pure in-memory; no subprocess or filesystem calls.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.config import ToolsConfig
from src.tools.base import ToolResult
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(*tool_names: str) -> ToolRegistry:
    """Return a ToolRegistry pre-populated with mock tools."""
    registry = ToolRegistry()
    for name in tool_names:
        tool = MagicMock()
        tool.name = name
        tool.description = f"Mock {name}"
        tool.execute.return_value = ToolResult(
            success=True, output=f"ran {name}", data={}
        )
        registry.register(tool)
    return registry


def _config(allowed: tuple[str, ...] = ("tool_a", "tool_b"), timeout: float = 5.0) -> ToolsConfig:
    return ToolsConfig(allowed_tools=allowed, execution_timeout_s=timeout)


def _no_cancel() -> threading.Event:
    return threading.Event()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_calls_allowed_tool_and_returns_tool_result() -> None:
    registry = _make_registry("tool_a")
    executor = ToolExecutor(registry, _config())

    results = executor.execute(
        [{"tool": "tool_a", "args": {"x": 1}}], _no_cancel()
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].output == "ran tool_a"


@pytest.mark.unit
def test_execute_blocks_disallowed_tool() -> None:
    registry = _make_registry("tool_a")
    executor = ToolExecutor(registry, _config(allowed=("tool_a",)))

    results = executor.execute(
        [{"tool": "forbidden_tool", "args": {}}], _no_cancel()
    )

    assert len(results) == 1
    assert results[0].success is False
    assert "not allowed" in results[0].output.lower()


@pytest.mark.unit
def test_execute_stops_early_when_cancel_flag_set() -> None:
    registry = _make_registry("tool_a", "tool_b")
    executor = ToolExecutor(registry, _config())

    cancel = threading.Event()
    cancel.set()  # pre-set — no tool should run

    results = executor.execute(
        [
            {"tool": "tool_a", "args": {}},
            {"tool": "tool_b", "args": {}},
        ],
        cancel,
    )

    assert results == []
    registry.get("tool_a").execute.assert_not_called()
    registry.get("tool_b").execute.assert_not_called()


@pytest.mark.unit
def test_execute_returns_failure_on_tool_exception() -> None:
    registry = ToolRegistry()
    boom = MagicMock()
    boom.name = "boom_tool"
    boom.description = "Raises on execute"
    boom.execute.side_effect = RuntimeError("kaboom")
    registry.register(boom)

    executor = ToolExecutor(registry, _config(allowed=("boom_tool",)))
    results = executor.execute([{"tool": "boom_tool", "args": {}}], _no_cancel())

    assert len(results) == 1
    assert results[0].success is False
    assert "kaboom" in results[0].output


@pytest.mark.unit
def test_execute_empty_tool_calls_returns_empty_list() -> None:
    registry = _make_registry("tool_a")
    executor = ToolExecutor(registry, _config())

    results = executor.execute([], _no_cancel())

    assert results == []


@pytest.mark.unit
def test_execute_enforces_timeout() -> None:
    """A tool that sleeps longer than the timeout should produce a Timeout failure."""
    registry = ToolRegistry()

    slow = MagicMock()
    slow.name = "slow_tool"
    slow.description = "Sleeps forever"

    def _sleep(*_: Any, **__: Any) -> ToolResult:
        time.sleep(10)
        return ToolResult(success=True, output="never", data={})

    slow.execute.side_effect = _sleep
    registry.register(slow)

    executor = ToolExecutor(registry, _config(allowed=("slow_tool",), timeout=0.2))
    results = executor.execute([{"tool": "slow_tool", "args": {}}], _no_cancel())

    assert len(results) == 1
    assert results[0].success is False
    assert "timeout" in results[0].output.lower()


@pytest.mark.unit
def test_execute_processes_multiple_tool_calls_in_order() -> None:
    call_order: list[str] = []

    registry = ToolRegistry()
    for name in ("tool_a", "tool_b", "tool_c"):
        tool = MagicMock()
        tool.name = name
        tool.description = f"Mock {name}"
        captured = name  # capture loop variable

        def _execute(args: dict, _n: str = captured) -> ToolResult:
            call_order.append(_n)
            return ToolResult(success=True, output=f"ran {_n}", data={})

        tool.execute.side_effect = _execute
        registry.register(tool)

    executor = ToolExecutor(
        registry, _config(allowed=("tool_a", "tool_b", "tool_c"))
    )
    results = executor.execute(
        [
            {"tool": "tool_a", "args": {}},
            {"tool": "tool_b", "args": {}},
            {"tool": "tool_c", "args": {}},
        ],
        _no_cancel(),
    )

    assert [r.output for r in results] == ["ran tool_a", "ran tool_b", "ran tool_c"]
    assert call_order == ["tool_a", "tool_b", "tool_c"]


@pytest.mark.unit
def test_execute_with_empty_args_dict_works() -> None:
    registry = _make_registry("tool_a")
    executor = ToolExecutor(registry, _config())

    # No "args" key in the call dict — executor should default to {}
    results = executor.execute([{"tool": "tool_a"}], _no_cancel())

    assert len(results) == 1
    assert results[0].success is True
    registry.get("tool_a").execute.assert_called_once_with({})
