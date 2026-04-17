"""
Unit tests for src.tools.registry.ToolRegistry.

All tests are pure in-memory; no subprocess or filesystem calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str = "A test tool") -> MagicMock:
    """Return a MagicMock that satisfies the Tool Protocol."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.execute.return_value = ToolResult(success=True, output="ok", data={})
    return tool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_and_get_returns_tool() -> None:
    registry = ToolRegistry()
    tool = _make_tool("my_tool")
    registry.register(tool)
    assert registry.get("my_tool") is tool


@pytest.mark.unit
def test_get_unknown_name_returns_none() -> None:
    registry = ToolRegistry()
    assert registry.get("no_such_tool") is None


@pytest.mark.unit
def test_is_registered_returns_true_for_registered_tool() -> None:
    registry = ToolRegistry()
    tool = _make_tool("existing_tool")
    registry.register(tool)
    assert registry.is_registered("existing_tool") is True


@pytest.mark.unit
def test_is_registered_returns_false_for_unknown_tool() -> None:
    registry = ToolRegistry()
    assert registry.is_registered("ghost_tool") is False


@pytest.mark.unit
def test_list_tools_returns_name_and_description() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("alpha", "Alpha tool"))
    registry.register(_make_tool("beta", "Beta tool"))

    result = registry.list_tools()

    # Should be sorted by name
    assert result == [
        {"name": "alpha", "description": "Alpha tool"},
        {"name": "beta", "description": "Beta tool"},
    ]


@pytest.mark.unit
def test_registering_same_name_twice_overwrites() -> None:
    registry = ToolRegistry()
    first = _make_tool("dup_tool", "first version")
    second = _make_tool("dup_tool", "second version")

    registry.register(first)
    registry.register(second)

    assert registry.get("dup_tool") is second
    assert len(registry.list_tools()) == 1
    assert registry.list_tools()[0]["description"] == "second version"
