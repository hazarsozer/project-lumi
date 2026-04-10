"""
Tests for src.llm.tool_call_parser.parse_tool_calls.

Mocking strategy
----------------
parse_tool_calls is a pure text-parsing function — no hardware, no models, no
external dependencies.  All tests are self-contained and deterministic.

All tests are marked ``unit``.
"""

from __future__ import annotations

import pytest

# RED: these imports will fail until src/llm/tool_call_parser.py is written.
from src.llm.tool_call_parser import parse_tool_calls  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_tool_call(json_body: str) -> str:
    """Wrap a JSON body in the tool-call delimiters the parser expects."""
    return f"<tool_call>{json_body}</tool_call>"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parses_single_tool_call() -> None:
    """A single valid tool-call block must be extracted as one dict."""
    text = _wrap_tool_call('{"tool": "get_weather", "args": {"city": "London"}}')
    result = parse_tool_calls(text)
    assert len(result) == 1
    assert result[0]["tool"] == "get_weather"
    assert result[0]["args"] == {"city": "London"}


@pytest.mark.unit
def test_parses_multiple_tool_calls() -> None:
    """Multiple sequential tool-call blocks must each be extracted."""
    text = (
        _wrap_tool_call('{"tool": "tool_a", "args": {"x": 1}}')
        + " some text in between "
        + _wrap_tool_call('{"tool": "tool_b", "args": {"y": 2}}')
    )
    result = parse_tool_calls(text)
    assert len(result) == 2
    assert result[0]["tool"] == "tool_a"
    assert result[1]["tool"] == "tool_b"


@pytest.mark.unit
def test_returns_empty_on_no_tool_calls() -> None:
    """Plain text with no tool-call blocks must return an empty list."""
    result = parse_tool_calls("The weather today is sunny and warm.")
    assert result == []


@pytest.mark.unit
def test_returns_empty_on_malformed_json() -> None:
    """A tool-call block containing broken JSON must return [] without raising."""
    text = _wrap_tool_call('{"tool": "broken", "args": {missing_quote: true}')
    result = parse_tool_calls(text)
    assert result == []


@pytest.mark.unit
def test_validates_required_keys_missing_tool() -> None:
    """A block missing the 'tool' key must be excluded from results."""
    text = _wrap_tool_call('{"args": {"x": 1}}')
    result = parse_tool_calls(text)
    assert result == []


@pytest.mark.unit
def test_validates_required_keys_missing_args() -> None:
    """A block missing the 'args' key must be excluded from results."""
    text = _wrap_tool_call('{"tool": "do_something"}')
    result = parse_tool_calls(text)
    assert result == []


@pytest.mark.unit
def test_returns_empty_on_empty_string() -> None:
    """An empty input string must return an empty list without raising."""
    result = parse_tool_calls("")
    assert result == []


@pytest.mark.unit
def test_partial_valid_calls_filtered() -> None:
    """A mix of valid and invalid blocks must return only the valid ones."""
    valid = _wrap_tool_call('{"tool": "valid_tool", "args": {}}')
    invalid = _wrap_tool_call('{"tool": "bad", "broken": }')
    missing_args = _wrap_tool_call('{"tool": "no_args_key"}')
    text = valid + invalid + missing_args
    result = parse_tool_calls(text)
    assert len(result) == 1
    assert result[0]["tool"] == "valid_tool"


@pytest.mark.unit
def test_args_can_be_empty_dict() -> None:
    """A tool call with an empty args dict must be valid and included."""
    text = _wrap_tool_call('{"tool": "ping", "args": {}}')
    result = parse_tool_calls(text)
    assert len(result) == 1
    assert result[0]["tool"] == "ping"
    assert result[0]["args"] == {}


@pytest.mark.unit
def test_nested_args_preserved() -> None:
    """Nested argument structures must be preserved exactly."""
    text = _wrap_tool_call(
        '{"tool": "create_file", "args": {"path": "/tmp/x.txt", "content": {"lines": [1, 2, 3]}}}'
    )
    result = parse_tool_calls(text)
    assert len(result) == 1
    assert result[0]["args"]["content"]["lines"] == [1, 2, 3]


@pytest.mark.unit
def test_non_dict_json_value_is_excluded() -> None:
    """A tool-call block whose JSON body is a valid non-dict value (e.g. a list or
    a string) must be excluded from results (line 42 — isinstance check)."""
    # JSON list inside the delimiter — valid JSON, but not a dict.
    text_list = _wrap_tool_call('[{"tool": "tool_a", "args": {}}]')
    assert parse_tool_calls(text_list) == []

    # JSON string inside the delimiter.
    text_string = _wrap_tool_call('"just a string"')
    assert parse_tool_calls(text_string) == []

    # JSON number inside the delimiter.
    text_number = _wrap_tool_call("42")
    assert parse_tool_calls(text_number) == []
