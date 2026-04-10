"""
Tests for src.llm.reflex_router.ReflexRouter.

Mocking strategy
----------------
ReflexRouter must be entirely rule-based (regex / keyword matching) — it must
not call any ML model.  These tests therefore require no mocks at all; they
validate the routing logic in pure Python.

All tests are marked ``unit``.
"""

from __future__ import annotations

import pytest

# RED: these imports will fail until src/llm/reflex_router.py is written.
from src.llm.reflex_router import ReflexRouter  # type: ignore[import]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_greeting_returns_response() -> None:
    """'hello' must return a non-None, non-empty string."""
    router = ReflexRouter()
    result = router.route("hello")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_hi_returns_response() -> None:
    """'hi' must also be recognized as a greeting."""
    router = ReflexRouter()
    result = router.route("hi")
    assert result is not None


@pytest.mark.unit
def test_unknown_query_returns_none() -> None:
    """A complex technical query must return None (not handled by reflex layer)."""
    router = ReflexRouter()
    result = router.route("what is quantum entanglement and how does it work")
    assert result is None


@pytest.mark.unit
def test_time_query_returns_response() -> None:
    """'what time is it' must be recognized and return a non-None string."""
    router = ReflexRouter()
    result = router.route("what time is it")
    assert result is not None
    assert isinstance(result, str)


@pytest.mark.unit
def test_empty_string_returns_none() -> None:
    """An empty string input must return None without raising."""
    router = ReflexRouter()
    result = router.route("")
    assert result is None


@pytest.mark.unit
def test_route_is_case_insensitive() -> None:
    """'HELLO' must produce the same non-None result as 'hello'."""
    router = ReflexRouter()
    lower_result = router.route("hello")
    upper_result = router.route("HELLO")
    assert upper_result is not None
    assert lower_result is not None


@pytest.mark.unit
def test_whitespace_only_returns_none() -> None:
    """A string of only whitespace must return None without raising."""
    router = ReflexRouter()
    result = router.route("   ")
    assert result is None


@pytest.mark.unit
def test_route_returns_string_or_none() -> None:
    """route() must always return either str or None — never raise unexpectedly."""
    router = ReflexRouter()
    inputs = ["hello", "goodbye", "set a timer", "random gibberish xyz123", ""]
    for text in inputs:
        result = router.route(text)
        assert result is None or isinstance(result, str)
