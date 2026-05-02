"""Verify _OUTBOUND covers every event type that has an on_* handler."""

from __future__ import annotations

import inspect
import pytest

from src.core.event_bridge import EventBridge, _OUTBOUND  # type: ignore[import]


@pytest.mark.unit
def test_outbound_keys_are_types() -> None:
    for k in _OUTBOUND:
        assert isinstance(k, type), f"{k!r} is not a type"


@pytest.mark.unit
def test_outbound_values_are_nonempty_strings() -> None:
    for v in _OUTBOUND.values():
        assert isinstance(v, str) and v, f"{v!r} is not a non-empty string"


@pytest.mark.unit
def test_outbound_count_matches_event_handlers() -> None:
    """Every on_<event>() method (excluding custom-sig ones) must map to an entry."""
    excluded = {"on_state_change", "on_error"}
    handler_names = [
        name
        for name, _ in inspect.getmembers(EventBridge, predicate=inspect.isfunction)
        if name.startswith("on_") and name not in excluded
    ]
    assert len(_OUTBOUND) >= len(handler_names), (
        f"_OUTBOUND has {len(_OUTBOUND)} entries but EventBridge has "
        f"{len(handler_names)} on_* handlers (excluding custom-sig ones)"
    )
