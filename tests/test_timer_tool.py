"""Tests for TimerTool."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.core.events import TimerExpiredEvent
from src.tools.timer_tool import TimerTool, _seconds_to_human


@pytest.fixture()
def events() -> list:
    return []


@pytest.fixture()
def tool(events: list) -> TimerTool:
    return TimerTool(post_event=events.append)


class TestTimerToolSuccess:
    def test_returns_success_immediately(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=60, label="pasta")
        assert result.success is True
        assert "pasta" in result.output
        assert "60" in result.output or "minute" in result.output

    def test_fires_event_after_delay(self, events: list) -> None:
        tool = TimerTool(post_event=events.append)
        tool.execute(seconds=1, label="quick")

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not events:
            time.sleep(0.05)

        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, TimerExpiredEvent)
        assert evt.label == "quick"
        assert evt.seconds == 1

    def test_data_fields_present(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=30, label="tea")
        assert result.data["label"] == "tea"
        assert result.data["seconds"] == 30

    def test_default_label_used_when_omitted(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=10)
        assert result.success is True
        assert "Timer" in result.output

    def test_blank_label_falls_back_to_timer(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=10, label="   ")
        assert "Timer" in result.output

    def test_multiple_concurrent_timers(self, events: list) -> None:
        tool = TimerTool(post_event=events.append)
        tool.execute(seconds=1, label="A")
        tool.execute(seconds=1, label="B")

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and len(events) < 2:
            time.sleep(0.05)

        labels = {e.label for e in events}
        assert labels == {"A", "B"}


class TestTimerToolEdgeCases:
    def test_negative_seconds_fails(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=-1, label="bad")
        assert result.success is False

    def test_zero_seconds_fails(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=0, label="bad")
        assert result.success is False

    def test_missing_seconds_fails(self, tool: TimerTool) -> None:
        result = tool.execute(label="bad")
        assert result.success is False

    def test_float_seconds_fails(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=1.5, label="bad")
        assert result.success is False

    def test_exceeds_max_seconds_fails(self, tool: TimerTool) -> None:
        result = tool.execute(seconds=86_401, label="too long")
        assert result.success is False

    def test_execute_never_raises(self, tool: TimerTool) -> None:
        # Any kwargs, no exception
        result = tool.execute(seconds="notanint", label=None)
        assert result.success is False


class TestTimerToolProtocol:
    def test_name(self, tool: TimerTool) -> None:
        assert tool.name == "set_timer"

    def test_description(self, tool: TimerTool) -> None:
        assert isinstance(tool.description, str) and len(tool.description) > 0


class TestSecondsToHuman:
    def test_seconds_singular(self) -> None:
        assert _seconds_to_human(1) == "1 second"

    def test_seconds_plural(self) -> None:
        assert _seconds_to_human(45) == "45 seconds"

    def test_minutes_singular(self) -> None:
        assert _seconds_to_human(60) == "1 minute"

    def test_minutes_plural(self) -> None:
        assert _seconds_to_human(120) == "2 minutes"

    def test_mixed(self) -> None:
        assert _seconds_to_human(90) == "1m 30s"

    def test_large(self) -> None:
        assert _seconds_to_human(3600) == "60 minutes"
