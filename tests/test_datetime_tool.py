"""Tests for DateTimeTool."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from src.tools.datetime_tool import DateTimeTool


@pytest.fixture()
def tool() -> DateTimeTool:
    return DateTimeTool()


class TestDateTimeToolSuccess:
    def test_returns_success(self, tool: DateTimeTool) -> None:
        result = tool.execute()
        assert result.success is True

    def test_output_contains_date(self, tool: DateTimeTool) -> None:
        result = tool.execute()
        now = datetime.now()
        assert str(now.year) in result.output

    def test_data_fields_present(self, tool: DateTimeTool) -> None:
        result = tool.execute()
        for field in ("iso", "human", "year", "month", "day", "hour", "minute", "weekday"):
            assert field in result.data, f"Missing data field: {field}"

    def test_iso_format_valid(self, tool: DateTimeTool) -> None:
        result = tool.execute()
        datetime.fromisoformat(result.data["iso"])  # raises if invalid

    def test_weekday_is_string(self, tool: DateTimeTool) -> None:
        result = tool.execute()
        assert isinstance(result.data["weekday"], str)
        assert result.data["weekday"] in (
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        )


class TestDateTimeToolEdgeCases:
    def test_ignores_extra_kwargs(self, tool: DateTimeTool) -> None:
        result = tool.execute(foo="bar", baz=42)
        assert result.success is True

    def test_execute_never_raises(self, tool: DateTimeTool) -> None:
        with patch("src.tools.datetime_tool.datetime") as mock_dt:
            mock_dt.now.side_effect = OSError("clock unavailable")
            result = tool.execute()
        assert result.success is False
        assert result.output != ""


class TestDateTimeToolProtocol:
    def test_has_name(self, tool: DateTimeTool) -> None:
        assert tool.name == "datetime"

    def test_has_description(self, tool: DateTimeTool) -> None:
        assert isinstance(tool.description, str) and len(tool.description) > 0
