"""Datetime tool for Project Lumi — returns current local time, no dependencies."""

from __future__ import annotations

import logging
from datetime import datetime

from src.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class DateTimeTool:
    """Return the current local date and time.

    Schema::

        {"tool": "datetime", "args": {}}

    Returns the current datetime in both ISO-8601 and human-readable format.
    """

    name: str = "datetime"
    description: str = (
        "Get the current local date and time. "
        "No args required. "
        "Returns date, time, weekday, and ISO timestamp."
    )

    def execute(self, **kwargs: object) -> ToolResult:  # noqa: D102
        try:
            now = datetime.now()
            iso = now.isoformat(timespec="seconds")
            human = now.strftime("%A, %-d %B %Y at %H:%M")
            output = f"Current date and time: {human} (ISO: {iso})"
            return ToolResult(
                success=True,
                output=output,
                data={
                    "iso": iso,
                    "human": human,
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    "weekday": now.strftime("%A"),
                },
            )
        except Exception as exc:
            logger.exception("DateTimeTool.execute failed")
            return ToolResult(success=False, output=f"Failed to get datetime: {exc}", data={})
