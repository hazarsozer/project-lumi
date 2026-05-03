"""Countdown timer tool for Project Lumi."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from src.core.events import TimerExpiredEvent
from src.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_MAX_SECONDS = 86_400  # 24 hours — sanity cap


class TimerTool:
    """Set a countdown timer that fires a TimerExpiredEvent when it expires.

    Schema::

        {"tool": "set_timer", "args": {"seconds": <int>, "label": "<str>"}}

    The timer runs in a daemon thread so it survives the current conversation
    turn but does not block the event loop. Multiple concurrent timers are
    supported.

    Behaviour when the timer fires:
    - If Lumi is IDLE, she speaks a verbal alarm.
    - If Lumi is busy (PROCESSING, SPEAKING), the alarm is logged and skipped.
    """

    name: str = "set_timer"
    description: str = (
        "Set a countdown timer. "
        "Args: seconds (int, 1-86400), label (str, e.g. 'pasta'). "
        "Fires a verbal alarm when the timer expires."
    )

    def __init__(self, post_event: Callable[[Any], None]) -> None:
        self._post_event = post_event

    def execute(self, **kwargs: object) -> ToolResult:  # noqa: D102
        seconds = kwargs.get("seconds")
        label = kwargs.get("label", "Timer")

        if not isinstance(seconds, int) or seconds <= 0:
            return ToolResult(
                success=False,
                output="set_timer requires 'seconds' as a positive integer.",
                data={},
            )
        if seconds > _MAX_SECONDS:
            return ToolResult(
                success=False,
                output=f"set_timer maximum is {_MAX_SECONDS} seconds (24 hours).",
                data={},
            )
        if not isinstance(label, str):
            label = str(label)
        label = label.strip() or "Timer"

        def _countdown() -> None:
            import time
            time.sleep(seconds)
            logger.info("TimerTool: '%s' expired after %ds", label, seconds)
            try:
                self._post_event(TimerExpiredEvent(label=label, seconds=seconds))
            except Exception:
                logger.exception("TimerTool: failed to post TimerExpiredEvent")

        thread = threading.Thread(
            target=_countdown,
            daemon=True,
            name=f"TimerThread-{label}",
        )
        thread.start()
        logger.debug("TimerTool: '%s' started for %ds", label, seconds)

        human = _seconds_to_human(seconds)
        return ToolResult(
            success=True,
            output=f"Timer set: '{label}' will fire in {human}.",
            data={"label": label, "seconds": seconds},
        )


def _seconds_to_human(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, secs = divmod(seconds, 60)
    if secs == 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes}m {secs}s"
