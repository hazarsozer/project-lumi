"""
Centralized logging configuration for Project Lumi.

All modules should obtain loggers via:
    import logging
    logger = logging.getLogger(__name__)

Call setup_logging() once at application startup (e.g., in main.py).
Repeated calls are silently ignored via the module-level guard.
"""

from __future__ import annotations

import json
import logging
import traceback


# Module-level guard: prevents duplicate handler registration on repeated calls.
_LOGGING_CONFIGURED: bool = False

_DEV_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


class _JsonFormatter(logging.Formatter):
    """Formats each log record as a single JSON object on one line.

    Output fields:
        timestamp  — ISO-8601 datetime string
        level      — uppercase level name (INFO, WARNING, …)
        logger     — logger name (typically the module __name__)
        message    — the formatted log message
        exc_info   — formatted traceback string, present only when an
                     exception is attached to the record
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt or _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).rstrip()

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger for the entire application.

    Args:
        level: Logging level name — one of DEBUG, INFO, WARNING, ERROR,
               CRITICAL.  Case-insensitive.  Defaults to "INFO".
        json_format: When True, emit structured JSON (one object per line)
                     suitable for log aggregators.  When False (default),
                     use a human-readable development format.

    Design notes:
        - Idempotent: the function is a no-op if called more than once.
          This prevents handler duplication when modules call it on import.
        - Only the root logger is configured; child loggers created via
          logging.getLogger(__name__) inherit the root handler by default.
        - print() is never used — all output goes through logging.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_format:
        formatter: logging.Formatter = _JsonFormatter(datefmt=_DATE_FORMAT)
    else:
        formatter = logging.Formatter(fmt=_DEV_FORMAT, datefmt=_DATE_FORMAT)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    _LOGGING_CONFIGURED = True
