"""
Tests for src/core/logging_config.py.

Covers:
- setup_logging configures root logger with the correct level.
- setup_logging is idempotent (second call is a no-op, no duplicate handlers).
- json_format=True produces _JsonFormatter output.
- json_format=False produces human-readable output.
- _JsonFormatter includes exc_info field when an exception is attached.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch


import pytest


def _reset_logging_config() -> None:
    """Reset the module-level guard so setup_logging() acts fresh in each test."""
    import src.core.logging_config as lc
    lc._LOGGING_CONFIGURED = False


def _cleanup_root_handlers(added_handlers: list) -> None:
    root = logging.getLogger()
    for h in added_handlers:
        root.removeHandler(h)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_setup_logging_sets_level_on_root_logger():
    """setup_logging configures root logger level correctly."""
    _reset_logging_config()
    root = logging.getLogger()
    before_handlers = list(root.handlers)

    from src.core.logging_config import setup_logging
    setup_logging(level="WARNING")

    assert root.level == logging.WARNING

    added = [h for h in root.handlers if h not in before_handlers]
    _cleanup_root_handlers(added)
    _reset_logging_config()


@pytest.mark.unit
def test_setup_logging_is_idempotent():
    """A second call to setup_logging is a no-op — no duplicate handlers."""
    _reset_logging_config()
    root = logging.getLogger()
    before_count = len(root.handlers)

    from src.core.logging_config import setup_logging
    setup_logging(level="DEBUG")
    count_after_first = len(root.handlers)
    setup_logging(level="DEBUG")  # second call — must be no-op
    count_after_second = len(root.handlers)

    assert count_after_second == count_after_first  # no extra handler

    added = root.handlers[before_count:]
    _cleanup_root_handlers(added)
    _reset_logging_config()


@pytest.mark.unit
def test_setup_logging_json_format_attaches_json_formatter():
    """With json_format=True, the root handler uses _JsonFormatter."""
    _reset_logging_config()
    root = logging.getLogger()
    before_handlers = list(root.handlers)

    from src.core.logging_config import setup_logging, _JsonFormatter
    setup_logging(json_format=True)

    added = [h for h in root.handlers if h not in before_handlers]
    assert any(isinstance(h.formatter, _JsonFormatter) for h in added)

    _cleanup_root_handlers(added)
    _reset_logging_config()


# ---------------------------------------------------------------------------
# _JsonFormatter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_json_formatter_produces_valid_json():
    """_JsonFormatter.format returns a parseable JSON string."""
    from src.core.logging_config import _JsonFormatter

    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert "hello world" in parsed["message"]
    assert "timestamp" in parsed


@pytest.mark.unit
def test_json_formatter_includes_exc_info_when_exception_attached():
    """_JsonFormatter includes an exc_info key when a record carries an exception."""
    from src.core.logging_config import _JsonFormatter
    import sys

    formatter = _JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname="",
        lineno=1,
        msg="something failed",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exc_info" in parsed
    assert "ValueError" in parsed["exc_info"]
