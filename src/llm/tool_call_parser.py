"""Parser for tool-call blocks embedded in LLM output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract and validate tool-call blocks from model output.

    Scans *text* for ``<tool_call>…</tool_call>`` delimiters, parses the
    enclosed content as JSON, and returns only those blocks that contain both
    a ``tool`` key and an ``args`` key.  Malformed JSON or missing keys are
    silently dropped — this function never raises.

    Args:
        text: Raw text output from the LLM, possibly containing tool-call
              blocks alongside natural language.

    Returns:
        A (possibly empty) list of validated tool-call dicts, each guaranteed
        to have at least ``"tool"`` and ``"args"`` keys.
    """
    results: list[dict[str, Any]] = []

    for match in _TOOL_CALL_PATTERN.finditer(text):
        body = match.group(1).strip()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed tool-call JSON: %r", body[:80])
            continue

        if not isinstance(data, dict):
            continue
        if "tool" not in data or "args" not in data:
            continue

        results.append(data)

    return results
