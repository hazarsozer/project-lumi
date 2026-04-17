"""Rule-based reflex router for fast, stateless response generation.

Handles simple pattern-matched queries (greetings, time) without
invoking the LLM.  All logic is pure regex — no ML model is used.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

_GREETING_PATTERN = re.compile(
    r"\b(hello|hi|hey|howdy|greetings)\b", re.IGNORECASE
)
_TIME_PATTERN = re.compile(
    r"\b(what\s+time|current\s+time|time\s+is\s+it)\b", re.IGNORECASE
)
_RAG_PATTERN = re.compile(
    r"\b(search|look\s*up|find|check)\b.{0,40}"
    r"\b(my\s+(docs?|notes?|files?|wiki|journal|documents?)"
    r"|in\s+my\s+(docs?|notes?|files?|wiki|journal|documents?))\b",
    re.IGNORECASE,
)


class ReflexRouter:
    """Handles simple, pattern-matched queries without invoking the LLM."""

    def route(self, text: str) -> str | None:
        """Match *text* against known patterns and return a canned response.

        Args:
            text: Raw user input.

        Returns:
            A response string if the text matches a known pattern, or ``None``
            if the query should be forwarded to the reasoning router.
        """
        stripped = text.strip()
        if not stripped:
            return None

        if _GREETING_PATTERN.search(stripped):
            return "Hello! How can I help you?"

        if _TIME_PATTERN.search(stripped):
            now = datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."

        return None

    def route_rag_intent(self, text: str) -> bool:
        """Return True if *text* expresses an intent to search personal docs.

        Boolean signal only — does not return a canned response.  The
        Orchestrator uses the result to set ``use_rag=True`` on the reasoning
        router so retrieval runs before LLM inference.
        """
        return bool(_RAG_PATTERN.search(text.strip()))
