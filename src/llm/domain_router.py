"""Regex-based domain classifier for Lumi query routing."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Safety checked first so tool-invocation verbs (open, run) can't shadow
# harmful phrases like "run this malware" or "open the exploit".
_REFUSAL_PATTERN = re.compile(
    r"\b(hack|exploit|malware|virus|worm|ransomware|spyware"
    r"|how\s+to\s+make\s+(a\s+)?(bomb|weapon|drug|poison)"
    r"|bomb|illegal\s+(activity|drug)|kill\s+(someone|a\s+person))\b",
    re.IGNORECASE,
)

_TOOL_CALL_PATTERN = re.compile(
    r"\b(open|launch|start|run|execute|take\s+a?\s*screenshot"
    r"|click|type|create\s+(a\s+)?file|delete|move|copy|paste"
    r"|search\s+for|find\s+(?:a\s+|the\s+)?file)\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_PATTERN = re.compile(
    r"(write\s+(my|a|an|the)\s*(essay|report|thesis|homework|paper|cover\s+letter)"
    r"|do\s+my\s+homework"
    r"|write\s+my\b)",
    re.IGNORECASE,
)

_KNOWLEDGE_LIMIT_PATTERN = re.compile(
    r"\b(stock\s+price|weather(\s+forecast)?|latest\s+news|real.?time"
    r"|right\s+now|what('s|\s+is)\s+(currently|happening)"
    r"|happening\s+in\s+the\s+world|who\s+is\s+winning\s+right\s+now)\b",
    re.IGNORECASE,
)

_CONCISE_FACTUAL_PATTERN = re.compile(
    r"^\s*(what\s+(is|are|year|was)|who\s+(is|was)|when\s+did|where\s+is"
    r"|how\s+many|capital\s+of)\b",
    re.IGNORECASE,
)

# Prose threshold: queries ≥ 6 words that didn't match above are open-ended.
_PROSE_MIN_WORDS = 6


class DomainRouter:
    """Classifies a user query into one of six fine-tune domains.

    Domains match the categories in scripts/synth_dataset.py. The classifier
    is pure regex with no ML — worst-case latency is well under 1ms.

    Priority order (earlier wins):
        refusal_no_apology → tool_call → out_of_scope → knowledge_limit
        → concise_factual → plain_prose → general
    """

    def classify(self, text: str) -> str:
        """Return the domain label for *text*.

        Never raises. Returns ``"general"`` for empty, whitespace-only, or
        ambiguous input.
        """
        stripped = text.strip()
        if not stripped:
            return "general"

        if _REFUSAL_PATTERN.search(stripped):
            return "refusal_no_apology"

        if _TOOL_CALL_PATTERN.search(stripped):
            return "tool_call"

        if _OUT_OF_SCOPE_PATTERN.search(stripped):
            return "out_of_scope"

        if _KNOWLEDGE_LIMIT_PATTERN.search(stripped):
            return "knowledge_limit"

        if _CONCISE_FACTUAL_PATTERN.search(stripped):
            return "concise_factual"

        if len(stripped.split()) >= _PROSE_MIN_WORDS:
            return "plain_prose"

        return "general"
