"""Conversation memory with JSON persistence for Project Lumi."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


# Threshold at which rotation is triggered. Promote to LumiConfig.llm.memory_max_turns
# when user-configurable memory size becomes a requested feature.
MAX_TURNS: int = 40

# Turns to keep verbatim after rotation. The older turns are summarized into a
# single system message. Must be < MAX_TURNS.
RETAIN_RECENT: int = 20


class ConversationMemory:
    """Stores conversation turns with optional JSON persistence and LLM-based rotation.

    Rotation strategy (triggered when len(history) > MAX_TURNS):
    - If a summariser callable is registered: the oldest (len - RETAIN_RECENT) turns
      are passed to the summariser, which returns a summary string. They are replaced
      by a single system message: {"role": "system", "content": "Summary of earlier
      conversation: <text>"}. The newest RETAIN_RECENT turns are kept verbatim.
    - If no summariser is registered: simple truncation to RETAIN_RECENT (no LLM call,
      no crash — graceful degradation).
    - If the summariser raises: fall back to truncation and log a warning.
    """

    def __init__(
        self,
        memory_dir: str,
        summariser: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        expanded = Path(memory_dir).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, str]] = []
        self._file: Path = expanded / "conversation.json"
        self._summariser = summariser

    def set_summariser(
        self,
        callback: Callable[[list[dict[str, str]]], str] | None,
    ) -> None:
        """Inject or replace the summariser callable after construction."""
        self._summariser = callback

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn and rotate when the history exceeds MAX_TURNS."""
        self._history.append({"role": role, "content": content})
        self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        """Summarize-and-replace when history exceeds MAX_TURNS."""
        if len(self._history) <= MAX_TURNS:
            return

        recent = self._history[-RETAIN_RECENT:]
        oldest = self._history[:-RETAIN_RECENT]

        if self._summariser is not None:
            try:
                summary_text = self._summariser(oldest)
                summary_entry: dict[str, str] = {
                    "role": "system",
                    "content": f"Summary of earlier conversation: {summary_text}",
                }
                self._history = [summary_entry] + recent
            except Exception:
                logger.warning(
                    "Memory summariser raised — falling back to truncation",
                    exc_info=True,
                )
                self._history = recent
        else:
            self._history = recent

    def get_history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    def prune(self, max_turns: int) -> None:
        """Keep only the last *max_turns* entries, discarding the oldest."""
        self._history = self._history[-max_turns:]

    def clear(self) -> None:
        """Wipe in-memory history and delete the persistence file if it exists."""
        self._history = []
        if self._file.exists():
            try:
                self._file.unlink()
            except OSError:
                logger.warning("Failed to delete persistence file: %s", self._file)

    def save(self) -> None:
        """Write current history to the JSON persistence file."""
        try:
            with self._file.open("w", encoding="utf-8") as fh:
                json.dump(self._history, fh, indent=2, ensure_ascii=False)
        except OSError:
            logger.error("Failed to save conversation history to %s", self._file)

    def load(self) -> None:
        """Load history from the JSON file and rotate if needed."""
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._history = data
                self._maybe_rotate()
            else:
                logger.warning("Unexpected format in %s — starting fresh", self._file)
                self._history = []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s (%s) — starting fresh", self._file, exc)
            self._history = []
