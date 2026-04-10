"""Conversation memory with JSON persistence for Project Lumi."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Stores conversation turns in memory with optional JSON file persistence."""

    def __init__(self, memory_dir: str) -> None:
        expanded = Path(memory_dir).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, str]] = []
        self._file: Path = expanded / "conversation.json"

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn to conversation history."""
        self._history.append({"role": role, "content": content})

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
        """Load history from the JSON file. No-op if the file does not exist."""
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._history = data
            else:
                logger.warning("Unexpected format in %s — starting fresh", self._file)
                self._history = []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s (%s) — starting fresh", self._file, exc)
            self._history = []
