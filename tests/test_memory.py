"""
Tests for src.llm.memory.ConversationMemory.

Mocking strategy
----------------
ConversationMemory persists to the filesystem.  All I/O tests use pytest's
``tmp_path`` fixture to write into a throwaway directory — no home-directory
side-effects occur during the test run.

All tests are marked ``unit``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# RED: these imports will fail until src/llm/memory.py is written.
from src.llm.memory import ConversationMemory  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(tmp_path: Path) -> ConversationMemory:
    """Return a ConversationMemory instance writing to a tmp directory."""
    return ConversationMemory(memory_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initial_history_empty(tmp_path: Path) -> None:
    """A freshly created ConversationMemory must have an empty history."""
    mem = _make_memory(tmp_path)
    assert mem.get_history() == []


@pytest.mark.unit
def test_add_turn_appended(tmp_path: Path) -> None:
    """add_turn must append entries in order with the correct role and content."""
    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Hello")
    mem.add_turn("assistant", "Hi there!")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


@pytest.mark.unit
def test_get_history_returns_copy(tmp_path: Path) -> None:
    """Mutating the list returned by get_history must not affect internal state."""
    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Test")
    history = mem.get_history()
    # Mutate the returned list.
    history.append({"role": "user", "content": "injected"})
    # Internal state must be unchanged.
    assert len(mem.get_history()) == 1


@pytest.mark.unit
def test_prune_keeps_last_n(tmp_path: Path) -> None:
    """prune(max_turns=N) must retain only the N most-recent turns."""
    mem = _make_memory(tmp_path)
    for i in range(10):
        mem.add_turn("user", f"message {i}")
    mem.prune(max_turns=3)
    history = mem.get_history()
    assert len(history) == 3
    # The three most recent messages (7, 8, 9) must be retained.
    assert history[-1]["content"] == "message 9"
    assert history[0]["content"] == "message 7"


@pytest.mark.unit
def test_clear_wipes_history(tmp_path: Path) -> None:
    """clear() must remove all turns from memory."""
    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Something")
    mem.clear()
    assert mem.get_history() == []


@pytest.mark.unit
def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """save() followed by load() on a new instance must restore the history."""
    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Persisted message")
    mem.add_turn("assistant", "Persisted reply")
    mem.save()

    # Create a fresh instance pointing at the same directory.
    mem2 = _make_memory(tmp_path)
    mem2.load()
    history = mem2.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "Persisted message"
    assert history[1]["content"] == "Persisted reply"


@pytest.mark.unit
def test_load_nonexistent_file_starts_empty(tmp_path: Path) -> None:
    """load() when no file exists must leave history empty (no exception raised)."""
    mem = _make_memory(tmp_path)
    # No save() was called — the file does not exist.
    mem.load()  # must not raise
    assert mem.get_history() == []


@pytest.mark.unit
def test_prune_noop_when_under_limit(tmp_path: Path) -> None:
    """prune() must be a no-op when the history is already within max_turns."""
    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Only one")
    mem.prune(max_turns=10)
    assert len(mem.get_history()) == 1


@pytest.mark.unit
def test_add_turn_accepts_system_role(tmp_path: Path) -> None:
    """add_turn must accept 'system' as a valid role without raising."""
    mem = _make_memory(tmp_path)
    mem.add_turn("system", "You are a helpful assistant.")
    history = mem.get_history()
    assert history[0]["role"] == "system"


# ---------------------------------------------------------------------------
# clear() — persistence file deletion error path (lines 37-40)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_clear_wipes_history_and_deletes_file(tmp_path: Path) -> None:
    """clear() must delete the persistence file when it exists."""
    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Before clear")
    mem.save()
    assert mem._file.exists()

    mem.clear()

    assert mem.get_history() == []
    assert not mem._file.exists()


@pytest.mark.unit
def test_clear_handles_oserror_on_file_deletion(tmp_path: Path) -> None:
    """clear() must not raise if unlinking the persistence file fails with OSError."""
    from unittest.mock import patch
    from pathlib import Path as _Path

    mem = _make_memory(tmp_path)
    mem.add_turn("user", "Something")
    mem.save()
    assert mem._file.exists()

    # Patch Path.unlink at the class level so the instance method raises OSError.
    with patch.object(_Path, "unlink", side_effect=OSError("permission denied")):
        # Must not raise despite the OSError.
        mem.clear()

    # In-memory history is still cleared.
    assert mem.get_history() == []


# ---------------------------------------------------------------------------
# save() — OSError during write (lines 47-48)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_logs_error_on_oserror(tmp_path: Path) -> None:
    """save() must not raise if an OSError occurs while writing the file."""
    from unittest.mock import patch
    from pathlib import Path as _Path

    mem = _make_memory(tmp_path)
    mem.add_turn("user", "This will fail to save")

    # Patch Path.open at the class level to raise OSError during write.
    with patch.object(_Path, "open", side_effect=OSError("disk full")):
        # Must not raise.
        mem.save()

    # In-memory history is unaffected by the save failure.
    assert len(mem.get_history()) == 1


# ---------------------------------------------------------------------------
# load() — unexpected JSON format (list vs. non-list) and decode errors (lines 60-64)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_unexpected_json_format_starts_fresh(tmp_path: Path) -> None:
    """load() when the JSON file contains a non-list value must start fresh."""
    import json

    mem = _make_memory(tmp_path)
    # Write a JSON object (not a list) to the persistence file.
    mem._file.write_text(json.dumps({"role": "user", "content": "oops"}), encoding="utf-8")

    mem.load()

    assert mem.get_history() == []


@pytest.mark.unit
def test_load_malformed_json_starts_fresh(tmp_path: Path) -> None:
    """load() when the JSON file is malformed must start fresh without raising."""
    mem = _make_memory(tmp_path)
    # Write deliberately broken JSON.
    mem._file.write_text("{ not valid json !!!", encoding="utf-8")

    mem.load()  # must not raise

    assert mem.get_history() == []


@pytest.mark.unit
def test_load_oserror_starts_fresh(tmp_path: Path) -> None:
    """load() when reading the file raises OSError must start fresh without raising."""
    from unittest.mock import patch
    from pathlib import Path as _Path

    mem = _make_memory(tmp_path)
    # Create the file so the existence check passes.
    mem._file.write_text("[]", encoding="utf-8")

    # Patch Path.open at the class level to raise OSError during read.
    with patch.object(_Path, "open", side_effect=OSError("read error")):
        mem.load()  # must not raise

    assert mem.get_history() == []
