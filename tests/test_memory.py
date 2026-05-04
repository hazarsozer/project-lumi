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


# ---------------------------------------------------------------------------
# Rotation — triggered by add_turn() when len(history) > MAX_TURNS
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rotation_triggers_at_max_turns_plus_one(tmp_path: Path) -> None:
    """Rotation must fire when len(history) first exceeds MAX_TURNS."""
    from src.llm.memory import MAX_TURNS, RETAIN_RECENT

    mem = ConversationMemory(
        memory_dir=str(tmp_path),
        summariser=lambda turns: "stub summary",
    )
    for i in range(MAX_TURNS + 1):
        mem.add_turn("user", f"message {i}")

    history = mem.get_history()
    assert len(history) == RETAIN_RECENT + 1
    assert history[0]["role"] == "system"
    assert "Summary of earlier conversation:" in history[0]["content"]
    assert "stub summary" in history[0]["content"]


@pytest.mark.unit
def test_rotation_keeps_newest_turns_verbatim(tmp_path: Path) -> None:
    """The RETAIN_RECENT newest turns must survive rotation unchanged."""
    from src.llm.memory import MAX_TURNS, RETAIN_RECENT

    mem = ConversationMemory(
        memory_dir=str(tmp_path),
        summariser=lambda turns: "summary",
    )
    for i in range(MAX_TURNS + 1):
        mem.add_turn("user", f"message {i}")

    history = mem.get_history()
    assert history[-1]["content"] == f"message {MAX_TURNS}"
    assert history[1]["content"] == f"message {MAX_TURNS - RETAIN_RECENT + 1}"


@pytest.mark.unit
def test_rotation_no_summariser_truncates_gracefully(tmp_path: Path) -> None:
    """Without a summariser, rotation silently truncates to RETAIN_RECENT (no crash)."""
    from src.llm.memory import MAX_TURNS, RETAIN_RECENT

    mem = ConversationMemory(memory_dir=str(tmp_path))  # no summariser
    for i in range(MAX_TURNS + 1):
        mem.add_turn("user", f"message {i}")

    history = mem.get_history()
    assert len(history) == RETAIN_RECENT
    assert all(e["role"] == "user" for e in history)


@pytest.mark.unit
def test_rotation_summariser_exception_falls_back_to_truncation(tmp_path: Path) -> None:
    """A summariser that raises must not crash — rotation falls back to truncation."""
    from src.llm.memory import MAX_TURNS, RETAIN_RECENT

    def failing_summariser(turns: list[dict]) -> str:
        raise RuntimeError("LLM unavailable")

    mem = ConversationMemory(memory_dir=str(tmp_path), summariser=failing_summariser)
    for i in range(MAX_TURNS + 1):
        mem.add_turn("user", f"message {i}")

    history = mem.get_history()
    assert len(history) == RETAIN_RECENT


@pytest.mark.unit
def test_no_rotation_below_threshold(tmp_path: Path) -> None:
    """Summariser must not be called when len(history) <= MAX_TURNS."""
    from src.llm.memory import MAX_TURNS

    call_count = [0]

    def counting_summariser(turns: list[dict]) -> str:
        call_count[0] += 1
        return "should not appear"

    mem = ConversationMemory(memory_dir=str(tmp_path), summariser=counting_summariser)
    for i in range(MAX_TURNS):  # exactly at threshold — no rotation
        mem.add_turn("user", f"message {i}")

    assert call_count[0] == 0
    assert len(mem.get_history()) == MAX_TURNS


@pytest.mark.unit
def test_set_summariser_post_construction(tmp_path: Path) -> None:
    """set_summariser must allow injecting a callable after construction."""
    from src.llm.memory import MAX_TURNS, RETAIN_RECENT

    mem = ConversationMemory(memory_dir=str(tmp_path))
    mem.set_summariser(lambda turns: "injected summary")

    for i in range(MAX_TURNS + 1):
        mem.add_turn("user", f"message {i}")

    history = mem.get_history()
    assert len(history) == RETAIN_RECENT + 1
    assert "injected summary" in history[0]["content"]


@pytest.mark.unit
def test_rotation_on_load(tmp_path: Path) -> None:
    """load() must trigger rotation if the persisted file has > MAX_TURNS entries."""
    import json as _json
    from src.llm.memory import MAX_TURNS, RETAIN_RECENT

    turns = [{"role": "user", "content": f"old {i}"} for i in range(MAX_TURNS + 5)]
    file_path = tmp_path / "conversation.json"
    file_path.write_text(_json.dumps(turns), encoding="utf-8")

    summariser_called = [False]

    def stub_summariser(old: list[dict]) -> str:
        summariser_called[0] = True
        return "past summary"

    mem = ConversationMemory(memory_dir=str(tmp_path), summariser=stub_summariser)
    mem.load()

    history = mem.get_history()
    assert len(history) == RETAIN_RECENT + 1
    assert summariser_called[0]
    assert "past summary" in history[0]["content"]
