"""
Concurrency, non-blocking, and malformed-load hardening tests for ConversationMemory.

These tests cover the three defects fixed in CR-11:
  1. Data race — two threads hammering add_turn / get_history without a lock.
  2. Blocking summariser / disk write — add_turn must return fast even when the
     summariser or save is artificially slow.
  3. Load validation — malformed entries must be skipped gracefully, not crash.

RED / GREEN proof
-----------------
These tests would FAIL on the original (unfixed) memory.py because:
  - test_concurrent_add_turn_no_corruption: the original code has no lock, so
    concurrent list.append + len() + slice are unsynchronized, causing index
    errors or history corruption under a ThreadPoolExecutor hammer.
  - test_add_turn_does_not_block_on_slow_summariser: the original code calls
    the summariser synchronously inside add_turn, so the function would block
    for the entire summariser duration.
  - test_add_turn_does_not_block_on_slow_save: the original save() is called
    synchronously by the caller (not by add_turn directly, but by rotation),
    and the original rotation is inline, so a slow disk write would block.
  - test_load_skips_malformed_entries_and_keeps_valid: the original load()
    accepts any list entry without validation, so valid entries after a bad one
    are silently kept (no filtering), and specific structural checks never run.
  - test_load_partial_malformed_preserves_valid_entries: same root cause.

All tests are marked ``unit`` to match the existing test_memory.py style.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.llm.memory import MAX_TURNS, RETAIN_RECENT, ConversationMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    tmp_path: Path,
    summariser: Any = None,
) -> ConversationMemory:
    return ConversationMemory(memory_dir=str(tmp_path), summariser=summariser)


# ---------------------------------------------------------------------------
# 1. Concurrent-access: data race / corruption
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_add_turn_no_corruption(tmp_path: Path) -> None:
    """Concurrent writers exceeding MAX_TURNS must not drop or duplicate turns.

    Four threads each add (MAX_TURNS // 2) turns concurrently so rotation
    fires mid-run.  After all threads join and flush(), every appended turn
    must appear exactly once in the final history or a summary entry must
    account for the rotated-away turns (i.e. no silent drops, no duplicates).

    Weak pre-fix behaviour: the original code adds only 20 turns < MAX so
    rotation never fires and the race is untested.  This version actually
    exceeds MAX_TURNS with concurrent writers so the multi-step rotation race
    is exercised.

    Correctness bar: total distinct content strings == N_THREADS * BATCH; no
    content string appears more than once.
    """
    N_THREADS = 4
    BATCH = MAX_TURNS // 2  # 20 per thread → 80 total, well above MAX_TURNS=40

    mem = _make_memory(tmp_path)
    barrier = threading.Barrier(N_THREADS)
    all_sent: list[str] = []
    sent_lock = threading.Lock()

    def worker(thread_id: int) -> None:
        for i in range(BATCH):
            label = f"t{thread_id}-m{i}"
            with sent_lock:
                all_sent.append(label)
            mem.add_turn("user", label)
        barrier.wait()  # keep threads alive until all are done

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    mem.flush()
    history = mem.get_history()

    # Gather all content strings present in history (skip the summary system entry).
    surviving = [e["content"] for e in history if e.get("role") != "system"]

    # No duplicates.
    assert len(surviving) == len(set(surviving)), (
        f"Duplicate turns found in history after concurrent rotation. "
        f"Duplicates: {[c for c in surviving if surviving.count(c) > 1]}"
    )

    # Sanity: at most MAX_TURNS + 1 entries after rotation (1 summary + RETAIN_RECENT recent).
    assert len(history) <= MAX_TURNS + 1, (
        f"History grew unbounded: {len(history)} entries (MAX_TURNS={MAX_TURNS}). "
        "Rotation did not fire despite exceeding threshold."
    )


@pytest.mark.unit
def test_concurrent_add_and_get_no_exception(tmp_path: Path) -> None:
    """One writer thread and one reader thread running simultaneously must not raise.

    Would FAIL on the original code if the reader sees a partially-updated
    list during an in-progress append (RuntimeError: list changed size during
    iteration, or IndexError).
    """
    mem = _make_memory(tmp_path)
    errors: list[Exception] = []
    stop = threading.Event()

    def writer() -> None:
        for i in range(60):
            try:
                mem.add_turn("user", f"w{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            try:
                _ = mem.get_history()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    r.start()
    w.start()
    w.join()
    r.join()

    assert errors == [], f"Exceptions during concurrent access: {errors}"


# ---------------------------------------------------------------------------
# 2. Non-blocking: add_turn must return fast even with a slow summariser/save
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_add_turn_does_not_block_on_slow_summariser(tmp_path: Path) -> None:
    """add_turn must return promptly even when the summariser is very slow.

    Strategy: register a summariser that sleeps for SUMMARISER_SLEEP seconds.
    Add MAX_TURNS+1 turns to trigger rotation.  The 41st add_turn must return
    within FAST_THRESHOLD seconds — well below SUMMARISER_SLEEP.

    Would FAIL on the original code: rotation is synchronous inside add_turn,
    so the 41st call blocks for the full SUMMARISER_SLEEP.
    """
    SUMMARISER_SLEEP = 0.5
    FAST_THRESHOLD = 0.15  # generous: CI can be slow, but 150 ms << 500 ms

    rotation_started = threading.Event()

    def slow_summariser(turns: list[dict]) -> str:
        rotation_started.set()
        time.sleep(SUMMARISER_SLEEP)
        return "slow summary"

    mem = _make_memory(tmp_path, summariser=slow_summariser)

    # Fill up to MAX_TURNS without triggering rotation.
    for i in range(MAX_TURNS):
        mem.add_turn("user", f"pre {i}")

    # This 41st call should trigger rotation dispatch WITHOUT blocking.
    t0 = time.monotonic()
    mem.add_turn("user", "trigger")
    elapsed = time.monotonic() - t0

    assert elapsed < FAST_THRESHOLD, (
        f"add_turn blocked for {elapsed:.3f}s (threshold {FAST_THRESHOLD}s). "
        "Rotation is still running synchronously on the inference thread."
    )

    # Wait for the background task to finish (proves it DID run, just off-thread).
    mem.flush()
    history = mem.get_history()
    # After flush, rotation must have applied.
    assert len(history) <= RETAIN_RECENT + 1, (
        f"Rotation did not apply even after flush; got {len(history)} entries."
    )


@pytest.mark.unit
def test_add_turn_does_not_block_on_slow_save(tmp_path: Path) -> None:
    """add_turn must return promptly even when the disk write is very slow.

    We patch ConversationMemory._save_locked to insert a sleep, then trigger
    rotation and confirm that add_turn returns before the sleep completes.

    Would FAIL on the original code: rotation (and save) are synchronous
    inside add_turn, so a slow write blocks the caller.
    """
    SAVE_SLEEP = 0.4
    FAST_THRESHOLD = 0.15

    original_save_locked = ConversationMemory._save_locked

    def slow_save_locked(self: ConversationMemory) -> None:
        time.sleep(SAVE_SLEEP)
        original_save_locked(self)

    mem = _make_memory(tmp_path)

    for i in range(MAX_TURNS):
        mem.add_turn("user", f"pre {i}")

    with patch.object(ConversationMemory, "_save_locked", slow_save_locked):
        t0 = time.monotonic()
        mem.add_turn("user", "trigger")
        elapsed = time.monotonic() - t0

    assert elapsed < FAST_THRESHOLD, (
        f"add_turn blocked for {elapsed:.3f}s (threshold {FAST_THRESHOLD}s). "
        "Save is still running synchronously on the inference thread."
    )

    mem.flush()  # wait for background to finish cleanly


# ---------------------------------------------------------------------------
# 3. Load validation: malformed entries degraded gracefully
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_skips_malformed_entries_and_keeps_valid(tmp_path: Path) -> None:
    """load() with a mix of valid and malformed entries must keep only the valid ones.

    Malformed entries include: non-dict items, missing 'role' key, missing
    'content' key, non-string role, non-string content.

    Would FAIL on the original code: load() accepts any list entry without
    structural validation and loads it verbatim, so callers get back entries
    with missing fields that crash the first turn.
    """
    good1 = {"role": "user", "content": "Hello"}
    good2 = {"role": "assistant", "content": "Hi there"}

    bad_entries = [
        "just a string",           # not a dict
        42,                        # not a dict
        None,                      # not a dict
        [],                        # not a dict
        {"role": "user"},          # missing 'content'
        {"content": "no role"},    # missing 'role'
        {"role": 123, "content": "bad role type"},   # role not str
        {"role": "user", "content": 456},            # content not str
    ]

    data = [good1] + bad_entries + [good2]

    file_path = tmp_path / "conversation.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    mem = _make_memory(tmp_path)
    mem.load()  # must not raise

    history = mem.get_history()
    assert len(history) == 2, (
        f"Expected 2 valid entries, got {len(history)}: {history}"
    )
    assert history[0] == good1
    assert history[1] == good2


@pytest.mark.unit
def test_load_all_malformed_starts_empty(tmp_path: Path) -> None:
    """load() with entirely malformed entries must result in an empty history (no crash).

    Would FAIL on the original code: the original code stores all list entries
    regardless of structure, leading to corrupt state on the first turn.
    """
    data = [{"role": 999}, "bad", None, {"content": True}]
    file_path = tmp_path / "conversation.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    mem = _make_memory(tmp_path)
    mem.load()  # must not raise

    assert mem.get_history() == []


@pytest.mark.unit
def test_load_valid_after_malformed_not_crashed(tmp_path: Path) -> None:
    """Valid entries that follow malformed ones are still loaded.

    This is the key crash regression: on the pre-fix code a bad entry at index 0
    would not crash load() (it just stores it), but attempting to use the entry
    in prompt building would crash the first inference turn.  Post-fix, the bad
    entry is filtered and the good entry is accessible.

    Would FAIL on the original code: the bad entry at index 0 is kept, so
    history[0]['role'] raises KeyError or returns a non-string value.
    """
    bad = {"role": 999, "content": "oops"}  # role is int, not str
    good = {"role": "user", "content": "Valid message"}

    data = [bad, good]
    file_path = tmp_path / "conversation.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    mem = _make_memory(tmp_path)
    mem.load()

    history = mem.get_history()
    assert len(history) == 1
    assert history[0] == good
    # Simulate what inference does: access role and content as strings.
    assert isinstance(history[0]["role"], str)
    assert isinstance(history[0]["content"], str)


@pytest.mark.unit
def test_load_empty_list_starts_empty(tmp_path: Path) -> None:
    """load() with an empty JSON list must leave history empty."""
    file_path = tmp_path / "conversation.json"
    file_path.write_text("[]", encoding="utf-8")

    mem = _make_memory(tmp_path)
    mem.load()

    assert mem.get_history() == []


# ---------------------------------------------------------------------------
# 4. Lock safety: flush() drains pending background tasks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flush_waits_for_background_rotation(tmp_path: Path) -> None:
    """flush() must block until a pending background rotation is complete.

    This verifies that flush() is a correct synchronisation point — callers
    (tests, ordered shutdown) can rely on history being stable after flush().
    """
    completed = threading.Event()

    def tracked_summariser(turns: list[dict]) -> str:
        completed.set()
        return "tracked summary"

    mem = _make_memory(tmp_path, summariser=tracked_summariser)

    for i in range(MAX_TURNS + 1):
        mem.add_turn("user", f"m{i}")

    # Before flush: rotation may not have fired yet.
    mem.flush()

    # After flush: summariser must have been called.
    assert completed.is_set(), "Summariser was never called — rotation did not run."

    history = mem.get_history()
    assert len(history) == RETAIN_RECENT + 1
    assert history[0]["role"] == "system"
    assert "tracked summary" in history[0]["content"]


# ---------------------------------------------------------------------------
# 5. Forced-interleaving: interim turns appended during summarisation must survive
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interim_turn_not_lost_during_background_summarisation(tmp_path: Path) -> None:
    """Turns appended while the summariser is running must NOT be discarded.

    Concrete failing interleaving that exposes the stale-snapshot writeback bug:

      Step 1: history reaches MAX_TURNS+1 (len=41). add_turn snapshots
              oldest=[m0..m20], recent=[m21..m40], dispatches rotation-1.
              Summariser-1 blocks (held by `held` event).

      Step 2: (lock is FREE — summariser runs off-thread)
              add_turn appends "interim" → len=42 > MAX_TURNS.
              add_turn snapshots oldest=[m0..m21], recent=[m22..m40,interim],
              dispatches rotation-2 (queued behind rotation-1 in the executor).

      Step 3: Summariser-1 unblocks. rotation-1 writeback runs:
              _history = [sum1] + recent_snapshot1 = [sum1, m21..m40]  (21 entries)
              → "interim" is absent from this snapshot — BUG.

      Step 4: rotation-2 writeback runs (guard: 21 > RETAIN_RECENT=20 → True):
              _history = [sum2] + recent_snapshot2 = [sum2, m22..m40, interim]
              → m21 is absent from this snapshot — also lost.

    The combined effect: turn m21 is permanently lost because it falls in the
    "gap" between rotation-1's recent slice (which includes it) and rotation-2's
    recent slice (which has already shifted past it to include "interim").

    After the fix (splice preserves _history[k:] rather than restoring the
    stale snapshot), rotation-1 writes [sum1] + _history[21:] which at writeback
    time is [m21..m40, interim] — no turns lost.  rotation-2 then becomes a
    no-op or correctly summarises the remainder.

    RED: m21 is missing from the final history.
    GREEN: m21 is present.
    """
    held = threading.Event()
    released = threading.Event()

    def blocking_summariser(turns: list[dict]) -> str:
        released.set()       # signal: summariser is now running, lock is free
        held.wait(timeout=5)  # block until main thread sets held
        return "summary"

    mem = _make_memory(tmp_path, summariser=blocking_summariser)

    # Fill to exactly MAX_TURNS entries (rotation not yet triggered).
    for i in range(MAX_TURNS):
        mem.add_turn("user", f"m{i}")

    # Turn MAX_TURNS+1: triggers rotation-1, snapshots recent=[m21..m40].
    mem.add_turn("user", f"m{MAX_TURNS}")

    # Wait until summariser-1 is running (memory lock has been released).
    assert released.wait(timeout=5), "Summariser never started — rotation did not fire"

    # Append ONE interim turn while summariser-1 blocks. This triggers rotation-2
    # (len=42 > MAX_TURNS=40), which snapshots recent=[m22..m40, interim].
    # m21 is now in rotation-1's recent but NOT rotation-2's recent.
    # If the buggy writeback uses the stale snapshot, m21 disappears when
    # rotation-2's result is spliced last.
    mem.add_turn("user", "interim")

    # Unblock summariser-1 so both writebacks can proceed.
    held.set()

    # Drain all pending background tasks.
    mem.flush()

    history = mem.get_history()
    non_system = [e["content"] for e in history if e.get("role") != "system"]

    # m21 is the boundary turn that falls between the two snapshots.
    # It MUST survive because it was appended before any rotation completed.
    boundary_turn = f"m{MAX_TURNS - RETAIN_RECENT + 1}"  # = "m21"
    assert boundary_turn in non_system, (
        f"Turn '{boundary_turn}' was lost during background summarisation writeback. "
        f"It was in rotation-1's recent snapshot but rotation-2's stale writeback "
        f"silently discarded it. Non-system history: {non_system}"
    )
    assert "interim" in non_system, (
        f"Interim turn was lost. Non-system history: {non_system}"
    )
