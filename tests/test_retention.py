"""
Tests for retention / purge:
  - ConversationMemory: age-based entry purge (CR-09, Issue #10)
  - DocumentStore: chunk purge via purge_expired_chunks() (CR-09, Issue #10)

Clock-injection strategy
------------------------
All time-sensitive assertions use injected clocks (``_now_fn`` for memory,
``now=`` parameter for DocumentStore.purge_expired_chunks) so that tests are
fully deterministic — no real wall-clock sleeps are used anywhere.

RED → GREEN proof
-----------------
These tests would FAIL on the pre-patch code because:
  - purge_old_entries() did not exist (NameError / AttributeError).
  - add_turn() did not stamp entries with 'ts'.
  - load() did not call _purge_locked().
  - _save_locked() did not call _purge_locked().
  - purge_expired_chunks() did not exist on DocumentStore.
After the patch all tests pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.core.config import LLMConfig, RAGConfig
from src.llm.memory import ConversationMemory

# ---------------------------------------------------------------------------
# Conditional import for RAG (requires sqlite_vec extra)
# ---------------------------------------------------------------------------
try:
    import sqlite_vec as _sqlite_vec_module  # noqa: F401 — just to test availability
    from src.rag.store import DocumentStore

    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False
    DocumentStore = None  # type: ignore[assignment,misc]

_skip_rag = pytest.mark.skipif(
    not _RAG_AVAILABLE,
    reason="RAG extras not installed (uv sync --extra rag)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 4  # tiny dim, keeps tests fast


def _make_memory(
    tmp_path: Path,
    max_age_s: float = 0.0,
    now_fn=None,
) -> ConversationMemory:
    return ConversationMemory(
        memory_dir=str(tmp_path),
        max_age_s=max_age_s,
        _now_fn=now_fn,
    )


def _make_store(tmp_path: Path):
    """Create and initialise a DocumentStore backed by a temp DB file."""
    db = tmp_path / "test_rag.db"
    cfg = RAGConfig(db_path=str(db))
    s = DocumentStore(cfg, embedding_dim=EMBEDDING_DIM)  # type: ignore[misc]
    s.init_schema()
    return s


def _insert_doc_and_chunk(store, path: str, ingested_at: float, text: str = "hello world"):
    """Insert a document with a custom ingested_at timestamp, then one chunk.

    Bypasses upsert_document to control the exact timestamp.
    Returns (doc_id, chunk_id).
    """
    conn = store._conn()
    conn.execute(
        "INSERT OR REPLACE INTO documents(path, sha256, ingested_at) VALUES (?,?,?)",
        (path, "sha_" + path.replace("/", "_")[-8:], ingested_at),
    )
    conn.commit()
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE path = ?", (path,)
    ).fetchone()[0]
    chunk_cur = conn.execute(
        "INSERT INTO chunks(document_id, chunk_idx, text, char_start, char_end) "
        "VALUES (?,?,?,?,?)",
        (doc_id, 0, text, 0, len(text)),
    )
    conn.commit()
    return doc_id, chunk_cur.lastrowid


# ===========================================================================
# ConversationMemory retention tests
# ===========================================================================


class TestConversationMemoryRetention:
    """Age-based purge for ConversationMemory."""

    # -----------------------------------------------------------------------
    # purge_old_entries() public API
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_purge_disabled_when_max_age_zero(self, tmp_path: Path) -> None:
        """purge_old_entries() is a no-op when max_age_s=0 (default)."""
        clock = [0.0]
        mem = _make_memory(tmp_path, max_age_s=0.0, now_fn=lambda: clock[0])
        clock[0] = 1000.0
        mem.add_turn("user", "old message")
        clock[0] = 2000.0
        removed = mem.purge_old_entries()
        assert removed == 0
        assert len(mem.get_history()) == 1

    @pytest.mark.unit
    def test_purge_removes_entries_older_than_max_age(self, tmp_path: Path) -> None:
        """Entries older than max_age_s must be removed; recent ones kept."""
        clock = [0.0]
        max_age_s = 3600.0  # 1 hour
        mem = _make_memory(tmp_path, max_age_s=max_age_s, now_fn=lambda: clock[0])

        # "old" entry: added at t=0.
        clock[0] = 0.0
        mem.add_turn("user", "old message")

        # "recent" entry: added at t = max_age_s + 1 (just after the boundary).
        clock[0] = max_age_s + 1.0
        mem.add_turn("user", "recent message")

        # Advance clock so "old" is more than max_age_s in the past.
        # cutoff = (max_age_s + 2) - max_age_s = 2 > ts_of_old=0 → expired.
        clock[0] = max_age_s + 2.0
        removed = mem.purge_old_entries()

        assert removed == 1
        history = mem.get_history()
        assert len(history) == 1
        assert history[0]["content"] == "recent message"

    @pytest.mark.unit
    def test_purge_all_entries_when_all_old(self, tmp_path: Path) -> None:
        """When every entry is expired, purge returns the full count and history is empty."""
        clock = [0.0]
        max_age_s = 60.0
        mem = _make_memory(tmp_path, max_age_s=max_age_s, now_fn=lambda: clock[0])

        clock[0] = 0.0
        for i in range(5):
            mem.add_turn("user", f"old {i}")

        clock[0] = max_age_s + 10.0  # advance past expiry
        removed = mem.purge_old_entries()

        assert removed == 5
        assert mem.get_history() == []

    @pytest.mark.unit
    def test_purge_keeps_entries_without_timestamp(self, tmp_path: Path) -> None:
        """Entries without a 'ts' field (legacy / summary entries) must be preserved."""
        clock = [1000.0]
        max_age_s = 60.0
        mem = _make_memory(tmp_path, max_age_s=max_age_s, now_fn=lambda: clock[0])

        # Manually inject a legacy entry (no 'ts' field) by writing directly.
        with mem._lock:
            mem._history.append({"role": "system", "content": "legacy summary"})
            mem._history.append({"role": "user", "content": "also no ts"})

        clock[0] = 2000.0  # far in the future — any timestamped entry would expire
        removed = mem.purge_old_entries()

        # Legacy entries (no 'ts') must survive.
        assert removed == 0
        history = mem.get_history()
        assert len(history) == 2

    @pytest.mark.unit
    def test_purge_returns_zero_no_expired(self, tmp_path: Path) -> None:
        """Returns 0 when no entries have expired."""
        clock = [0.0]
        max_age_s = 3600.0
        mem = _make_memory(tmp_path, max_age_s=max_age_s, now_fn=lambda: clock[0])

        clock[0] = 100.0
        mem.add_turn("user", "fresh message")
        clock[0] = 200.0  # 100 s later — well within max_age_s=3600 s
        removed = mem.purge_old_entries()
        assert removed == 0

    # -----------------------------------------------------------------------
    # Purge on save()
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_save_purges_old_entries(self, tmp_path: Path) -> None:
        """save() must purge expired entries before writing to disk."""
        clock = [0.0]
        max_age_s = 60.0
        mem = _make_memory(tmp_path, max_age_s=max_age_s, now_fn=lambda: clock[0])

        clock[0] = 0.0
        mem.add_turn("user", "will expire")
        clock[0] = 100.0
        mem.add_turn("user", "stays")

        # At t=130: "will expire" is 130s old > max_age_s=60 → purged.
        clock[0] = 130.0
        mem.save()

        # In-memory: old entry gone.
        history = mem.get_history()
        assert len(history) == 1
        assert history[0]["content"] == "stays"

        # On-disk: old entry also gone (re-load into a fresh instance with no
        # retention so the reload does not re-purge with a different clock).
        mem2 = _make_memory(tmp_path, max_age_s=0.0)
        mem2.load()
        disk_history = mem2.get_history()
        assert len(disk_history) == 1
        assert disk_history[0]["content"] == "stays"

    # -----------------------------------------------------------------------
    # Purge on load()
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_load_purges_old_entries(self, tmp_path: Path) -> None:
        """load() must purge expired entries before applying rotation."""
        now_ref = 10_000.0
        max_age_s = 3600.0

        entries = [
            {"role": "user", "content": "very old", "ts": now_ref - max_age_s - 1.0},
            {"role": "user", "content": "within age", "ts": now_ref - max_age_s + 1.0},
            {"role": "assistant", "content": "recent", "ts": now_ref - 10.0},
        ]
        (tmp_path / "conversation.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )

        mem = _make_memory(
            tmp_path,
            max_age_s=max_age_s,
            now_fn=lambda: now_ref,
        )
        mem.load()

        history = mem.get_history()
        contents = [e["content"] for e in history]
        assert "very old" not in contents, "Expired entry must be purged on load"
        assert "within age" in contents, "Entry within max_age_s must survive"
        assert "recent" in contents, "Recent entry must survive"

    @pytest.mark.unit
    def test_load_purge_uses_injected_clock(self, tmp_path: Path) -> None:
        """load() purge uses the injected _now_fn, not wall-clock time."""
        frozen_now = 50_000.0
        max_age_s = 1_000.0

        entries = [
            {"role": "user", "content": "gone", "ts": frozen_now - max_age_s - 1},
            {"role": "user", "content": "kept", "ts": frozen_now - 1},
        ]
        (tmp_path / "conversation.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )

        mem = _make_memory(
            tmp_path,
            max_age_s=max_age_s,
            now_fn=lambda: frozen_now,
        )
        mem.load()

        history = mem.get_history()
        assert len(history) == 1
        assert history[0]["content"] == "kept"

    # -----------------------------------------------------------------------
    # Thread safety: purge uses the lock
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_purge_is_threadsafe(self, tmp_path: Path) -> None:
        """purge_old_entries() and add_turn() running concurrently must not raise."""
        import threading

        # Use a list-backed mutable clock to avoid race on float.
        clock_val = [0.0]
        clock_lock = threading.Lock()
        max_age_s = 10.0

        def now_fn() -> float:
            with clock_lock:
                return clock_val[0]

        mem = _make_memory(tmp_path, max_age_s=max_age_s, now_fn=now_fn)
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(50):
                try:
                    with clock_lock:
                        clock_val[0] = float(i)
                    mem.add_turn("user", f"msg {i}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        def purger() -> None:
            for _ in range(30):
                try:
                    with clock_lock:
                        clock_val[0] = 100.0
                    mem.purge_old_entries()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=purger)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent purge/add raised: {errors}"

    # -----------------------------------------------------------------------
    # Config integration: LLMConfig.memory_max_age_days field
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_llm_config_has_memory_max_age_days(self) -> None:
        """LLMConfig must have a memory_max_age_days field with a default of 0.0 (opt-in)."""
        cfg = LLMConfig()
        assert hasattr(cfg, "memory_max_age_days"), (
            "LLMConfig must have 'memory_max_age_days' field (added in CR-09)"
        )
        assert cfg.memory_max_age_days == 0.0

    @pytest.mark.unit
    def test_llm_config_memory_max_age_days_zero_disables_retention(self) -> None:
        """Setting memory_max_age_days=0.0 in config converts to max_age_s=0.0."""
        cfg = LLMConfig(memory_max_age_days=0.0)
        max_age_s = cfg.memory_max_age_days * 86_400.0
        assert max_age_s == 0.0

    # -----------------------------------------------------------------------
    # get_history() strips 'ts' field
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_get_history_does_not_expose_ts_field(self, tmp_path: Path) -> None:
        """get_history() must return only {role, content} — 'ts' is internal."""
        clock = [500.0]
        mem = _make_memory(tmp_path, max_age_s=3600.0, now_fn=lambda: clock[0])
        mem.add_turn("user", "hello")
        history = mem.get_history()
        assert len(history) == 1
        assert set(history[0].keys()) == {"role", "content"}, (
            f"get_history() must return only {{role, content}}, got: {history[0].keys()}"
        )

    # -----------------------------------------------------------------------
    # RAGConfig retention field (no sqlite_vec needed — pure config test)
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_rag_config_has_rag_max_age_days(self) -> None:
        """RAGConfig must have a rag_max_age_days field with a default of 0.0."""
        cfg = RAGConfig()
        assert hasattr(cfg, "rag_max_age_days"), (
            "RAGConfig must have 'rag_max_age_days' field (added in CR-09)"
        )
        assert cfg.rag_max_age_days == 0.0

    @pytest.mark.unit
    def test_rag_config_custom_age(self) -> None:
        """RAGConfig.rag_max_age_days must accept a custom value."""
        cfg = RAGConfig(rag_max_age_days=90.0)
        assert cfg.rag_max_age_days == 90.0


# ===========================================================================
# DocumentStore retention tests
# ===========================================================================


@_skip_rag
class TestDocumentStoreRetention:
    """Age-based purge for DocumentStore."""

    @pytest.fixture()
    def store(self, tmp_path: Path):
        s = _make_store(tmp_path)
        yield s
        s.close()

    # -----------------------------------------------------------------------
    # purge_expired_chunks() — basic behaviour
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_purge_disabled_when_max_age_zero(self, store) -> None:
        """purge_expired_chunks(0.0) must be a no-op (returns 0)."""
        now = time.time()
        _insert_doc_and_chunk(store, "/old.md", ingested_at=now - 1_000_000.0)
        result = store.purge_expired_chunks(0.0, now=now)
        assert result == 0

    @pytest.mark.unit
    def test_purge_deletes_expired_chunks(self, store) -> None:
        """Chunks whose parent document is older than max_age_s must be deleted."""
        now = 1_000_000.0
        max_age_s = 3600.0

        old_doc_id, old_chunk_id = _insert_doc_and_chunk(
            store, "/old.md", ingested_at=now - max_age_s - 1.0
        )
        fresh_doc_id, fresh_chunk_id = _insert_doc_and_chunk(
            store, "/fresh.md", ingested_at=now - 60.0
        )

        deleted = store.purge_expired_chunks(max_age_s, now=now)

        assert deleted == 1, f"Expected 1 chunk deleted, got {deleted}"
        conn = store._conn()
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE id = ?", (old_chunk_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE id = ?", (fresh_chunk_id,)
        ).fetchone()[0] == 1

    @pytest.mark.unit
    def test_purge_also_deletes_document_row(self, store) -> None:
        """The document row must be removed along with its chunks."""
        now = 500_000.0
        max_age_s = 100.0
        old_doc_id, _ = _insert_doc_and_chunk(
            store, "/old_doc.md", ingested_at=now - max_age_s - 10.0
        )

        store.purge_expired_chunks(max_age_s, now=now)

        count = store._conn().execute(
            "SELECT COUNT(*) FROM documents WHERE id = ?", (old_doc_id,)
        ).fetchone()[0]
        assert count == 0

    @pytest.mark.unit
    def test_purge_removes_vectors_too(self, store) -> None:
        """Vector rows for purged chunks must also be deleted."""
        import sqlite_vec

        now = 999_000.0
        max_age_s = 1000.0
        old_doc_id, old_chunk_id = _insert_doc_and_chunk(
            store, "/vec_old.md", ingested_at=now - max_age_s - 1.0
        )
        conn = store._conn()
        conn.execute("DELETE FROM vectors WHERE chunk_id = ?", (old_chunk_id,))
        conn.execute(
            "INSERT INTO vectors(chunk_id, embedding) VALUES (?,?)",
            (old_chunk_id, sqlite_vec.serialize_float32([0.1, 0.2, 0.3, 0.4])),
        )
        conn.commit()

        store.purge_expired_chunks(max_age_s, now=now)

        vec_count = conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE chunk_id = ?", (old_chunk_id,)
        ).fetchone()[0]
        assert vec_count == 0

    @pytest.mark.unit
    def test_purge_fresh_docs_untouched(self, store) -> None:
        """Documents ingested recently must not be deleted."""
        now = 2_000_000.0
        max_age_s = 86_400.0

        _insert_doc_and_chunk(store, "/doc1.md", ingested_at=now - 3600.0)
        _insert_doc_and_chunk(store, "/doc2.md", ingested_at=now - 100.0)

        deleted = store.purge_expired_chunks(max_age_s, now=now)
        assert deleted == 0

        conn = store._conn()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2

    @pytest.mark.unit
    def test_purge_multiple_docs_some_expired(self, store) -> None:
        """Only expired documents are removed; fresh ones survive."""
        now = 5_000_000.0
        max_age_s = 1000.0

        _insert_doc_and_chunk(store, "/exp1.md", ingested_at=now - 2000.0)
        _insert_doc_and_chunk(store, "/exp2.md", ingested_at=now - 1500.0)
        _insert_doc_and_chunk(store, "/fresh.md", ingested_at=now - 500.0)

        deleted = store.purge_expired_chunks(max_age_s, now=now)
        assert deleted == 2

        conn = store._conn()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        remaining = conn.execute("SELECT path FROM documents").fetchone()[0]
        assert remaining == "/fresh.md"

    @pytest.mark.unit
    def test_purge_no_docs_returns_zero(self, store) -> None:
        """purge_expired_chunks on an empty store returns 0 without error."""
        result = store.purge_expired_chunks(3600.0, now=time.time())
        assert result == 0

    @pytest.mark.unit
    def test_purge_uses_injected_now_not_wall_clock(self, store) -> None:
        """Injected ``now`` must be used instead of time.time()."""
        _insert_doc_and_chunk(store, "/x.md", ingested_at=100.0)
        # max_age_s=50, now=200: cutoff=150, ingested_at=100 < 150 → expired.
        deleted = store.purge_expired_chunks(50.0, now=200.0)
        assert deleted == 1

    @pytest.mark.unit
    def test_purge_boundary_exactly_at_cutoff_is_kept(self, store) -> None:
        """ingested_at == (now - max_age_s) is NOT expired (strict < cutoff check)."""
        now = 1000.0
        max_age_s = 100.0
        # ingested_at = 900.0 = now - max_age_s → exactly at boundary, must be kept.
        _insert_doc_and_chunk(store, "/boundary.md", ingested_at=now - max_age_s)
        deleted = store.purge_expired_chunks(max_age_s, now=now)
        assert deleted == 0

