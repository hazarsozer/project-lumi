"""Tests for DocumentStore — CRUD, FTS search, kNN vector search, stats."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from src.core.config import RAGConfig
from src.rag.store import (
    ChunkRecord,
    DocumentRecord,
    DocumentStore,
    SearchHit,
    StoreStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 4  # tiny dim keeps tests fast


def _make_config(tmp_path: Path) -> RAGConfig:
    db = tmp_path / "test_rag.db"
    return RAGConfig(db_path=str(db))


@pytest.fixture()
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(_make_config(tmp_path), embedding_dim=EMBEDDING_DIM)
    s.init_schema()
    yield s
    s.close()


def _insert_doc_and_chunk(
    store: DocumentStore,
    path: str = "/notes/a.md",
    text: str = "hello world",
    embedding: list[float] | None = None,
) -> tuple[DocumentRecord, ChunkRecord]:
    doc = store.upsert_document(path, sha256="abc123")
    chunk = store.insert_chunk(doc.id, 0, text, 0, len(text))
    if embedding is not None:
        store.insert_vector(chunk.id, embedding)
    return doc, chunk


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------


class TestInitSchema:
    def test_creates_documents_table(self, store: DocumentStore) -> None:
        count = store._conn().execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()[0]
        assert count == 1

    def test_creates_chunks_table(self, store: DocumentStore) -> None:
        count = store._conn().execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()[0]
        assert count == 1

    def test_creates_fts_table(self, store: DocumentStore) -> None:
        count = store._conn().execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='fts'"
        ).fetchone()[0]
        assert count >= 1

    def test_creates_vectors_table(self, store: DocumentStore) -> None:
        count = store._conn().execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='vectors'"
        ).fetchone()[0]
        assert count >= 1

    def test_wal_mode_enabled(self, store: DocumentStore) -> None:
        mode = store._conn().execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_idempotent(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        s = DocumentStore(cfg, embedding_dim=EMBEDDING_DIM)
        s.init_schema()
        s.init_schema()  # must not raise
        s.close()


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestUpsertDocument:
    def test_insert_returns_record(self, store: DocumentStore) -> None:
        doc = store.upsert_document("/a.md", "sha1")
        assert isinstance(doc, DocumentRecord)
        assert doc.path == "/a.md"
        assert doc.sha256 == "sha1"
        assert doc.id > 0

    def test_ingested_at_is_recent(self, store: DocumentStore) -> None:
        before = time.time()
        doc = store.upsert_document("/a.md", "sha1")
        assert doc.ingested_at >= before

    def test_upsert_updates_sha256(self, store: DocumentStore) -> None:
        store.upsert_document("/a.md", "old")
        doc = store.upsert_document("/a.md", "new")
        assert doc.sha256 == "new"

    def test_upsert_preserves_id(self, store: DocumentStore) -> None:
        first = store.upsert_document("/a.md", "v1")
        second = store.upsert_document("/a.md", "v2")
        assert first.id == second.id

    def test_get_by_path_returns_record(self, store: DocumentStore) -> None:
        store.upsert_document("/b.md", "sha2")
        doc = store.get_document_by_path("/b.md")
        assert doc is not None
        assert doc.sha256 == "sha2"

    def test_get_by_path_missing_returns_none(self, store: DocumentStore) -> None:
        assert store.get_document_by_path("/nonexistent.md") is None


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


class TestChunks:
    def test_insert_chunk_returns_record(self, store: DocumentStore) -> None:
        doc = store.upsert_document("/a.md", "h")
        chunk = store.insert_chunk(doc.id, 0, "hello", 0, 5)
        assert isinstance(chunk, ChunkRecord)
        assert chunk.text == "hello"
        assert chunk.chunk_idx == 0
        assert chunk.document_id == doc.id

    def test_get_chunk_by_id(self, store: DocumentStore) -> None:
        doc = store.upsert_document("/a.md", "h")
        inserted = store.insert_chunk(doc.id, 0, "world", 0, 5)
        fetched = store.get_chunk_by_id(inserted.id)
        assert fetched is not None
        assert fetched.text == "world"

    def test_get_chunk_by_id_missing_returns_none(
        self, store: DocumentStore
    ) -> None:
        assert store.get_chunk_by_id(99999) is None

    def test_multiple_chunks_per_document(self, store: DocumentStore) -> None:
        doc = store.upsert_document("/a.md", "h")
        c0 = store.insert_chunk(doc.id, 0, "chunk zero", 0, 10)
        c1 = store.insert_chunk(doc.id, 1, "chunk one", 10, 19)
        assert c0.id != c1.id
        assert c1.chunk_idx == 1

    def test_delete_document_chunks(self, store: DocumentStore) -> None:
        doc = store.upsert_document("/a.md", "h")
        store.insert_chunk(doc.id, 0, "text", 0, 4)
        store.insert_chunk(doc.id, 1, "more", 4, 8)
        deleted = store.delete_document_chunks(doc.id)
        assert deleted == 2
        assert store._conn().execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc.id,)
        ).fetchone()[0] == 0

    def test_delete_also_removes_vectors(self, store: DocumentStore) -> None:
        doc, chunk = _insert_doc_and_chunk(
            store, embedding=[0.1, 0.2, 0.3, 0.4]
        )
        store.delete_document_chunks(doc.id)
        count = store._conn().execute(
            "SELECT COUNT(*) FROM vectors WHERE chunk_id = ?", (chunk.id,)
        ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


class TestVectors:
    def test_insert_vector_roundtrip(self, store: DocumentStore) -> None:
        doc, chunk = _insert_doc_and_chunk(store)
        store.insert_vector(chunk.id, [1.0, 0.0, 0.0, 0.0])

        count = store._conn().execute(
            "SELECT COUNT(*) FROM vectors WHERE chunk_id = ?", (chunk.id,)
        ).fetchone()[0]
        assert count == 1

    def test_insert_vector_upsert(self, store: DocumentStore) -> None:
        doc, chunk = _insert_doc_and_chunk(store)
        store.insert_vector(chunk.id, [1.0, 0.0, 0.0, 0.0])
        store.insert_vector(chunk.id, [0.0, 1.0, 0.0, 0.0])  # overwrite

        count = store._conn().execute(
            "SELECT COUNT(*) FROM vectors WHERE chunk_id = ?", (chunk.id,)
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


class TestSearchFTS:
    def _seed(self, store: DocumentStore) -> None:
        doc = store.upsert_document("/notes.md", "h")
        store.insert_chunk(doc.id, 0, "the quick brown fox", 0, 19)
        store.insert_chunk(doc.id, 1, "lazy dog sleeps", 19, 34)
        store.insert_chunk(doc.id, 2, "fox jumps over the dog", 34, 56)

    def test_returns_matching_chunks(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_fts("fox", top_k=10)
        texts = [h.text for h in hits]
        assert any("fox" in t for t in texts)

    def test_returns_at_most_top_k(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_fts("the", top_k=1)
        assert len(hits) <= 1

    def test_scores_in_zero_one(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_fts("fox", top_k=10)
        for h in hits:
            assert 0.0 <= h.score <= 1.0

    def test_top_hit_has_highest_score(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_fts("fox dog", top_k=10)
        if len(hits) > 1:
            assert hits[0].score >= hits[1].score

    def test_no_match_returns_empty(self, store: DocumentStore) -> None:
        self._seed(store)
        assert store.search_fts("zzzyyyxxx", top_k=10) == []

    def test_hit_includes_doc_path(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_fts("fox", top_k=10)
        assert all(h.doc_path == "/notes.md" for h in hits)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestSearchVectors:
    def _seed(self, store: DocumentStore) -> list[tuple[ChunkRecord, list[float]]]:
        doc = store.upsert_document("/vecs.md", "h")
        pairs = [
            ("north pole", [1.0, 0.0, 0.0, 0.0]),
            ("south pole", [0.0, 1.0, 0.0, 0.0]),
            ("equator",   [0.0, 0.0, 1.0, 0.0]),
            ("near north", [0.9, 0.1, 0.0, 0.0]),
        ]
        records = []
        for i, (text, vec) in enumerate(pairs):
            chunk = store.insert_chunk(doc.id, i, text, i * 10, i * 10 + 10)
            store.insert_vector(chunk.id, vec)
            records.append((chunk, vec))
        return records

    def test_nearest_neighbour_correct(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(hits) == 1
        assert hits[0].text == "north pole"

    def test_second_nearest(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], top_k=2)
        texts = [h.text for h in hits]
        assert "near north" in texts

    def test_top_k_respected(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_vectors([0.5, 0.5, 0.0, 0.0], top_k=2)
        assert len(hits) <= 2

    def test_scores_in_zero_one(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], top_k=4)
        for h in hits:
            assert 0.0 <= h.score <= 1.0

    def test_closer_vector_has_higher_score(self, store: DocumentStore) -> None:
        self._seed(store)
        hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], top_k=4)
        # "north pole" should score higher than "south pole"
        scores = {h.text: h.score for h in hits}
        assert scores["north pole"] > scores["south pole"]

    def test_empty_table_returns_empty(self, store: DocumentStore) -> None:
        hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert hits == []


# ---------------------------------------------------------------------------
# Stats + metadata
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_store_stats(self, store: DocumentStore) -> None:
        s = store.stats()
        assert s.doc_count == 0
        assert s.chunk_count == 0
        assert s.last_indexed is None

    def test_stats_after_insert(self, store: DocumentStore) -> None:
        _insert_doc_and_chunk(store, path="/a.md")
        _insert_doc_and_chunk(store, path="/b.md")
        s = store.stats()
        assert s.doc_count == 2
        assert s.chunk_count == 2

    def test_set_and_get_last_indexed(self, store: DocumentStore) -> None:
        ts = 1_700_000_000.0
        store.set_last_indexed(ts)
        s = store.stats()
        assert s.last_indexed == pytest.approx(ts)

    def test_set_last_indexed_overwrites(self, store: DocumentStore) -> None:
        store.set_last_indexed(1000.0)
        store.set_last_indexed(2000.0)
        assert store.stats().last_indexed == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        s = DocumentStore(cfg, embedding_dim=EMBEDDING_DIM)
        s.init_schema()
        s.close()
        s.close()  # must not raise

    def test_reopens_after_close(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        s = DocumentStore(cfg, embedding_dim=EMBEDDING_DIM)
        s.init_schema()
        s.upsert_document("/x.md", "h")
        s.close()

        s2 = DocumentStore(cfg, embedding_dim=EMBEDDING_DIM)
        s2.init_schema()
        doc = s2.get_document_by_path("/x.md")
        assert doc is not None
        s2.close()
