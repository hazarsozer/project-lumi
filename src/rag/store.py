"""
SQLite-backed document store for Project Lumi's RAG retriever.

Each public method is safe to call from a single thread — the store uses
threading.local() to give each thread its own SQLite connection, so callers
must not share a DocumentStore across threads without understanding that each
thread opens its own connection to the same database file.

WAL mode is enabled at connection time, which means readers (retrieval in the
inference thread) and writers (the ingest CLI in a separate process) do not
block each other.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from src.core.config import RAGConfig
from src.rag.errors import RAGUnavailableError

logger = logging.getLogger(__name__)

# Matches all-MiniLM-L6-v2 output dimension; overridable via __init__.
_DEFAULT_EMBEDDING_DIM: int = 384


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    path: str
    sha256: str
    ingested_at: float


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    document_id: int
    chunk_idx: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    score: float  # higher = more relevant (normalised to [0, 1] range)
    text: str
    doc_path: str
    chunk_idx: int


@dataclass(frozen=True)
class StoreStats:
    doc_count: int
    chunk_count: int
    last_indexed: float | None  # unix timestamp or None if never ingested


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------


class DocumentStore:
    """Thin repository layer over the RAG SQLite database.

    Args:
        config: RAGConfig instance; ``db_path`` is expanded with
                ``Path.expanduser()`` so tildes work correctly.
        embedding_dim: Dimension of the embedding vectors stored in the
                       ``vectors`` vec0 table.  Must match the model used at
                       ingest time.  Defaults to 384 (all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        config: RAGConfig,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._db_path = Path(config.db_path).expanduser()
        self._embedding_dim = embedding_dim
        self._local: threading.local = threading.local()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return this thread's SQLite connection, opening it if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open_connection()
            self._local.conn = conn
        return conn

    def _open_connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            conn.close()
            raise RAGUnavailableError(
                f"sqlite-vec extension failed to load: {exc}"
            ) from exc

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create all tables if they do not already exist.

        Safe to call multiple times — all statements use ``IF NOT EXISTS``.
        """
        conn = self._conn()
        schema_path = Path(__file__).parent / "schema.sql"
        conn.executescript(schema_path.read_text(encoding="utf-8"))

        # vec0 virtual table dimensions are runtime config, so it cannot live
        # in the static schema.sql file.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, "
            f"embedding FLOAT[{self._embedding_dim}])"
        )
        conn.commit()
        logger.debug("RAG schema initialised at %s", self._db_path)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def upsert_document(self, path: str, sha256: str) -> DocumentRecord:
        """Insert or update a document record.  Returns the final row."""
        conn = self._conn()
        now = time.time()
        conn.execute(
            """
            INSERT INTO documents(path, sha256, ingested_at) VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE
                SET sha256=excluded.sha256, ingested_at=excluded.ingested_at
            """,
            (path, sha256, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, path, sha256, ingested_at FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        return DocumentRecord(
            id=row["id"],
            path=row["path"],
            sha256=row["sha256"],
            ingested_at=row["ingested_at"],
        )

    def get_document_by_path(self, path: str) -> DocumentRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT id, path, sha256, ingested_at FROM documents WHERE path = ?",
                (path,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            path=row["path"],
            sha256=row["sha256"],
            ingested_at=row["ingested_at"],
        )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def insert_chunk(
        self,
        document_id: int,
        chunk_idx: int,
        text: str,
        char_start: int,
        char_end: int,
    ) -> ChunkRecord:
        """Append a text chunk for the given document.  Returns the new row."""
        conn = self._conn()
        cur = conn.execute(
            """
            INSERT INTO chunks(document_id, chunk_idx, text, char_start, char_end)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, chunk_idx, text, char_start, char_end),
        )
        conn.commit()
        chunk_id = cur.lastrowid
        return ChunkRecord(
            id=chunk_id,
            document_id=document_id,
            chunk_idx=chunk_idx,
            text=text,
            char_start=char_start,
            char_end=char_end,
        )

    def get_chunk_by_id(self, chunk_id: int) -> ChunkRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT id, document_id, chunk_idx, text, char_start, char_end "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return ChunkRecord(
            id=row["id"],
            document_id=row["document_id"],
            chunk_idx=row["chunk_idx"],
            text=row["text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
        )

    def delete_document_chunks(self, document_id: int) -> int:
        """Delete all chunks (and their vectors) for a document.

        Returns the number of chunks deleted.
        """
        conn = self._conn()
        chunk_ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
        ]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(
                f"DELETE FROM vectors WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.commit()
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Vectors
    # ------------------------------------------------------------------

    def insert_vector(self, chunk_id: int, embedding: list[float]) -> None:
        """Store the embedding vector for a chunk, replacing any existing row."""
        conn = self._conn()
        # vec0 virtual tables do not support INSERT OR REPLACE; delete first.
        conn.execute("DELETE FROM vectors WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO vectors(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(embedding)),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip characters that FTS5 interprets as syntax operators.

        FTS5 treats '.', '?', '"', '(', ')', '-', '*', '^' as special.
        Replace any non-word, non-space character with a space and collapse
        runs so the result is a plain bag-of-words query.
        """
        sanitized = re.sub(r"[^\w\s]", " ", query)
        return re.sub(r"\s+", " ", sanitized).strip()

    def search_fts(self, query: str, top_k: int) -> list[SearchHit]:
        """BM25 keyword search via SQLite FTS5.

        SQLite's bm25() returns negative values (more negative = more
        relevant).  We negate to produce a positive score where higher is
        better, then normalise to [0, 1] across the result set.
        """
        clean_query = self._sanitize_fts_query(query)
        if not clean_query:
            return []
        rows = (
            self._conn()
            .execute(
                """
            SELECT c.id       AS chunk_id,
                   c.text,
                   c.chunk_idx,
                   d.path,
                   -bm25(fts) AS raw_score
            FROM fts
            JOIN chunks    c ON c.id  = fts.rowid
            JOIN documents d ON d.id  = c.document_id
            WHERE fts MATCH ?
            ORDER BY raw_score DESC
            LIMIT ?
            """,
                (clean_query, top_k),
            )
            .fetchall()
        )

        if not rows:
            return []

        max_score = max(r["raw_score"] for r in rows) or 1.0
        return [
            SearchHit(
                chunk_id=r["chunk_id"],
                score=r["raw_score"] / max_score,
                text=r["text"],
                doc_path=r["path"],
                chunk_idx=r["chunk_idx"],
            )
            for r in rows
        ]

    def search_vectors(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        """k-nearest-neighbour search via sqlite-vec.

        Distance is L2; we convert to a similarity score in [0, 1] via
        1 / (1 + distance) so that higher score = more relevant, consistent
        with search_fts().
        """
        serialised = sqlite_vec.serialize_float32(embedding)
        rows = (
            self._conn()
            .execute(
                """
            SELECT v.chunk_id,
                   v.distance,
                   c.text,
                   c.chunk_idx,
                   d.path
            FROM vectors v
            JOIN chunks    c ON c.id  = v.chunk_id
            JOIN documents d ON d.id  = c.document_id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
                (serialised, top_k),
            )
            .fetchall()
        )

        return [
            SearchHit(
                chunk_id=r["chunk_id"],
                score=1.0 / (1.0 + r["distance"]),
                text=r["text"],
                doc_path=r["path"],
                chunk_idx=r["chunk_idx"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def stats(self) -> StoreStats:
        conn = self._conn()
        doc_count: int = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count: int = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_indexed'"
        ).fetchone()
        last_indexed: float | None = float(row[0]) if row else None
        return StoreStats(
            doc_count=doc_count,
            chunk_count=chunk_count,
            last_indexed=last_indexed,
        )

    def set_last_indexed(self, timestamp: float) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_indexed', ?)",
            (str(timestamp),),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close this thread's connection.  No-op if already closed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
