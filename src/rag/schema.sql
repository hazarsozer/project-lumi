-- Project Lumi — RAG personal knowledge-base schema
-- SQLite 3.35+ required for RETURNING; sqlite-vec extension required for vec0.
--
-- WAL mode is set at connection open time by DocumentStore.init_schema(),
-- not here, because PRAGMA cannot appear inside a CREATE statement.
--
-- Run order: execute this file once against a fresh database, or call
-- DocumentStore.init_schema() which executes each statement idempotently.

-- ─────────────────────────────────────────────────────────────────────────────
-- documents — one row per ingested source file
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT    NOT NULL UNIQUE,   -- absolute path on disk
    sha256      TEXT    NOT NULL,          -- hex digest of file contents at ingest
    ingested_at REAL    NOT NULL           -- unix timestamp (time.time())
);

-- ─────────────────────────────────────────────────────────────────────────────
-- chunks — fixed-size text windows produced by the chunker
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_idx   INTEGER NOT NULL,          -- 0-based position within the document
    text        TEXT    NOT NULL,
    char_start  INTEGER NOT NULL,          -- byte offset of first char in original
    char_end    INTEGER NOT NULL           -- byte offset of last char (exclusive)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- vectors — one embedding vector per chunk (sqlite-vec vec0 virtual table)
-- ─────────────────────────────────────────────────────────────────────────────
-- Dimensions must match the embedding model (all-MiniLM-L6-v2 → 384).
-- This table is created at runtime by DocumentStore.init_schema() via a
-- parameterised CREATE VIRTUAL TABLE statement so the dimension can come
-- from RAGConfig.  The stub below documents the intended shape.
--
-- CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
--     chunk_id  INTEGER PRIMARY KEY,
--     embedding FLOAT[384]
-- );

-- ─────────────────────────────────────────────────────────────────────────────
-- fts — full-text search index for BM25 keyword retrieval
-- ─────────────────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers keep the FTS index in sync with the chunks table.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts(fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts(fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO fts(rowid, text) VALUES (new.id, new.text);
END;

-- ─────────────────────────────────────────────────────────────────────────────
-- meta — key/value store for global state (last_indexed, schema_version, etc.)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');
