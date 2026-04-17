"""
Document ingest CLI for the Lumi RAG personal knowledge base.

Walks a folder, hashes each supported file (SHA-256), skips files whose
hash hasn't changed since the last ingest, and writes chunks + embeddings
to the RAG SQLite database.

Usage:
    uv run python scripts/ingest_docs.py
    uv run python scripts/ingest_docs.py --corpus ~/notes --db ~/.lumi/rag.db
    uv run python scripts/ingest_docs.py --force   # re-ingest all files

Exit code:
    0 — completed (even if some files failed; errors are logged individually)
    1 — fatal setup error (config invalid, DB unreachable, embedder failed)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config import RAGConfig, load_config
from src.rag.embedder import get_embedder
from src.rag.errors import IngestError, RAGUnavailableError
from src.rag.loader import is_supported, load
from src.rag.chunker import chunk_text
from src.rag.store import DocumentStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _ingest_file(
    path: Path,
    store: DocumentStore,
    embedder,
    config: RAGConfig,
    force: bool,
) -> dict:
    """Ingest one file.  Returns a stats dict."""
    sha256 = _sha256(path)
    existing = store.get_document_by_path(str(path))

    if not force and existing and existing.sha256 == sha256:
        return {"status": "skipped", "chunks": 0}

    try:
        doc_loaded = load(path)
    except IngestError as exc:
        logger.error("Load failed for %s: %s", path, exc)
        return {"status": "error", "chunks": 0}

    chunks = chunk_text(
        doc_loaded.text,
        size=config.chunk_size,
        overlap=config.chunk_overlap,
    )

    if not chunks:
        logger.warning("No chunks produced from %s (file may be empty).", path)
        return {"status": "empty", "chunks": 0}

    texts = [c.text for c in chunks]
    try:
        embeddings = embedder.encode(texts)
    except Exception as exc:
        logger.error("Embedding failed for %s: %s", path, exc)
        return {"status": "error", "chunks": 0}

    # Atomic replacement: delete old chunks first, then re-insert.
    doc_record = store.upsert_document(str(path), sha256)
    store.delete_document_chunks(doc_record.id)

    for chunk, embedding in zip(chunks, embeddings):
        chunk_record = store.insert_chunk(
            document_id=doc_record.id,
            chunk_idx=chunk.chunk_idx,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        store.insert_vector(chunk_record.id, embedding)

    return {"status": "ok", "chunks": len(chunks)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the Lumi RAG store.")
    parser.add_argument("--corpus", help="Directory to ingest (overrides config.rag.corpus_dir)")
    parser.add_argument("--db", help="Path to the RAG SQLite database (overrides config.rag.db_path)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-ingest all files even if unchanged")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    rag = cfg.rag
    corpus_dir = Path(args.corpus or rag.corpus_dir).expanduser()
    db_path_override = args.db

    if db_path_override:
        import dataclasses
        rag = dataclasses.replace(rag, db_path=db_path_override)

    if not corpus_dir.is_dir():
        logger.error("Corpus directory does not exist: %s", corpus_dir)
        sys.exit(1)

    try:
        store = DocumentStore(rag, embedding_dim=get_embedder(rag.embedding_model).embedding_dim)
        store.init_schema()
    except RAGUnavailableError as exc:
        logger.error("RAG store unavailable: %s", exc)
        sys.exit(1)

    embedder = get_embedder(rag.embedding_model)

    files = [p for p in corpus_dir.rglob("*") if p.is_file() and is_supported(p)]
    if not files:
        logger.warning("No supported files found in %s", corpus_dir)
        sys.exit(0)

    logger.info("Found %d file(s) in %s", len(files), corpus_dir)

    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "empty": 0, "error": 0}
    total_chunks = 0

    for path in sorted(files):
        result = _ingest_file(path, store, embedder, rag, force=args.force)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
        total_chunks += result["chunks"]
        logger.info("%-8s  %s  (%d chunks)", status.upper(), path.name, result["chunks"])

    store.set_last_indexed(time.time())
    elapsed = time.time() - t0

    logger.info(
        "\nDone in %.1fs — %d ingested, %d skipped, %d empty, %d errors. "
        "Total chunks: %d",
        elapsed, counts["ok"], counts["skipped"], counts["empty"], counts.get("error", 0),
        total_chunks,
    )
    store.close()


if __name__ == "__main__":
    main()
