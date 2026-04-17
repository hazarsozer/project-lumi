"""Integration tests for scripts/ingest_docs.py — idempotency and stats."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


def _run_ingest(corpus: Path, db: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "scripts/ingest_docs.py",
        "--corpus", str(corpus),
        "--db", str(db),
    ] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True)


def _write_doc(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


class TestIngestScript:
    def test_exits_zero_on_success(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "a.txt", "Hello world. This is a document.")
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        assert result.returncode == 0, result.stderr

    def test_ingests_md_and_txt(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "notes.md", "# Title\n\nSome content here.")
        _write_doc(corpus, "plain.txt", "Plain text document.")
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        assert result.returncode == 0
        assert "OK" in result.stderr or "ok" in result.stderr.lower()

    def test_idempotent_second_run_skips(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "note.txt", "Some content that should be chunked.")
        db = tmp_path / "rag.db"

        _run_ingest(corpus, db)
        result2 = _run_ingest(corpus, db)
        assert result2.returncode == 0
        assert "SKIPPED" in result2.stderr or "skipped" in result2.stderr.lower()

    def test_force_flag_reingest(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "note.txt", "Content for forced re-ingest.")
        db = tmp_path / "rag.db"

        _run_ingest(corpus, db)
        result = _run_ingest(corpus, db, extra_args=["--force"])
        assert result.returncode == 0
        # With --force, skipped count should be 0 — file should be re-ingested.
        assert "SKIPPED" not in result.stderr.upper() or "0 skipped" in result.stderr.lower()

    def test_changed_file_reingested(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        doc = _write_doc(corpus, "note.txt", "Original content.")
        db = tmp_path / "rag.db"
        _run_ingest(corpus, db)

        doc.write_text("Modified content — different hash now.", encoding="utf-8")
        result = _run_ingest(corpus, db)
        assert result.returncode == 0
        assert "OK" in result.stderr or "ok" in result.stderr.lower()

    def test_empty_corpus_exits_zero(self, tmp_path: Path):
        corpus = tmp_path / "empty"
        corpus.mkdir()
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        assert result.returncode == 0

    def test_missing_corpus_exits_nonzero(self, tmp_path: Path):
        db = tmp_path / "rag.db"
        result = _run_ingest(tmp_path / "nonexistent", db)
        assert result.returncode != 0

    def test_unsupported_files_ignored(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "script.py").write_text("print('hi')", encoding="utf-8")
        (corpus / "data.csv").write_text("a,b,c", encoding="utf-8")
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        # No supported files — should exit zero with a warning.
        assert result.returncode == 0
