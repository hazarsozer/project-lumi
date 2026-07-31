"""Integration tests for scripts/ingest_docs.py — idempotency and stats."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

# Root of the repository — resolved once so subprocess.run can pin cwd= to it
# regardless of any os.chdir() calls made by other tests in the full suite.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Each test in TestIngestScript spawns at least one subprocess that loads the
# SentenceTransformer model (~7–20 s per call on a warm disk cache; longer
# under memory pressure after other tests have run).  Two-subprocess tests
# (idempotent, force, changed-file) need up to ~60 s total.  We use 120 s as
# a conservative per-test timeout so the 30 s global backstop does not fire.
_INGEST_TIMEOUT = 120


def _run_ingest(
    corpus: Path, db: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "scripts/ingest_docs.py",
        "--corpus",
        str(corpus),
        "--db",
        str(db),
    ] + (extra_args or [])
    # cwd= is pinned to the project root so the subprocess always finds
    # scripts/ingest_docs.py and config.yaml via relative paths, independent
    # of any os.chdir() calls made by other tests in the same pytest session.
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))


def _write_doc(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


class TestIngestScript:
    @pytest.mark.timeout(_INGEST_TIMEOUT)
    def test_exits_zero_on_success(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "a.txt", "Hello world. This is a document.")
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        assert result.returncode == 0, result.stderr

    @pytest.mark.timeout(_INGEST_TIMEOUT)
    def test_ingests_md_and_txt(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "notes.md", "# Title\n\nSome content here.")
        _write_doc(corpus, "plain.txt", "Plain text document.")
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        assert result.returncode == 0
        assert "OK" in result.stderr or "ok" in result.stderr.lower()

    @pytest.mark.timeout(_INGEST_TIMEOUT)
    def test_idempotent_second_run_skips(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "note.txt", "Some content that should be chunked.")
        db = tmp_path / "rag.db"

        _run_ingest(corpus, db)
        result2 = _run_ingest(corpus, db)
        assert result2.returncode == 0
        assert "SKIPPED" in result2.stderr or "skipped" in result2.stderr.lower()

    @pytest.mark.timeout(_INGEST_TIMEOUT)
    def test_force_flag_reingest(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        _write_doc(corpus, "note.txt", "Content for forced re-ingest.")
        db = tmp_path / "rag.db"

        _run_ingest(corpus, db)
        result = _run_ingest(corpus, db, extra_args=["--force"])
        assert result.returncode == 0
        # With --force, skipped count should be 0 — file should be re-ingested.
        assert (
            "SKIPPED" not in result.stderr.upper()
            or "0 skipped" in result.stderr.lower()
        )

    @pytest.mark.timeout(_INGEST_TIMEOUT)
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

    @pytest.mark.timeout(_INGEST_TIMEOUT)
    def test_empty_corpus_exits_zero(self, tmp_path: Path):
        corpus = tmp_path / "empty"
        corpus.mkdir()
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        assert result.returncode == 0

    def test_missing_corpus_exits_nonzero(self, tmp_path: Path):
        # No model load — exits before reaching get_embedder(); default 30 s is fine.
        db = tmp_path / "rag.db"
        result = _run_ingest(tmp_path / "nonexistent", db)
        assert result.returncode != 0

    @pytest.mark.timeout(_INGEST_TIMEOUT)
    def test_unsupported_files_ignored(self, tmp_path: Path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "script.py").write_text("print('hi')", encoding="utf-8")
        (corpus / "data.csv").write_text("a,b,c", encoding="utf-8")
        db = tmp_path / "rag.db"
        result = _run_ingest(corpus, db)
        # No supported files — should exit zero with a warning.
        assert result.returncode == 0
