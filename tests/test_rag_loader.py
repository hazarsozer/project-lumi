"""Tests for the document loader — .md, .txt, and .pdf dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.errors import IngestError
from src.rag.loader import LoadedDocument, is_supported, load


class TestIsSupported:
    def test_md_supported(self, tmp_path: Path):
        assert is_supported(tmp_path / "doc.md") is True

    def test_txt_supported(self, tmp_path: Path):
        assert is_supported(tmp_path / "notes.txt") is True

    def test_pdf_supported(self, tmp_path: Path):
        assert is_supported(tmp_path / "manual.pdf") is True

    def test_py_not_supported(self, tmp_path: Path):
        assert is_supported(tmp_path / "script.py") is False

    def test_docx_not_supported(self, tmp_path: Path):
        assert is_supported(tmp_path / "report.docx") is False

    def test_case_insensitive(self, tmp_path: Path):
        assert is_supported(tmp_path / "README.MD") is True


class TestLoadText:
    def test_loads_md_file(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("# Hello\n\nWorld.", encoding="utf-8")
        doc = load(f)
        assert isinstance(doc, LoadedDocument)
        assert "Hello" in doc.text
        assert doc.extension == ".md"

    def test_loads_txt_file(self, tmp_path: Path):
        f = tmp_path / "plain.txt"
        f.write_text("Plain text content.", encoding="utf-8")
        doc = load(f)
        assert doc.text == "Plain text content."
        assert doc.extension == ".txt"

    def test_char_count_matches(self, tmp_path: Path):
        content = "Some text here."
        f = tmp_path / "a.txt"
        f.write_text(content, encoding="utf-8")
        doc = load(f)
        assert doc.char_count == len(doc.text)

    def test_path_stored_as_string(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("x", encoding="utf-8")
        doc = load(f)
        assert isinstance(doc.path, str)

    def test_normalises_excess_blank_lines(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("Para one.\n\n\n\n\nPara two.", encoding="utf-8")
        doc = load(f)
        assert "\n\n\n" not in doc.text

    def test_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "file.csv"
        f.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(IngestError, match="Unsupported"):
            load(f)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(IngestError, match="not found"):
            load(tmp_path / "nonexistent.txt")


class TestLoadPDF:
    def test_pdf_extraction(self, tmp_path: Path):
        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # A blank page yields no text — just verify no crash.
        doc = load(pdf_path)
        assert doc.extension == ".pdf"
        assert isinstance(doc.text, str)
