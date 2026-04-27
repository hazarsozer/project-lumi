"""
Document loader for the Lumi RAG ingest pipeline.

Dispatches to the appropriate reader based on file extension and returns
normalised plain text + lightweight metadata.  All I/O is synchronous;
call from the ingest CLI process, not from the inference thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.rag.errors import IngestError

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".pdf"})


@dataclass(frozen=True)
class LoadedDocument:
    """Normalised content of one source file."""

    path: str  # absolute path on disk
    text: str  # plain UTF-8 text extracted from the file
    char_count: int  # len(text)
    extension: str  # lower-case file extension including the dot


def is_supported(path: Path) -> bool:
    """Return True if *path* has a supported extension."""
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def load(path: Path) -> LoadedDocument:
    """Read *path* and return a :class:`LoadedDocument`.

    Args:
        path: Absolute or relative path to the document.

    Returns:
        A :class:`LoadedDocument` with normalised text.

    Raises:
        :class:`~src.rag.errors.IngestError`: if the file cannot be read or
            has an unsupported extension.
    """
    path = path.resolve()
    ext = path.suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"Unsupported file type '{ext}' for '{path}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    if not path.exists():
        raise IngestError(f"File not found: {path}")

    try:
        if ext == ".pdf":
            text = _load_pdf(path)
        else:
            text = _load_text(path)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Failed to read '{path}': {exc}") from exc

    text = _normalise(text)
    return LoadedDocument(
        path=str(path),
        text=text,
        char_count=len(text),
        extension=ext,
    )


# ---------------------------------------------------------------------------
# Private readers
# ---------------------------------------------------------------------------


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise IngestError(
            "pypdf is required to load PDF files. " "Install it with: uv add pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    if not pages:
        logger.warning("PDF '%s' yielded no extractable text.", path)
    return "\n\n".join(pages)


def _normalise(text: str) -> str:
    """Collapse excessive blank lines and strip leading/trailing whitespace."""
    import re

    # Collapse 3+ consecutive newlines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
