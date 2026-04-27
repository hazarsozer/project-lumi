"""
RAG ingest voice tool for Project Lumi.

Wraps the document-ingest pipeline as a ``Tool`` so the user can say
"index my notes folder" and Lumi will call this tool to ingest documents
into the RAG vector store.

Tool schema::

    {"tool": "rag_ingest", "args": {"path": "<folder or file path>"}}

Returns on success::

    ToolResult(
        success=True,
        output="Ingested 5 documents (20 chunks).",
        data={"status": "ok", "docs_indexed": 5, "chunks_total": 20, "errors": 0},
    )

Returns on error::

    ToolResult(
        success=False,
        output="Path does not exist: /bad/path",
        data={"status": "error", "message": "..."},
    )

Security decisions:
- Paths containing ".." components are rejected before any filesystem access.
- The path must exist (file or directory) before ingest begins.
- Only string path arguments are accepted.

Threading:
- ``execute()`` is synchronous and safe to call from any thread.
- The ToolExecutor already wraps tool calls in a background thread, so
  ``execute()`` does not need to spawn its own thread.
- An optional ``event_callback`` receives a ``ToolResultEvent`` after
  completion, letting the orchestrator or UI react to the result.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.core.config import RAGConfig
from src.core.events import ToolResultEvent
from src.rag.chunker import chunk_text
from src.rag.embedder import get_embedder
from src.rag.loader import is_supported, load
from src.rag.store import DocumentStore
from src.tools.base import ToolResult

logger = logging.getLogger(__name__)

# Default RAGConfig used when no config is supplied at construction time.
# This matches the project's default paths — callers that need custom DB paths
# should pass a RAGConfig explicitly.
_DEFAULT_RAG_CONFIG = RAGConfig()

# Large-ingest confirmation thresholds.
_LARGE_FILE_BYTES: int = 10 * 1024 * 1024  # 10 MB
_LARGE_DIR_FILE_COUNT: int = 100


class RagIngestTool:
    """Ingest documents from a local folder or file into the RAG vector store.

    Args:
        rag_config:      RAGConfig to use for the DocumentStore and embedder.
                         Defaults to the project's default RAGConfig.
        event_callback:  Optional callable receiving a ``ToolResultEvent``
                         after execution completes (success or failure).
                         Useful for notifying the orchestrator or UI without
                         polling.
    """

    name: str = "rag_ingest"
    description: str = (
        "Index a local folder or file into the Lumi knowledge base so it can "
        "be retrieved in future conversations. "
        'Args: {"path": "<absolute or relative folder/file path>"}. '
        "Returns: docs indexed count on success."
    )

    def __init__(
        self,
        rag_config: RAGConfig | None = None,
        event_callback: Callable[[ToolResultEvent], None] | None = None,
    ) -> None:
        self._rag_config = rag_config or _DEFAULT_RAG_CONFIG
        self._event_callback = event_callback

    # ------------------------------------------------------------------
    # Public API — Tool Protocol
    # ------------------------------------------------------------------

    def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the rag_ingest tool.

        Validates the ``path`` argument, then delegates to ``_run_ingest``.
        Posts a ``ToolResultEvent`` via ``event_callback`` (if set) before
        returning.

        Args:
            args: Dict with key ``"path"`` (str) pointing to a folder or file.

        Returns:
            ToolResult — never raises.
        """
        result = self._execute_inner(args)
        self._maybe_post_event(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_inner(self, args: dict[str, Any]) -> ToolResult:
        """Validate args and run ingest; return ToolResult."""
        raw_path: Any = args.get("path", "")

        # --- Type validation ---
        if not isinstance(raw_path, str) or not raw_path:
            logger.warning("RagIngestTool: missing or non-string 'path' arg.")
            return ToolResult(
                success=False,
                output="Missing required arg: path (must be a non-empty string)",
                data={"status": "error", "message": "invalid path argument"},
            )

        # --- Security: reject path traversal ---
        path = Path(raw_path)
        if ".." in path.parts:
            logger.warning("RagIngestTool: path traversal rejected for '%s'.", raw_path)
            return ToolResult(
                success=False,
                output=f"Invalid path (traversal not allowed): {raw_path}",
                data={"status": "error", "message": "path traversal rejected"},
            )

        # --- Existence check ---
        if not path.exists():
            logger.warning("RagIngestTool: path does not exist: '%s'.", raw_path)
            return ToolResult(
                success=False,
                output=f"Path does not exist: {raw_path}",
                data={"status": "error", "message": f"path not found: {raw_path}"},
            )

        # --- Large-ingest confirmation gate ---
        confirmed: bool = bool(args.get("confirmed", False))
        if not confirmed:
            guard = self._check_confirmation_needed(path)
            if guard is not None:
                logger.info(
                    "RagIngestTool: confirmation required for '%s': %s",
                    raw_path,
                    guard["reason"],
                )
                return ToolResult(
                    success=False,
                    output=guard["reason"],
                    data={
                        "status": "needs_confirmation",
                        "needs_confirmation": True,
                        "reason": guard["reason"],
                        "estimated_chunks": guard["estimated_chunks"],
                    },
                )

        # --- Run ingest ---
        try:
            stats = self._run_ingest(path)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RagIngestTool: ingest failed with %s: %s.", type(exc).__name__, exc
            )
            return ToolResult(
                success=False,
                output=str(exc),
                data={"status": "error", "message": str(exc)},
            )

        docs = stats["docs_indexed"]
        chunks = stats["chunks_total"]
        errors = stats["errors"]
        output = f"Ingested {docs} document(s) ({chunks} chunks, {errors} errors)."
        logger.info("RagIngestTool: %s", output)

        return ToolResult(
            success=True,
            output=output,
            data={
                "status": "ok",
                "docs_indexed": docs,
                "chunks_total": chunks,
                "errors": errors,
            },
        )

    def _maybe_post_event(self, result: ToolResult) -> None:
        """Post a ToolResultEvent via the callback if one was provided."""
        if self._event_callback is None:
            return
        event = ToolResultEvent(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            data=dict(result.data),
        )
        try:
            self._event_callback(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RagIngestTool: event_callback raised %s: %s.", type(exc).__name__, exc
            )

    def _check_confirmation_needed(self, path: Path) -> dict[str, Any] | None:
        """Return a confirmation payload when the ingest target is large.

        Returns None if no confirmation is needed; otherwise a dict with
        ``reason`` and ``estimated_chunks`` keys.
        """
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                return None
            if size > _LARGE_FILE_BYTES:
                size_mb = size / (1024 * 1024)
                estimated = max(1, size // (_LARGE_FILE_BYTES // 20))
                return {
                    "reason": (
                        f"File is large ({size_mb:.1f} MB > 10 MB). "
                        "Pass confirmed=True to proceed."
                    ),
                    "estimated_chunks": estimated,
                }
        else:
            try:
                file_count = sum(1 for p in path.rglob("*") if p.is_file())
            except OSError:
                return None
            if file_count > _LARGE_DIR_FILE_COUNT:
                return {
                    "reason": (
                        f"Directory contains {file_count} files (> 100). "
                        "Pass confirmed=True to proceed."
                    ),
                    "estimated_chunks": file_count * 2,
                }
        return None

    def _run_ingest(self, path: Path) -> dict[str, int]:
        """Walk *path* and ingest all supported files into the RAG store.

        Args:
            path: A ``Path`` that exists (file or directory).

        Returns:
            Dict with keys ``docs_indexed``, ``chunks_total``, ``errors``.
        """
        rag = self._rag_config
        embedder = get_embedder(rag.embedding_model)
        store = DocumentStore(rag, embedding_dim=embedder.embedding_dim)
        store.init_schema()

        # Collect files
        if path.is_file():
            files = [path] if is_supported(path) else []
        else:
            files = [p for p in path.rglob("*") if p.is_file() and is_supported(p)]

        docs_indexed = 0
        chunks_total = 0
        errors = 0

        for file_path in sorted(files):
            result = self._ingest_one(file_path, store, embedder, rag)
            status = result["status"]
            if status == "ok":
                docs_indexed += 1
                chunks_total += result["chunks"]
            elif status == "error":
                errors += 1
            # "skipped" and "empty" are not counted as indexed or errored

        store.set_last_indexed(time.time())
        store.close()

        return {
            "docs_indexed": docs_indexed,
            "chunks_total": chunks_total,
            "errors": errors,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def _ingest_one(
        self,
        path: Path,
        store: DocumentStore,
        embedder: Any,
        config: RAGConfig,
    ) -> dict[str, Any]:
        """Ingest a single file.  Returns a status dict."""
        from src.rag.errors import IngestError

        sha256 = self._sha256(path)
        existing = store.get_document_by_path(str(path))

        if existing and existing.sha256 == sha256:
            return {"status": "skipped", "chunks": 0}

        try:
            doc_loaded = load(path)
        except IngestError as exc:
            logger.error("RagIngestTool: load failed for %s: %s", path, exc)
            return {"status": "error", "chunks": 0}

        chunks = chunk_text(
            doc_loaded.text,
            size=config.chunk_size,
            overlap=config.chunk_overlap,
        )

        if not chunks:
            logger.warning("RagIngestTool: no chunks from %s (empty file?).", path)
            return {"status": "empty", "chunks": 0}

        texts = [c.text for c in chunks]
        try:
            embeddings = embedder.encode(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error("RagIngestTool: embedding failed for %s: %s", path, exc)
            return {"status": "error", "chunks": 0}

        doc_record = store.upsert_document(str(path), sha256)
        store.delete_document_chunks(doc_record.id)

        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk_record = store.insert_chunk(
                document_id=doc_record.id,
                chunk_idx=chunk.chunk_idx,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            store.insert_vector(chunk_record.id, embedding)

        return {"status": "ok", "chunks": len(chunks)}
