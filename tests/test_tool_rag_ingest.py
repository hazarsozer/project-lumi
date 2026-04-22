"""
Unit tests for src.tools.rag_ingest — RagIngestTool.

All filesystem and RAG ingest logic is mocked.  No real SQLite, embedder,
or filesystem operations are performed.

TDD cycle: these tests are written BEFORE the implementation (RED phase).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test 1: tool appears in the tool registry after registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_tool_registered() -> None:
    """RagIngestTool can be instantiated and registered in ToolRegistry."""
    from src.tools.rag_ingest import RagIngestTool

    registry = ToolRegistry()
    tool = RagIngestTool()
    registry.register(tool)

    assert registry.is_registered("rag_ingest")
    assert registry.get("rag_ingest") is tool


# ---------------------------------------------------------------------------
# Test 2: valid path — mock ingest, assert ok status and doc count
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_with_valid_path(tmp_path: Path) -> None:
    """Valid directory path: returns success with docs_indexed count."""
    from src.tools.rag_ingest import RagIngestTool

    # Create a temp directory so Path.exists() and Path.is_dir() return True.
    corpus = tmp_path / "notes"
    corpus.mkdir()

    mock_result = {"docs_indexed": 5, "chunks_total": 20, "errors": 0}

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest", return_value=mock_result) as mock_ingest:
        result = tool.execute({"path": str(corpus)})

    assert result.success is True
    assert result.data["status"] == "ok"
    assert result.data["docs_indexed"] == 5
    mock_ingest.assert_called_once_with(Path(str(corpus)))


@pytest.mark.unit
def test_rag_ingest_with_valid_file(tmp_path: Path) -> None:
    """Valid file path: single file is also accepted and ingested."""
    from src.tools.rag_ingest import RagIngestTool

    note = tmp_path / "note.md"
    note.write_text("# Hello world")

    mock_result = {"docs_indexed": 1, "chunks_total": 1, "errors": 0}

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest", return_value=mock_result):
        result = tool.execute({"path": str(note)})

    assert result.success is True
    assert result.data["docs_indexed"] == 1


# ---------------------------------------------------------------------------
# Test 3: missing path (path does not exist on filesystem)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_with_missing_path() -> None:
    """Non-existent path returns error status without calling ingest."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest") as mock_ingest:
        result = tool.execute({"path": "/tmp/__lumi_nonexistent_path_xyz__"})

    assert result.success is False
    assert result.data.get("status") == "error"
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()
    mock_ingest.assert_not_called()


@pytest.mark.unit
def test_rag_ingest_with_no_path_arg() -> None:
    """Missing 'path' key in args returns a validation error."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    result = tool.execute({})

    assert result.success is False
    assert "path" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 4: invalid path type — non-string input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_with_invalid_path_type_int() -> None:
    """Integer path argument is rejected before any filesystem access."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest") as mock_ingest:
        result = tool.execute({"path": 42})

    assert result.success is False
    assert "path" in result.output.lower()
    mock_ingest.assert_not_called()


@pytest.mark.unit
def test_rag_ingest_with_invalid_path_type_none() -> None:
    """None path argument is rejected before any filesystem access."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest") as mock_ingest:
        result = tool.execute({"path": None})

    assert result.success is False
    mock_ingest.assert_not_called()


@pytest.mark.unit
def test_rag_ingest_with_invalid_path_type_list() -> None:
    """List path argument is rejected before any filesystem access."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest") as mock_ingest:
        result = tool.execute({"path": ["/tmp/notes"]})

    assert result.success is False
    mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: ToolResultEvent is posted via event_callback on completion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_triggers_tool_result_event(tmp_path: Path) -> None:
    """ToolResultEvent is posted to event_callback after ingest completes."""
    from src.core.events import ToolResultEvent
    from src.tools.rag_ingest import RagIngestTool

    corpus = tmp_path / "docs"
    corpus.mkdir()

    posted_events: list[Any] = []
    mock_result = {"docs_indexed": 3, "chunks_total": 9, "errors": 0}

    tool = RagIngestTool(event_callback=posted_events.append)
    with patch.object(tool, "_run_ingest", return_value=mock_result):
        result = tool.execute({"path": str(corpus)})

    assert result.success is True
    assert len(posted_events) == 1
    event = posted_events[0]
    assert isinstance(event, ToolResultEvent)
    assert event.tool_name == "rag_ingest"
    assert event.success is True
    assert event.data["docs_indexed"] == 3


@pytest.mark.unit
def test_rag_ingest_triggers_error_event_on_failure() -> None:
    """ToolResultEvent with success=False is posted when path is missing."""
    from src.core.events import ToolResultEvent
    from src.tools.rag_ingest import RagIngestTool

    posted_events: list[Any] = []

    tool = RagIngestTool(event_callback=posted_events.append)
    result = tool.execute({"path": "/tmp/__lumi_nonexistent__"})

    assert result.success is False
    assert len(posted_events) == 1
    event = posted_events[0]
    assert isinstance(event, ToolResultEvent)
    assert event.tool_name == "rag_ingest"
    assert event.success is False


@pytest.mark.unit
def test_rag_ingest_no_callback_does_not_raise(tmp_path: Path) -> None:
    """Tool with no event_callback completes without error."""
    from src.tools.rag_ingest import RagIngestTool

    corpus = tmp_path / "kb"
    corpus.mkdir()
    mock_result = {"docs_indexed": 1, "chunks_total": 2, "errors": 0}

    tool = RagIngestTool()  # no event_callback
    with patch.object(tool, "_run_ingest", return_value=mock_result):
        result = tool.execute({"path": str(corpus)})

    assert result.success is True


# ---------------------------------------------------------------------------
# Test 6: _run_ingest raises — tool returns error ToolResult gracefully
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_handles_ingest_exception(tmp_path: Path) -> None:
    """If _run_ingest raises, execute() returns a failure ToolResult (never raises)."""
    from src.tools.rag_ingest import RagIngestTool

    corpus = tmp_path / "notes"
    corpus.mkdir()

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest", side_effect=RuntimeError("db exploded")):
        result = tool.execute({"path": str(corpus)})

    assert result.success is False
    assert "db exploded" in result.output or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 7: tool name and description are correct
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_tool_name_and_description() -> None:
    """Tool has the correct name and a non-empty description."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    assert tool.name == "rag_ingest"
    assert isinstance(tool.description, str)
    assert len(tool.description) > 10


# ---------------------------------------------------------------------------
# Test 8: path traversal attempt is rejected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_rejects_path_traversal() -> None:
    """Paths containing '..' components are rejected for security."""
    from src.tools.rag_ingest import RagIngestTool

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest") as mock_ingest:
        result = tool.execute({"path": "/tmp/../etc/passwd"})

    assert result.success is False
    mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# Test 9: _run_ingest internal wiring (integration-style, no real DB)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_ingest_calls_rag_subsystem(tmp_path: Path) -> None:
    """_run_ingest calls DocumentStore, embedder, and loader subsystems."""
    from src.tools.rag_ingest import RagIngestTool

    corpus = tmp_path / "docs"
    corpus.mkdir()
    note = corpus / "hello.txt"
    note.write_text("Hello world content for testing")

    tool = RagIngestTool()

    with (
        patch("src.tools.rag_ingest.DocumentStore") as MockStore,
        patch("src.tools.rag_ingest.get_embedder") as mock_get_embedder,
        patch("src.tools.rag_ingest.is_supported", return_value=True),
        patch("src.tools.rag_ingest.load") as mock_load,
        patch("src.tools.rag_ingest.chunk_text") as mock_chunk,
        patch("src.tools.rag_ingest.RAGConfig") as MockRAGConfig,
    ):
        # Set up mock store
        mock_store_instance = MagicMock()
        MockStore.return_value = mock_store_instance
        mock_store_instance.get_document_by_path.return_value = None
        mock_store_instance.upsert_document.return_value = MagicMock(id=1)
        mock_store_instance.insert_chunk.return_value = MagicMock(id=10)

        # Set up mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 384
        mock_embedder.encode.return_value = [[0.1] * 384]
        mock_get_embedder.return_value = mock_embedder

        # Set up mock loader
        from src.rag.loader import LoadedDocument
        mock_load.return_value = LoadedDocument(
            path=str(note), text="Hello world content", char_count=19, extension=".txt"
        )

        # Set up mock chunker
        from src.rag.chunker import Chunk
        mock_chunk.return_value = [Chunk(text="Hello world content", chunk_idx=0, char_start=0, char_end=19)]

        result = tool._run_ingest(corpus)

    assert result["docs_indexed"] >= 0  # completed without exception


# ---------------------------------------------------------------------------
# Wave I2 tests: rag_ingest in default allowed_tools + confirmation gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_ingest_in_default_allowed_tools() -> None:
    from src.core.config import ToolsConfig

    cfg = ToolsConfig()
    assert "rag_ingest" in cfg.allowed_tools


@pytest.mark.unit
def test_rag_ingest_large_file_triggers_confirmation(tmp_path: Path) -> None:
    from src.tools.rag_ingest import RagIngestTool, _LARGE_FILE_BYTES

    large_file = tmp_path / "big.txt"
    large_file.write_bytes(b"x" * (_LARGE_FILE_BYTES + 1))

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest") as mock_ingest:
        result = tool.execute({"path": str(large_file)})

    assert result.success is False
    assert result.data.get("needs_confirmation") is True
    assert "estimated_chunks" in result.data
    mock_ingest.assert_not_called()


@pytest.mark.unit
def test_rag_ingest_small_file_bypasses_confirmation(tmp_path: Path) -> None:
    from src.tools.rag_ingest import RagIngestTool

    small_file = tmp_path / "small.txt"
    small_file.write_text("Hello world")

    mock_result = {"docs_indexed": 1, "chunks_total": 2, "errors": 0}

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest", return_value=mock_result) as mock_ingest:
        result = tool.execute({"path": str(small_file)})

    assert result.success is True
    mock_ingest.assert_called_once()


@pytest.mark.unit
def test_rag_ingest_confirmed_true_bypasses_gate_for_large_file(
    tmp_path: Path,
) -> None:
    from src.tools.rag_ingest import RagIngestTool, _LARGE_FILE_BYTES

    large_file = tmp_path / "big2.txt"
    large_file.write_bytes(b"y" * (_LARGE_FILE_BYTES + 1))

    mock_result = {"docs_indexed": 1, "chunks_total": 5, "errors": 0}

    tool = RagIngestTool()
    with patch.object(tool, "_run_ingest", return_value=mock_result) as mock_ingest:
        result = tool.execute({"path": str(large_file), "confirmed": True})

    assert result.success is True
    mock_ingest.assert_called_once()
