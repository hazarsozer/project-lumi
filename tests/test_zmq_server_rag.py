"""Tests for EventBridge RAG event forwarding (Wave 4)."""

from __future__ import annotations

import json
import queue
from unittest.mock import MagicMock, call, patch

import pytest

from src.core.events import (
    RAGRetrievalEvent,
    RAGSetEnabledEvent,
    RAGStatusEvent,
)
from src.core.event_bridge import EventBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def server_and_queue(tmp_path):
    """Return an EventBridge wired to a real queue, with transport mocked out."""
    from src.core.config import IPCConfig
    from src.core.state_machine import StateMachine

    config = IPCConfig(address="tcp://127.0.0.1", port=0)
    eq: queue.Queue = queue.Queue()
    sm = StateMachine()

    with patch("src.core.event_bridge.IPCTransport") as MockTransport:
        mock_transport = MagicMock()
        MockTransport.return_value = mock_transport
        srv = EventBridge(config=config, event_queue=eq, state_machine=sm)

    return srv, eq, mock_transport


# ---------------------------------------------------------------------------
# Outbound: on_rag_retrieval
# ---------------------------------------------------------------------------

class TestOnRagRetrieval:
    def test_sends_rag_retrieval_event(self, server_and_queue):
        srv, eq, mock_transport = server_and_queue
        event = RAGRetrievalEvent(
            query="what is LightRAG?",
            hit_count=3,
            latency_ms=42,
            top_doc_paths=("docs/a.md", "docs/b.md"),
        )
        srv.on_rag_retrieval(event)

        assert mock_transport.send.call_count == 1
        frame: bytes = mock_transport.send.call_args[0][0]
        data = json.loads(frame.decode())

        assert data["event"] == "rag_retrieval"
        assert data["payload"]["query"] == "what is LightRAG?"
        assert data["payload"]["hit_count"] == 3
        assert data["payload"]["latency_ms"] == 42
        assert data["payload"]["top_doc_paths"] == ["docs/a.md", "docs/b.md"]
        assert data["version"] == "1.0"

    def test_empty_top_doc_paths(self, server_and_queue):
        srv, eq, mock_transport = server_and_queue
        event = RAGRetrievalEvent(
            query="hello",
            hit_count=0,
            latency_ms=5,
            top_doc_paths=(),
        )
        srv.on_rag_retrieval(event)

        frame = mock_transport.send.call_args[0][0]
        data = json.loads(frame.decode())
        assert data["payload"]["top_doc_paths"] == []


# ---------------------------------------------------------------------------
# Outbound: on_rag_status
# ---------------------------------------------------------------------------

class TestOnRagStatus:
    def test_sends_rag_status_event(self, server_and_queue):
        srv, eq, mock_transport = server_and_queue
        event = RAGStatusEvent(
            enabled=True,
            doc_count=12,
            chunk_count=48,
            last_indexed="2026-04-17T08:00:00",
        )
        srv.on_rag_status(event)

        assert mock_transport.send.call_count == 1
        frame = mock_transport.send.call_args[0][0]
        data = json.loads(frame.decode())

        assert data["event"] == "rag_status"
        assert data["payload"]["enabled"] is True
        assert data["payload"]["doc_count"] == 12
        assert data["payload"]["chunk_count"] == 48
        assert data["payload"]["last_indexed"] == "2026-04-17T08:00:00"

    def test_disabled_status(self, server_and_queue):
        srv, eq, mock_transport = server_and_queue
        event = RAGStatusEvent(enabled=False, doc_count=0, chunk_count=0, last_indexed="")
        srv.on_rag_status(event)

        frame = mock_transport.send.call_args[0][0]
        data = json.loads(frame.decode())
        assert data["payload"]["enabled"] is False
        assert data["payload"]["last_indexed"] == ""


# ---------------------------------------------------------------------------
# Inbound: rag_set_enabled
# ---------------------------------------------------------------------------

def _make_raw(event: str, payload: dict) -> bytes:
    import time
    return json.dumps(
        {"event": event, "payload": payload, "timestamp": time.time(), "version": "1.0"}
    ).encode()


class TestInboundRagSetEnabled:
    def test_posts_rag_set_enabled_true(self, server_and_queue):
        srv, eq, _ = server_and_queue
        raw = _make_raw("rag_set_enabled", {"enabled": True})
        srv._on_raw_message(raw)

        posted = eq.get_nowait()
        assert isinstance(posted, RAGSetEnabledEvent)
        assert posted.enabled is True

    def test_posts_rag_set_enabled_false(self, server_and_queue):
        srv, eq, _ = server_and_queue
        raw = _make_raw("rag_set_enabled", {"enabled": False})
        srv._on_raw_message(raw)

        posted = eq.get_nowait()
        assert isinstance(posted, RAGSetEnabledEvent)
        assert posted.enabled is False

    def test_missing_enabled_field_drops(self, server_and_queue):
        srv, eq, _ = server_and_queue
        raw = _make_raw("rag_set_enabled", {})
        srv._on_raw_message(raw)
        assert eq.empty()

    def test_non_bool_enabled_drops(self, server_and_queue):
        srv, eq, _ = server_and_queue
        raw = _make_raw("rag_set_enabled", {"enabled": "yes"})
        srv._on_raw_message(raw)
        assert eq.empty()

    def test_unknown_event_still_drops(self, server_and_queue):
        srv, eq, _ = server_and_queue
        raw = _make_raw("unknown_event", {})
        srv._on_raw_message(raw)
        assert eq.empty()
