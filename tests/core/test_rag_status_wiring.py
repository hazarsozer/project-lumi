"""
Unit tests for RAGStatus request/response wiring.

Covers:
  - EventBridge inbound: rag_status_request → RAGStatusRequestEvent posted to queue
  - EventBridge outbound: on_rag_status() → correct wire frame sent
  - Orchestrator _handle_rag_status_request with RAG disabled (zeroes)
  - Orchestrator _handle_rag_status_request with RAG active (real stats)
"""

from __future__ import annotations

import json
import queue
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import IPCConfig
from src.core.event_bridge import EventBridge
from src.core.events import RAGStatusEvent, RAGStatusRequestEvent
from src.core.state_machine import StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wire(event: str, payload: dict[str, Any] | None = None) -> bytes:
    envelope = {
        "event": event,
        "payload": payload or {},
        "timestamp": time.time(),
        "version": "1.0",
    }
    return json.dumps(envelope, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_queue() -> queue.Queue[Any]:
    return queue.Queue()


@pytest.fixture()
def bridge(event_queue: queue.Queue[Any]) -> EventBridge:
    config = IPCConfig(address="127.0.0.1", port=5555)
    with patch("src.core.event_bridge.WSTransport") as MockTransport:
        mock_transport_instance = MagicMock()
        MockTransport.return_value = mock_transport_instance
        eb = EventBridge(
            config=config,
            event_queue=event_queue,
            state_machine=StateMachine(),
        )
        eb._mock_transport = mock_transport_instance  # type: ignore[attr-defined]
    return eb


# ---------------------------------------------------------------------------
# Inbound: rag_status_request
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_status_request_posts_event(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """rag_status_request inbound frame must post RAGStatusRequestEvent."""
    bridge._on_raw_message(_wire("rag_status_request"))

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, RAGStatusRequestEvent)


@pytest.mark.unit
def test_rag_status_request_posts_exactly_one_event(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """rag_status_request must post exactly one event, no extras."""
    bridge._on_raw_message(_wire("rag_status_request"))

    event_queue.get(timeout=0.5)  # consume the one expected event
    assert event_queue.empty()


# ---------------------------------------------------------------------------
# Outbound: on_rag_status
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_rag_status_sends_correct_frame(bridge: EventBridge) -> None:
    """on_rag_status() must send a wire frame with all RAGStatusEvent fields."""
    event = RAGStatusEvent(
        enabled=True,
        doc_count=42,
        chunk_count=188,
        last_indexed="2026-04-30T20:00:00+00:00",
    )
    bridge.on_rag_status(event)

    raw_call = bridge._mock_transport.send.call_args  # type: ignore[attr-defined]
    assert raw_call is not None, "transport.send() was not called"

    sent_bytes = raw_call[0][0]
    decoded = json.loads(sent_bytes.decode("utf-8"))

    assert decoded["event"] == "rag_status"
    payload = decoded["payload"]
    assert payload["enabled"] is True
    assert payload["doc_count"] == 42
    assert payload["chunk_count"] == 188
    assert payload["last_indexed"] == "2026-04-30T20:00:00+00:00"


# ---------------------------------------------------------------------------
# Orchestrator handler: RAG disabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orchestrator_rag_status_request_rag_disabled() -> None:
    """Handler posts RAGStatusEvent with zeroes when RAG is not active."""
    from src.audio.speaker import SpeakerThread
    from src.core.config import LumiConfig
    from src.core.orchestrator import Orchestrator

    # LumiConfig() defaults to rag.enabled=False — no mutation needed.
    mock_speaker = MagicMock(spec=SpeakerThread)
    orch = Orchestrator(config=LumiConfig(), speaker=mock_speaker)

    orch._handle_rag_status_request(RAGStatusRequestEvent())

    event = orch._event_queue.get(timeout=0.5)
    assert isinstance(event, RAGStatusEvent)
    assert event.enabled is False
    assert event.doc_count == 0
    assert event.chunk_count == 0
    assert event.last_indexed == ""


@pytest.mark.unit
def test_orchestrator_rag_status_request_no_retriever() -> None:
    """Handler posts RAGStatusEvent(enabled=False) when retriever failed to init."""
    import dataclasses

    from src.audio.speaker import SpeakerThread
    from src.core.config import LumiConfig, RAGConfig
    from src.core.orchestrator import Orchestrator

    # Use dataclasses.replace to build a frozen config with rag.enabled=True.
    rag_cfg = RAGConfig(enabled=True)
    config = dataclasses.replace(LumiConfig(), rag=rag_cfg)
    mock_speaker = MagicMock(spec=SpeakerThread)

    # Patch DocumentStore (lazy-imported inside orchestrator's try block) to fail.
    with patch("src.rag.store.DocumentStore", side_effect=RuntimeError("no db")):
        orch = Orchestrator(config=config, speaker=mock_speaker)

    assert orch._rag_retriever is None  # init failure path

    orch._handle_rag_status_request(RAGStatusRequestEvent())

    event = orch._event_queue.get(timeout=0.5)
    assert isinstance(event, RAGStatusEvent)
    assert event.enabled is False
    assert event.doc_count == 0


# ---------------------------------------------------------------------------
# Orchestrator handler: RAG active with mock store
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orchestrator_rag_status_request_with_active_rag() -> None:
    """Handler posts RAGStatusEvent with live stats when RAG is active."""
    from src.audio.speaker import SpeakerThread
    from src.core.config import LumiConfig
    from src.core.orchestrator import Orchestrator
    from src.rag.store import StoreStats

    # Start with RAG disabled so no real DB/model init happens.
    mock_speaker = MagicMock(spec=SpeakerThread)
    orch = Orchestrator(config=LumiConfig(), speaker=mock_speaker)

    # Manually inject a mock RAG store and retriever as if init succeeded.
    mock_store = MagicMock()
    mock_store.stats.return_value = StoreStats(
        doc_count=7,
        chunk_count=31,
        last_indexed=1746000000.0,  # fixed unix timestamp
    )
    orch._rag_store = mock_store
    orch._rag_retriever = MagicMock()
    orch._rag_runtime_enabled = True

    orch._handle_rag_status_request(RAGStatusRequestEvent())

    event = orch._event_queue.get(timeout=0.5)
    assert isinstance(event, RAGStatusEvent)
    assert event.enabled is True
    assert event.doc_count == 7
    assert event.chunk_count == 31
    assert "2025" in event.last_indexed  # ISO-8601 year
    assert event.last_indexed.endswith("+00:00")
