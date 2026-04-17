"""Tests for ZMQServer.on_llm_token() — LLM token streaming over IPC.

Validates that LLMTokenEvent is correctly forwarded as a wire frame
with event='llm_token' and the expected payload fields.
"""

from __future__ import annotations

import json
import queue
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import IPCConfig
from src.core.events import LLMTokenEvent
from src.core.state_machine import StateMachine
from src.core.zmq_server import ZMQServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_sent(mock_transport: MagicMock) -> dict[str, Any]:
    """Decode the bytes argument of the most recent send() call."""
    assert mock_transport.send.called, "transport.send() was never called"
    raw: bytes = mock_transport.send.call_args[0][0]
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_transport() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def zmq_server(mock_transport: MagicMock) -> ZMQServer:
    config = IPCConfig(address="tcp://127.0.0.1", port=5555)
    event_q: queue.Queue[Any] = queue.Queue()
    sm = StateMachine()
    with patch("src.core.zmq_server.IPCTransport", return_value=mock_transport):
        server = ZMQServer(config=config, event_queue=event_q, state_machine=sm)
    return server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_llm_token_sends_llm_token_frame(
    zmq_server: ZMQServer,
    mock_transport: MagicMock,
) -> None:
    """on_llm_token() must send a JSON frame with event='llm_token' and
    payload containing token and utterance_id."""
    event = LLMTokenEvent(token="hello", utterance_id="u1")
    zmq_server.on_llm_token(event)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "llm_token"
    assert sent["payload"] == {"token": "hello", "utterance_id": "u1"}


@pytest.mark.unit
def test_on_llm_token_token_in_payload(
    zmq_server: ZMQServer,
    mock_transport: MagicMock,
) -> None:
    """The 'token' key must be present in the encoded frame payload."""
    event = LLMTokenEvent(token="world", utterance_id="u2")
    zmq_server.on_llm_token(event)

    sent = _decode_sent(mock_transport)
    assert "token" in sent["payload"]
    assert sent["payload"]["token"] == "world"


@pytest.mark.unit
def test_on_llm_token_utterance_id_in_payload(
    zmq_server: ZMQServer,
    mock_transport: MagicMock,
) -> None:
    """The 'utterance_id' key must be present in the encoded frame payload."""
    event = LLMTokenEvent(token="x", utterance_id="u3")
    zmq_server.on_llm_token(event)

    sent = _decode_sent(mock_transport)
    assert "utterance_id" in sent["payload"]
    assert sent["payload"]["utterance_id"] == "u3"
