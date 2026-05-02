"""
Unit tests for EventBridge (src/core/event_bridge.py).

Strategy:
- WSTransport is mocked at the class level using pytest-mock / unittest.mock
  so no real TCP sockets are opened.  All ``send()`` call arguments are
  decoded from UTF-8 JSON so assertions are structural rather than brittle
  byte-string comparisons.
- A real ``queue.Queue`` is used for inbound events so the tests exercise the
  actual put/get contract; ``queue.get(timeout=0.5)`` is used to avoid
  hanging in CI if a put is accidentally omitted.
- StateMachine is a real instance so observer registration is exercised.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import IPCConfig
from src.core.events import (
    InterruptEvent,
    LLMResponseReadyEvent,
    SpeechCompletedEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    VisemeEvent,
    ZMQMessage,
)
from src.core.state_machine import LumiState, StateMachine
from src.core.event_bridge import EventBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wire_bytes(event: str, payload: dict[str, Any]) -> bytes:
    """Encode a valid wire frame for use as simulated inbound data."""
    data = {
        "event": event,
        "payload": payload,
        "timestamp": time.time(),
        "version": "1.0",
    }
    return json.dumps(data).encode("utf-8")


def _decode_sent(mock_transport: MagicMock) -> dict[str, Any]:
    """Decode the bytes argument of the most recent ``send()`` call."""
    assert mock_transport.send.called, "transport.send() was never called"
    raw: bytes = mock_transport.send.call_args[0][0]
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ipc_config() -> IPCConfig:
    return IPCConfig(address="127.0.0.1", port=5555)


@pytest.fixture()
def event_queue() -> queue.Queue[Any]:
    return queue.Queue()


@pytest.fixture()
def state_machine() -> StateMachine:
    return StateMachine()


@pytest.fixture()
def mock_transport() -> MagicMock:
    """A MagicMock that stands in for WSTransport."""
    transport = MagicMock()
    return transport


@pytest.fixture()
def zmq_server(
    ipc_config: IPCConfig,
    event_queue: queue.Queue[Any],
    state_machine: StateMachine,
    mock_transport: MagicMock,
) -> EventBridge:
    """Construct an EventBridge with WSTransport patched out."""
    with patch("src.core.event_bridge.WSTransport", return_value=mock_transport):
        server = EventBridge(
            config=ipc_config,
            event_queue=event_queue,
            state_machine=state_machine,
        )
    return server


# ---------------------------------------------------------------------------
# Test 1 — state_change sends correct JSON
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_state_change_sends_correct_json(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """on_state_change() must send a JSON frame with event='state_change'
    and payload containing the new state's value string."""
    zmq_server.on_state_change(LumiState.IDLE, LumiState.LISTENING)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "state_change"
    assert sent["payload"] == {"state": "listening"}
    assert "timestamp" in sent
    assert sent["version"] == "1.0"


# ---------------------------------------------------------------------------
# Test 2 — inbound interrupt posts InterruptEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inbound_interrupt_posts_event(
    zmq_server: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """A valid inbound 'interrupt' frame must post InterruptEvent to the queue."""
    frame = _make_wire_bytes("interrupt", {})
    zmq_server._on_raw_message(frame)

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, InterruptEvent)
    assert event.source == "zmq"


# ---------------------------------------------------------------------------
# Test 3 — inbound user_text posts UserTextEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inbound_user_text_posts_event(
    zmq_server: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """A valid inbound 'user_text' frame must post UserTextEvent with correct text."""
    frame = _make_wire_bytes("user_text", {"text": "hello"})
    zmq_server._on_raw_message(frame)

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, UserTextEvent)
    assert event.text == "hello"


# ---------------------------------------------------------------------------
# Test 4 — malformed inbound JSON is logged and dropped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_malformed_inbound_logged_and_dropped(
    zmq_server: EventBridge,
    event_queue: queue.Queue[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_on_raw_message() with invalid JSON must not raise and must not put
    anything on the queue.  A WARNING must be emitted."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.core.event_bridge"):
        zmq_server._on_raw_message(b"not json {{{")

    assert event_queue.empty()
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warning_records, "Expected at least one WARNING log for malformed JSON"


# ---------------------------------------------------------------------------
# Test 5 — _encode/_decode roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_encode_decode_roundtrip() -> None:
    """Encoding then decoding a frame must produce a ZMQMessage with the
    same event name and payload."""
    original_payload: dict[str, Any] = {"key": "val", "num": 42}
    raw = EventBridge._encode("test_event", original_payload)
    result = EventBridge._decode(raw)

    assert result is not None
    assert isinstance(result, ZMQMessage)
    assert result.event == "test_event"
    assert result.payload == original_payload
    assert result.version == "1.0"
    assert isinstance(result.timestamp, float)


# ---------------------------------------------------------------------------
# Test 6 — viseme event is forwarded correctly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_viseme_event_forwarded(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """on_tts_viseme() must send JSON with event='tts_viseme' and the
    correct viseme/duration_ms fields derived from VisemeEvent."""
    viseme = VisemeEvent(
        utterance_id="utt-001",
        phoneme="AH",
        start_ms=100,
        duration_ms=80,
    )
    zmq_server.on_tts_viseme(viseme)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "tts_viseme"
    assert sent["payload"]["viseme"] == "AH"
    assert sent["payload"]["duration_ms"] == 80


# ---------------------------------------------------------------------------
# Test 7 — error event is forwarded correctly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_error_event_forwarded(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """on_error() must send JSON with event='error' and the supplied code
    and message in the payload."""
    zmq_server.on_error("LLM_TIMEOUT", "model did not respond")

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "error"
    assert sent["payload"]["code"] == "LLM_TIMEOUT"
    assert sent["payload"]["message"] == "model did not respond"


# ---------------------------------------------------------------------------
# Test 8 — unknown inbound event is dropped with a warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_inbound_event_dropped(
    zmq_server: EventBridge,
    event_queue: queue.Queue[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An inbound frame with an unknown event name must be dropped silently
    (queue stays empty) and a WARNING must be emitted."""
    import logging

    frame = _make_wire_bytes("unknown_xyz", {})

    with caplog.at_level(logging.WARNING, logger="src.core.event_bridge"):
        zmq_server._on_raw_message(frame)

    assert event_queue.empty()
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warning_records, "Expected a WARNING for unknown event type"
    assert any("unknown_xyz" in r.message for r in warning_records)


# ---------------------------------------------------------------------------
# Bonus — missing required fields are dropped with a warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_required_field_logged_and_dropped(
    zmq_server: EventBridge,
    event_queue: queue.Queue[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSON object that is missing 'payload' must be dropped with a WARNING."""
    import logging

    # Valid JSON but missing 'payload' field
    incomplete = json.dumps({
        "event": "interrupt",
        "timestamp": time.time(),
        "version": "1.0",
    }).encode("utf-8")

    with caplog.at_level(logging.WARNING, logger="src.core.event_bridge"):
        zmq_server._on_raw_message(incomplete)

    assert event_queue.empty()
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warning_records, "Expected a WARNING for missing 'payload' field"


# ---------------------------------------------------------------------------
# Bonus — additional outbound handler coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tts_start_forwarded(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """on_tts_start() must send JSON with event='tts_start'."""
    llm_event = LLMResponseReadyEvent(text="Hello there!")
    zmq_server.on_tts_start(llm_event)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "tts_start"
    assert sent["payload"]["text"] == "Hello there!"
    assert sent["payload"]["duration_ms"] == 0


@pytest.mark.unit
def test_tts_stop_forwarded(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """on_tts_stop() must send JSON with event='tts_stop' and empty payload."""
    speech_event = SpeechCompletedEvent(utterance_id="utt-001")
    zmq_server.on_tts_stop(speech_event)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "tts_stop"
    assert sent["payload"] == {}


@pytest.mark.unit
def test_transcript_forwarded(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """on_transcript() must send JSON with event='transcript'."""
    transcript_event = TranscriptReadyEvent(text="what is the weather?")
    zmq_server.on_transcript(transcript_event)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "transcript"
    assert sent["payload"]["text"] == "what is the weather?"


@pytest.mark.unit
def test_start_delegates_to_transport(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """start() must delegate to transport.start()."""
    zmq_server.start()
    mock_transport.start.assert_called_once()


@pytest.mark.unit
def test_stop_delegates_to_transport(
    zmq_server: EventBridge,
    mock_transport: MagicMock,
) -> None:
    """stop() must delegate to transport.stop()."""
    zmq_server.stop()
    mock_transport.stop.assert_called_once()


@pytest.mark.unit
def test_user_text_with_empty_string_is_dropped(
    zmq_server: EventBridge,
    event_queue: queue.Queue[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An inbound user_text frame with an empty string must be dropped."""
    import logging

    frame = _make_wire_bytes("user_text", {"text": ""})

    with caplog.at_level(logging.WARNING, logger="src.core.event_bridge"):
        zmq_server._on_raw_message(frame)

    assert event_queue.empty()
    assert any("empty" in r.message.lower() for r in caplog.records if r.levelname == "WARNING")


@pytest.mark.unit
def test_state_machine_observer_registered(
    ipc_config: IPCConfig,
    event_queue: queue.Queue[Any],
    state_machine: StateMachine,
    mock_transport: MagicMock,
) -> None:
    """EventBridge must register itself as an observer on the StateMachine so
    transitions automatically trigger on_state_change."""
    with patch("src.core.event_bridge.WSTransport", return_value=mock_transport):
        server = EventBridge(
            config=ipc_config,
            event_queue=event_queue,
            state_machine=state_machine,
        )

    # Trigger a real state transition — the observer should fire and call send().
    state_machine.transition_to(LumiState.LISTENING)

    sent = _decode_sent(mock_transport)
    assert sent["event"] == "state_change"
    assert sent["payload"]["state"] == "listening"
