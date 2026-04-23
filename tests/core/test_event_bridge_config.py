"""
Unit tests for the Wave S1 config IPC additions in EventBridge.

Covers the 4 new pieces added to src/core/event_bridge.py:
  - _handle_config_schema_request  (inbound)
  - _handle_config_update          (inbound)
  - send_config_schema             (outbound)
  - send_config_update_result      (outbound)

Strategy
--------
- IPCTransport is fully mocked so no real socket is created.
- Inbound tests call ``_on_raw_message(raw_bytes)`` directly, the same way
  IPCTransport's recv daemon thread would call it.
- Outbound tests assert the encoded JSON frame passed to ``_transport.send``.
- A real ``queue.Queue`` is used; ``queue.get(timeout=0.5)`` is used to avoid
  infinite hangs when a test checks that nothing was posted.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import IPCConfig
from src.core.events import ConfigSchemaRequestEvent, ConfigUpdateEvent
from src.core.event_bridge import EventBridge
from src.core.state_machine import StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wire_frame(event: str, payload: dict[str, Any]) -> bytes:
    """Build a raw UTF-8 JSON wire frame (without the length prefix).

    Matches the wire envelope expected by EventBridge._decode().
    """
    envelope = {
        "event": event,
        "payload": payload,
        "timestamp": time.time(),
        "version": "1.0",
    }
    return json.dumps(envelope, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_queue() -> queue.Queue[Any]:
    """Fresh event queue for each test."""
    return queue.Queue()


@pytest.fixture()
def state_machine() -> StateMachine:
    """Fresh StateMachine starting at IDLE."""
    return StateMachine()


@pytest.fixture()
def bridge(
    event_queue: queue.Queue[Any],
    state_machine: StateMachine,
) -> EventBridge:
    """EventBridge with IPCTransport fully mocked — no real socket.

    We patch IPCTransport at the point it is imported by event_bridge so no
    bind() or thread.start() calls reach the OS.
    """
    config = IPCConfig(address="tcp://127.0.0.1", port=5555)

    with patch("src.core.event_bridge.IPCTransport") as MockTransport:
        mock_transport_instance = MagicMock()
        MockTransport.return_value = mock_transport_instance

        eb = EventBridge(
            config=config,
            event_queue=event_queue,
            state_machine=state_machine,
        )
        # Attach the mock instance as a test-accessible attribute so tests
        # can assert on transport.send() calls.
        eb._mock_transport = mock_transport_instance  # type: ignore[attr-defined]

    return eb


# ---------------------------------------------------------------------------
# Inbound: config_schema_request
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_schema_request_posts_event(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """A valid config_schema_request frame posts ConfigSchemaRequestEvent to queue."""
    raw = _make_wire_frame("config_schema_request", {})
    bridge._on_raw_message(raw)

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, ConfigSchemaRequestEvent)


@pytest.mark.unit
def test_config_schema_request_ignores_payload_content(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """Payload is ignored — event is always posted regardless of payload content."""
    raw = _make_wire_frame("config_schema_request", {"unexpected_field": 42})
    bridge._on_raw_message(raw)

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, ConfigSchemaRequestEvent)
    # Queue must be empty after the single event.
    assert event_queue.empty()


# ---------------------------------------------------------------------------
# Inbound: config_update
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_update_posts_event_with_persist_true(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """Valid config_update with persist=True posts ConfigUpdateEvent to queue."""
    changes = {"audio.sensitivity": 0.7}
    raw = _make_wire_frame("config_update", {"changes": changes, "persist": True})
    bridge._on_raw_message(raw)

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, ConfigUpdateEvent)
    assert event.changes == {"audio.sensitivity": 0.7}
    assert event.persist is True


@pytest.mark.unit
def test_config_update_posts_event_with_persist_false(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """Valid config_update with persist=False posts ConfigUpdateEvent with persist=False."""
    changes = {"tts.volume": 0.8}
    raw = _make_wire_frame("config_update", {"changes": changes, "persist": False})
    bridge._on_raw_message(raw)

    event = event_queue.get(timeout=0.5)
    assert isinstance(event, ConfigUpdateEvent)
    assert event.changes == {"tts.volume": 0.8}
    assert event.persist is False


@pytest.mark.unit
def test_config_update_missing_changes_drops_message(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """config_update payload without 'changes' must not post any event."""
    raw = _make_wire_frame("config_update", {"persist": True})
    bridge._on_raw_message(raw)

    with pytest.raises(queue.Empty):
        event_queue.get(timeout=0.2)


@pytest.mark.unit
def test_config_update_changes_not_dict_drops_message(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """config_update where 'changes' is not a dict must not post any event."""
    raw = _make_wire_frame("config_update", {"changes": "audio.sensitivity=0.7", "persist": False})
    bridge._on_raw_message(raw)

    with pytest.raises(queue.Empty):
        event_queue.get(timeout=0.2)


@pytest.mark.unit
def test_config_update_persist_not_bool_drops_message(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
) -> None:
    """config_update where 'persist' is not a bool must not post any event."""
    raw = _make_wire_frame("config_update", {"changes": {"audio.sensitivity": 0.5}, "persist": "yes"})
    bridge._on_raw_message(raw)

    with pytest.raises(queue.Empty):
        event_queue.get(timeout=0.2)


@pytest.mark.unit
def test_config_update_missing_changes_logs_warning(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing 'changes' must emit a WARNING-level log (no exception raised)."""
    import logging

    raw = _make_wire_frame("config_update", {"persist": True})

    with caplog.at_level(logging.WARNING, logger="src.core.event_bridge"):
        bridge._on_raw_message(raw)  # must not raise

    assert any("changes" in record.message for record in caplog.records)


@pytest.mark.unit
def test_config_update_persist_not_bool_logs_warning(
    bridge: EventBridge,
    event_queue: queue.Queue[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-bool 'persist' must emit a WARNING-level log (no exception raised)."""
    import logging

    raw = _make_wire_frame(
        "config_update",
        {"changes": {"audio.sensitivity": 0.5}, "persist": "yes"},
    )

    with caplog.at_level(logging.WARNING, logger="src.core.event_bridge"):
        bridge._on_raw_message(raw)  # must not raise

    assert any("persist" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Outbound: send_config_schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_send_config_schema_calls_transport(bridge: EventBridge) -> None:
    """send_config_schema() must call _transport.send with a valid JSON frame."""
    fields = {"audio.sensitivity": {"label": "Sensitivity", "control": "slider"}}
    current_values = {"audio.sensitivity": 0.5}

    bridge.send_config_schema(fields=fields, current_values=current_values)

    mock_transport = bridge._mock_transport  # type: ignore[attr-defined]
    assert mock_transport.send.called, "_transport.send was not called"

    call_args = mock_transport.send.call_args
    raw: bytes = call_args[0][0]
    decoded = json.loads(raw.decode("utf-8"))

    assert decoded["event"] == "config_schema"
    assert decoded["version"] == "1.0"
    assert "timestamp" in decoded
    payload = decoded["payload"]
    assert payload["fields"] == fields
    assert payload["current_values"] == current_values


@pytest.mark.unit
def test_send_config_schema_payload_contains_required_keys(bridge: EventBridge) -> None:
    """The config_schema payload must always contain 'fields' and 'current_values'."""
    bridge.send_config_schema(fields={}, current_values={})

    mock_transport = bridge._mock_transport  # type: ignore[attr-defined]
    raw: bytes = mock_transport.send.call_args[0][0]
    payload = json.loads(raw.decode("utf-8"))["payload"]

    assert "fields" in payload
    assert "current_values" in payload


# ---------------------------------------------------------------------------
# Outbound: send_config_update_result
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_send_config_update_result_calls_transport(bridge: EventBridge) -> None:
    """send_config_update_result() must call _transport.send with a valid JSON frame."""
    bridge.send_config_update_result(
        applied_live=["audio.sensitivity"],
        pending_restart=["llm.model_path"],
        errors={"unknown.key": "Unknown config key"},
    )

    mock_transport = bridge._mock_transport  # type: ignore[attr-defined]
    assert mock_transport.send.called, "_transport.send was not called"

    raw: bytes = mock_transport.send.call_args[0][0]
    decoded = json.loads(raw.decode("utf-8"))

    assert decoded["event"] == "config_update_result"
    assert decoded["version"] == "1.0"
    assert "timestamp" in decoded

    payload = decoded["payload"]
    assert payload["applied_live"] == ["audio.sensitivity"]
    assert payload["pending_restart"] == ["llm.model_path"]
    assert payload["errors"] == {"unknown.key": "Unknown config key"}


@pytest.mark.unit
def test_send_config_update_result_empty_lists(bridge: EventBridge) -> None:
    """send_config_update_result() with all-empty collections sends a valid frame."""
    bridge.send_config_update_result(
        applied_live=[],
        pending_restart=[],
        errors={},
    )

    mock_transport = bridge._mock_transport  # type: ignore[attr-defined]
    raw: bytes = mock_transport.send.call_args[0][0]
    payload = json.loads(raw.decode("utf-8"))["payload"]

    assert payload["applied_live"] == []
    assert payload["pending_restart"] == []
    assert payload["errors"] == {}


@pytest.mark.unit
def test_send_config_update_result_payload_contains_required_keys(
    bridge: EventBridge,
) -> None:
    """The config_update_result payload must always have all three required keys."""
    bridge.send_config_update_result(
        applied_live=[], pending_restart=[], errors={}
    )

    mock_transport = bridge._mock_transport  # type: ignore[attr-defined]
    raw: bytes = mock_transport.send.call_args[0][0]
    payload = json.loads(raw.decode("utf-8"))["payload"]

    assert "applied_live" in payload
    assert "pending_restart" in payload
    assert "errors" in payload
