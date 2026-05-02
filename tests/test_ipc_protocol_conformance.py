"""
IPC protocol conformance test suite — Phase 5 Wave 4 (updated B3).

End-to-end integration tests that verify the full Python IPC stack using
FakeWSClient acting as the Tauri/React frontend.

Stack under test (no mocking of the transport layer):
    FakeWSClient  <--WS loopback-->  WSTransport  <-->  EventBridge
                                                             |
                                                      StateMachine / queue.Queue

Wire format (B3+): one UTF-8 JSON string per WebSocket message.

JSON envelope (outbound):
    {"event": str, "payload": dict, "timestamp": float, "version": "1.0"}

Fixture strategy:
- ``ipc_stack``   — starts WSTransport + EventBridge; tears down in finally.
- ``FakeTCPClient`` — alias for FakeWSClient; backward-compat method aliases
                      (send_message, recv_message, send_raw) preserved.

All recv_message() calls have an explicit timeout to prevent infinite hangs.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Generator

import pytest

from src.core.config import IPCConfig
from src.core.events import (
    InterruptEvent,
    UserTextEvent,
    VisemeEvent,
)
from src.core.state_machine import LumiState, StateMachine
from src.core.event_bridge import EventBridge
from tests.integration.fake_ws_client import FakeWSClient as FakeTCPClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECT_SETTLE_S: float = 0.08   # wait for async handler to register new client
_STOP_SETTLE_S: float = 0.10      # wait for handler to detect a close
_DEFAULT_RECV_TIMEOUT: float = 2.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_queue() -> queue.Queue[Any]:
    """A fresh event queue for each test."""
    return queue.Queue()


@pytest.fixture()
def state_machine() -> StateMachine:
    """A fresh StateMachine starting at IDLE."""
    return StateMachine()


@pytest.fixture()
def ipc_stack(
    event_queue: queue.Queue[Any],
    state_machine: StateMachine,
) -> Generator[tuple[EventBridge, int], None, None]:
    """Start a real WSTransport + EventBridge on an OS-assigned port.

    Uses port=0 so the OS assigns a free port atomically.  start() blocks
    until the WebSocket server is bound.

    Yields ``(event_bridge, bound_port)``.  Tears down in a ``finally`` block
    so the asyncio loop is always stopped even when the test body raises.
    """
    config = IPCConfig(address="127.0.0.1", port=0)
    server = EventBridge(
        config=config,
        event_queue=event_queue,
        state_machine=state_machine,
    )
    try:
        server.start()
        # Let the accept loop bind and start listening before tests connect.
        time.sleep(0.05)
        port = server.bound_port
        assert port is not None, "EventBridge.bound_port is None after start()"
        yield server, port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Test 1 — full state lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_state_lifecycle(
    ipc_stack: tuple[EventBridge, int],
    state_machine: StateMachine,
) -> None:
    """Connect a fake client, trigger IDLE→LISTENING, assert state_change frame.

    The EventBridge registers itself as a StateMachine observer at construction
    time.  When transition_to(LISTENING) fires, the observer calls _send()
    which writes a length-prefixed frame to the connected client.
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        # Wait for the accept loop to register the new connection.
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        state_machine.transition_to(LumiState.LISTENING)

        msg = client.recv_message(timeout=2.0)

    assert msg["event"] == "state_change"
    assert msg["payload"]["state"] == "listening"
    assert msg["version"] == "1.0"
    assert "timestamp" in msg


# ---------------------------------------------------------------------------
# Test 2 — interrupt from client posts InterruptEvent to queue
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_interrupt_returns_to_idle(
    ipc_stack: tuple[EventBridge, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends an 'interrupt' frame; EventBridge posts InterruptEvent.

    The EventBridge._on_raw_message callback fires on IPCTransport's recv
    daemon thread and puts an InterruptEvent(source='zmq') onto the queue.
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()
        client.send_message("interrupt", {})

        event = event_queue.get(timeout=2.0)

    assert isinstance(event, InterruptEvent)
    assert event.source == "zmq"


# ---------------------------------------------------------------------------
# Test 3 — user_text from client posts UserTextEvent to queue
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_user_text_triggers_event(
    ipc_stack: tuple[EventBridge, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends a 'user_text' frame; EventBridge posts UserTextEvent.

    The posted event must carry the exact 'text' value from the payload.
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()
        client.send_message("user_text", {"text": "hello"})

        event = event_queue.get(timeout=2.0)

    assert isinstance(event, UserTextEvent)
    assert event.text == "hello"


# ---------------------------------------------------------------------------
# Test 4 — viseme forwarding to client
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_viseme_forwarding(
    ipc_stack: tuple[EventBridge, int],
) -> None:
    """EventBridge.on_tts_viseme() sends a tts_viseme frame to the connected client.

    Verifies that viseme phoneme and duration_ms survive the encode→TCP→decode
    round-trip exactly.
    """
    zmq_server, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        viseme = VisemeEvent(
            utterance_id="utt-test",
            phoneme="AE",
            start_ms=0,
            duration_ms=80,
        )
        zmq_server.on_tts_viseme(viseme)

        msg = client.recv_message(timeout=2.0)

    assert msg["event"] == "tts_viseme"
    assert msg["payload"]["viseme"] == "AE"
    assert msg["payload"]["duration_ms"] == 80


# ---------------------------------------------------------------------------
# Test 5 — malformed client message: server does not crash
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_malformed_client_message_no_crash(
    ipc_stack: tuple[EventBridge, int],
    state_machine: StateMachine,
) -> None:
    """A valid length-prefixed frame carrying invalid JSON must not crash the server.

    After the malformed message the server must remain operational: a subsequent
    state transition must still produce a forwarded state_change frame.
    """
    zmq_server, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        # Send a frame whose body is not valid JSON.
        client.send_raw(b"not json }{{}}")

        # Small settle so the recv callback has time to process the bad frame.
        time.sleep(0.05)

        # Server must still be alive — trigger a state change and confirm it
        # arrives at the client.
        state_machine.transition_to(LumiState.LISTENING)

        msg = client.recv_message(timeout=2.0)

    assert msg["event"] == "state_change"
    assert msg["payload"]["state"] == "listening"


# ---------------------------------------------------------------------------
# Test 6 — client reconnect: second client receives events after first closes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_client_reconnect(
    ipc_stack: tuple[EventBridge, int],
    state_machine: StateMachine,
) -> None:
    """After the first client closes, a second client must receive subsequent events.

    IPCTransport implements a single-client model: when a new connection arrives
    (or is accepted after the previous one dropped), it becomes the active client.
    EventBridge.on_state_change() must deliver the frame to the new client.
    """
    _, port = ipc_stack

    # First client connects then disconnects (no handshake needed — closes immediately).
    first = FakeTCPClient(port)
    first.connect()
    time.sleep(_CONNECT_SETTLE_S)
    first.close()

    # Give _recv_loop time to detect the closure and clear _client_sock so
    # the accept loop is ready for the next connection.
    time.sleep(_STOP_SETTLE_S)

    # Second client connects.
    with FakeTCPClient(port) as second:
        time.sleep(_CONNECT_SETTLE_S)
        second.do_handshake()

        state_machine.transition_to(LumiState.LISTENING)

        msg = second.recv_message(timeout=2.0)

    assert msg["event"] == "state_change"
    assert msg["payload"]["state"] == "listening"
