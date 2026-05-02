"""
Full-turn IPC integration tests — UI ↔ Brain WebSocket round-trip.

These tests validate that the complete UI → Brain → UI round-trip works
correctly over a real WebSocket connection, going beyond the unit-level
protocol conformance tests in ``tests/test_ipc_protocol_conformance.py``.

Stack under test (no mocking of the transport layer):
    FakeWSClient  <--WS loopback-->  WSTransport  <-->  EventBridge
                                                             |
                                                      StateMachine / queue.Queue

Each test:
- Spins up a real ``WSTransport`` + ``EventBridge`` on OS-assigned port 0.
- Connects a ``FakeWSClient`` that mimics what the Tauri/React frontend does.
- Performs a single meaningful turn (send inbound event OR receive outbound event).
- Tears down cleanly via fixture ``finally`` blocks.

Mocking strategy
----------------
No mocking is used in this file.  The transport and server layers run for real
on loopback.  The only concession to test isolation is:
- ``port=0`` (OS-assigned) — eliminates TOCTOU races between free-port probing
  and binding.
- Short ``time.sleep()`` calls to let async handlers settle.
- ``queue.Queue.get(timeout=2.0)`` and ``recv_frame(timeout=2.0)`` prevent
  infinite hangs; any failure within 2 s surfaces as a timeout error, not a
  hung test process.

Markers
-------
All tests carry ``@pytest.mark.integration`` and are collected only when that
marker is not excluded.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Generator

import pytest

from src.core.config import IPCConfig
from src.core.events import (
    InterruptEvent,
    LLMResponseReadyEvent,
    RAGSetEnabledEvent,
    UserTextEvent,
)
from src.core.state_machine import LumiState, StateMachine
from src.core.event_bridge import EventBridge
from tests.integration.fake_ws_client import FakeWSClient as FakeTCPClient

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# Time (seconds) to let the WSTransport asyncio handler register a new client
# after the WebSocket handshake completes.
_CONNECT_SETTLE_S: float = 0.08

# Time (seconds) to let the async message handler process a just-sent frame
# and post an event to the queue before the test reads the queue.
_RECV_SETTLE_S: float = 0.05

# Maximum seconds any single recv_frame() or queue.get() may block.
_RECV_TIMEOUT_S: float = 2.0


# ---------------------------------------------------------------------------
# Shared fixtures
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
def ipc_stack(
    event_queue: queue.Queue[Any],
    state_machine: StateMachine,
) -> Generator[tuple[EventBridge, int], None, None]:
    """Start a real ``WSTransport`` + ``EventBridge`` on an OS-assigned port.

    Passes ``port=0`` so the OS assigns a free port atomically.
    ``start()`` blocks until the WebSocket server is bound, so
    ``EventBridge.bound_port`` is already valid when it returns.

    Yields:
        ``(event_bridge, bound_port)``

    Tears down in a ``finally`` block so the asyncio loop is always
    stopped even when the test body raises an exception.
    """
    config = IPCConfig(address="127.0.0.1", port=0)
    server = EventBridge(
        config=config,
        event_queue=event_queue,
        state_machine=state_machine,
    )
    try:
        server.start()
        # Give the accept loop thread time to reach the select() call so it
        # is ready to accept connections before any test code connects.
        time.sleep(0.05)
        port = server.bound_port
        assert port is not None, "EventBridge.bound_port is None after start()"
        yield server, port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Test 1 — client receives state_change on state transition
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_client_receives_state_change_on_connect(
    ipc_stack: tuple[EventBridge, int],
    state_machine: StateMachine,
) -> None:
    """Connected client receives a ``state_change`` frame when the state transitions.

    The EventBridge registers itself as a StateMachine observer at construction.
    When ``transition_to(LISTENING)`` fires, the observer's ``on_state_change``
    callback serialises the new state to a JSON frame and sends it over the TCP
    connection to the fake client.

    Validates:
    - ``event == "state_change"``
    - ``payload.state == "listening"``
    - ``version == "1.0"``
    - ``timestamp`` field is present
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        # Let the accept loop register the new connection.
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        state_machine.transition_to(LumiState.LISTENING)

        msg = client.recv_frame(timeout=_RECV_TIMEOUT_S)

    assert msg["event"] == "state_change"
    assert msg["payload"]["state"] == "listening"
    assert msg["version"] == "1.0"
    assert "timestamp" in msg


# ---------------------------------------------------------------------------
# Test 2 — client sends interrupt → InterruptEvent posted to queue
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_client_send_interrupt_posts_to_queue(
    ipc_stack: tuple[EventBridge, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends ``interrupt`` frame; EventBridge posts ``InterruptEvent`` to queue.

    The EventBridge ``_on_raw_message`` callback runs on IPCTransport's recv
    daemon thread.  It decodes the frame, matches the ``"interrupt"`` event
    name, and calls ``_handle_interrupt`` which puts an ``InterruptEvent``
    onto ``event_queue``.

    Validates:
    - The dequeued object is an ``InterruptEvent``.
    - ``event.source == "zmq"``.
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()
        client.send_frame("interrupt", {})

        event = event_queue.get(timeout=_RECV_TIMEOUT_S)

    assert isinstance(event, InterruptEvent)
    assert event.source == "zmq"


# ---------------------------------------------------------------------------
# Test 3 — client sends user_text → UserTextEvent posted to queue
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_client_send_user_text_posts_to_queue(
    ipc_stack: tuple[EventBridge, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends ``user_text`` frame; EventBridge posts ``UserTextEvent`` to queue.

    The ``_handle_user_text`` method validates that the payload contains a
    non-empty ``"text"`` string before posting to the queue, so the test also
    implicitly confirms that valid payloads are not dropped.

    Validates:
    - The dequeued object is a ``UserTextEvent``.
    - ``event.text == "hello"``.
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()
        client.send_frame("user_text", {"text": "hello"})

        event = event_queue.get(timeout=_RECV_TIMEOUT_S)

    assert isinstance(event, UserTextEvent)
    assert event.text == "hello"


# ---------------------------------------------------------------------------
# Test 4 — client sends rag_set_enabled → RAGSetEnabledEvent posted to queue
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_client_send_rag_set_enabled_posts_to_queue(
    ipc_stack: tuple[EventBridge, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends ``rag_set_enabled`` frame; EventBridge posts ``RAGSetEnabledEvent``.

    The ``_handle_rag_set_enabled`` method validates that ``payload["enabled"]``
    is a ``bool`` before posting.  This test confirms the happy path.

    Validates:
    - The dequeued object is a ``RAGSetEnabledEvent``.
    - ``event.enabled == True``.
    """
    _, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()
        client.send_frame("rag_set_enabled", {"enabled": True})

        event = event_queue.get(timeout=_RECV_TIMEOUT_S)

    assert isinstance(event, RAGSetEnabledEvent)
    assert event.enabled is True


# ---------------------------------------------------------------------------
# Test 5 — malformed frame does not crash server
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_malformed_frame_does_not_crash_server(
    ipc_stack: tuple[EventBridge, int],
    state_machine: StateMachine,
) -> None:
    """A valid length-prefix carrying invalid JSON body must not crash the server.

    ``EventBridge._on_raw_message`` calls ``_decode()`` which catches
    ``json.JSONDecodeError`` and logs a WARNING, returning ``None`` so the
    dispatch path is skipped entirely.

    After the malformed message the server must remain fully operational.
    Verified by triggering a state transition and confirming the client still
    receives the resulting ``state_change`` frame.

    Validates:
    - Server does not raise or exit after bad frame.
    - Subsequent ``state_change`` event is still delivered correctly.
    """
    zmq_server, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        # Send a frame whose body is not valid JSON (raw ASCII, no braces).
        client.send_raw_frame(b"NOTJSON")

        # Give the recv callback time to process the bad frame.
        time.sleep(_RECV_SETTLE_S)

        # Server must still be alive — trigger a state transition.
        state_machine.transition_to(LumiState.LISTENING)

        msg = client.recv_frame(timeout=_RECV_TIMEOUT_S)

    assert msg["event"] == "state_change"
    assert msg["payload"]["state"] == "listening"


# ---------------------------------------------------------------------------
# Test 6 — server sends tts_start to connected client
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_server_sends_tts_start(
    ipc_stack: tuple[EventBridge, int],
) -> None:
    """``EventBridge.on_tts_start()`` sends a ``tts_start`` frame to the client.

    ``on_tts_start`` takes an ``LLMResponseReadyEvent``, packages the ``text``
    field and a zero ``duration_ms`` into a payload, and calls ``_send``.
    This test validates the full encode → TCP → decode round-trip.

    Validates:
    - ``event == "tts_start"``
    - ``payload.text == "hello"``
    - ``payload.duration_ms == 0``
    """
    zmq_server, port = ipc_stack

    with FakeTCPClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        zmq_server.on_tts_start(LLMResponseReadyEvent(text="hello"))

        msg = client.recv_frame(timeout=_RECV_TIMEOUT_S)

    assert msg["event"] == "tts_start"
    assert msg["payload"]["text"] == "hello"
    assert msg["payload"]["duration_ms"] == 0
