"""
IPC protocol conformance test suite — Phase 5 Wave 4.

These are end-to-end integration tests that verify the full Python IPC stack
using a fake TCP client acting as the Godot frontend.

Stack under test (no mocking of the transport layer):
    FakeGodotClient  <--TCP loopback-->  IPCTransport  <-->  ZMQServer
                                                               |
                                                        StateMachine / queue.Queue

Wire format:
    4-byte big-endian uint32 length prefix + UTF-8 JSON body.

JSON envelope (outbound):
    {"event": str, "payload": dict, "timestamp": float, "version": "1.0"}

Fixture strategy:
- ``free_port``   — OS-assigned port via bind-to-0 trick.
- ``ipc_stack``   — starts IPCTransport + ZMQServer; tears down in finally.
- ``FakeGodotClient`` — context manager that connects, sends, and receives
                        length-prefixed JSON frames.

All recv_message() calls have an explicit timeout to prevent infinite hangs.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import time
import threading
from contextlib import contextmanager
from typing import Any, Generator

import pytest

from src.core.config import IPCConfig
from src.core.events import (
    InterruptEvent,
    UserTextEvent,
    VisemeEvent,
)
from src.core.ipc_transport import IPCTransport, _HEADER_FORMAT, _HEADER_SIZE
from src.core.state_machine import LumiState, StateMachine
from src.core.zmq_server import ZMQServer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECT_SETTLE_S: float = 0.08   # wait for accept loop to notice a new client
_STOP_SETTLE_S: float = 0.10      # wait for recv loop to detect a close
_DEFAULT_RECV_TIMEOUT: float = 2.0


# ---------------------------------------------------------------------------
# FakeGodotClient
# ---------------------------------------------------------------------------


class FakeGodotClient:
    """Minimal TCP client that speaks the Lumi IPC wire protocol.

    Wire format: 4-byte big-endian uint32 length prefix + UTF-8 JSON body.

    The JSON envelope expected by ZMQServer._decode:
        {
            "event":     str,
            "payload":   dict,
            "timestamp": float,
            "version":   "1.0"
        }

    Usage:
        client = FakeGodotClient(port)
        client.connect()
        client.send_message("interrupt", {})
        msg = client.recv_message(timeout=2.0)
        client.close()

    Or as a context manager (auto-closes):
        with FakeGodotClient(port) as client:
            ...
    """

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        """Open a blocking TCP connection to host:port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._host, self._port))
        self._sock = sock

    def send_message(self, event: str, payload: dict[str, Any]) -> None:
        """Encode and send a length-prefixed JSON frame.

        Wraps the event+payload in the full ZMQServer wire envelope so that
        ZMQServer._decode() accepts the message.

        Args:
            event:   Wire event name (e.g. "interrupt", "user_text").
            payload: Arbitrary JSON-serialisable dict.
        """
        if self._sock is None:
            raise RuntimeError("FakeGodotClient: not connected")

        envelope = {
            "event": event,
            "payload": payload,
            "timestamp": time.time(),
            "version": "1.0",
        }
        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        header = struct.pack(_HEADER_FORMAT, len(body))
        self._sock.sendall(header + body)

    def send_raw(self, raw_body: bytes) -> None:
        """Send a length-prefixed frame whose body is arbitrary raw bytes.

        Used to exercise malformed-message handling in the server.

        Args:
            raw_body: The bytes to send as the frame body (may be invalid JSON).
        """
        if self._sock is None:
            raise RuntimeError("FakeGodotClient: not connected")

        header = struct.pack(_HEADER_FORMAT, len(raw_body))
        self._sock.sendall(header + raw_body)

    def recv_message(self, timeout: float = _DEFAULT_RECV_TIMEOUT) -> dict[str, Any]:
        """Read one length-prefixed frame and return the decoded JSON dict.

        Args:
            timeout: Maximum seconds to wait for a complete frame.

        Returns:
            The decoded JSON object as a Python dict.

        Raises:
            TimeoutError:    If no complete frame arrives within ``timeout``.
            ConnectionError: If the socket is closed mid-read.
            json.JSONDecodeError: If the frame body is not valid JSON.
        """
        if self._sock is None:
            raise RuntimeError("FakeGodotClient: not connected")

        self._sock.settimeout(timeout)
        try:
            raw_len = self._recv_exactly(_HEADER_SIZE)
            (payload_len,) = struct.unpack(_HEADER_FORMAT, raw_len)
            body = self._recv_exactly(payload_len)
        except socket.timeout as exc:
            raise TimeoutError(
                f"FakeGodotClient: no frame received within {timeout}s"
            ) from exc
        finally:
            self._sock.settimeout(None)  # restore blocking mode

        return json.loads(body.decode("utf-8"))

    def close(self) -> None:
        """Close the TCP connection; safe to call multiple times."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FakeGodotClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recv_exactly(self, n: int) -> bytes:
        """Read exactly ``n`` bytes, accumulating partial reads."""
        buf = bytearray()
        while len(buf) < n:
            assert self._sock is not None
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(
                    f"FakeGodotClient: socket closed after {len(buf)}/{n} bytes"
                )
            buf.extend(chunk)
        return bytes(buf)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def free_port() -> int:
    """Ask the OS for an available TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    free_port: int,
    event_queue: queue.Queue[Any],
    state_machine: StateMachine,
) -> Generator[tuple[ZMQServer, int], None, None]:
    """Start a real IPCTransport + ZMQServer on a free port.

    Yields ``(zmq_server, port)``.  Tears down both in a ``finally`` block
    so sockets are always closed even when the test body raises.
    """
    config = IPCConfig(address="tcp://127.0.0.1", port=free_port)
    server = ZMQServer(
        config=config,
        event_queue=event_queue,
        state_machine=state_machine,
    )
    try:
        server.start()
        # Let the accept loop bind and start listening before tests connect.
        time.sleep(0.05)
        yield server, free_port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Test 1 — full state lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_state_lifecycle(
    ipc_stack: tuple[ZMQServer, int],
    state_machine: StateMachine,
) -> None:
    """Connect a fake client, trigger IDLE→LISTENING, assert state_change frame.

    The ZMQServer registers itself as a StateMachine observer at construction
    time.  When transition_to(LISTENING) fires, the observer calls _send()
    which writes a length-prefixed frame to the connected client.
    """
    _, port = ipc_stack

    with FakeGodotClient(port) as client:
        # Wait for the accept loop to register the new connection.
        time.sleep(_CONNECT_SETTLE_S)

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
    ipc_stack: tuple[ZMQServer, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends an 'interrupt' frame; ZMQServer posts InterruptEvent.

    The ZMQServer._on_raw_message callback fires on IPCTransport's recv
    daemon thread and puts an InterruptEvent(source='zmq') onto the queue.
    """
    _, port = ipc_stack

    with FakeGodotClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.send_message("interrupt", {})

        event = event_queue.get(timeout=2.0)

    assert isinstance(event, InterruptEvent)
    assert event.source == "zmq"


# ---------------------------------------------------------------------------
# Test 3 — user_text from client posts UserTextEvent to queue
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_user_text_triggers_event(
    ipc_stack: tuple[ZMQServer, int],
    event_queue: queue.Queue[Any],
) -> None:
    """Client sends a 'user_text' frame; ZMQServer posts UserTextEvent.

    The posted event must carry the exact 'text' value from the payload.
    """
    _, port = ipc_stack

    with FakeGodotClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.send_message("user_text", {"text": "hello"})

        event = event_queue.get(timeout=2.0)

    assert isinstance(event, UserTextEvent)
    assert event.text == "hello"


# ---------------------------------------------------------------------------
# Test 4 — viseme forwarding to client
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_viseme_forwarding(
    ipc_stack: tuple[ZMQServer, int],
) -> None:
    """ZMQServer.on_tts_viseme() sends a tts_viseme frame to the connected client.

    Verifies that viseme phoneme and duration_ms survive the encode→TCP→decode
    round-trip exactly.
    """
    zmq_server, port = ipc_stack

    with FakeGodotClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)

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
    ipc_stack: tuple[ZMQServer, int],
    state_machine: StateMachine,
) -> None:
    """A valid length-prefixed frame carrying invalid JSON must not crash the server.

    After the malformed message the server must remain operational: a subsequent
    state transition must still produce a forwarded state_change frame.
    """
    zmq_server, port = ipc_stack

    with FakeGodotClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)

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
    ipc_stack: tuple[ZMQServer, int],
    state_machine: StateMachine,
) -> None:
    """After the first client closes, a second client must receive subsequent events.

    IPCTransport implements a single-client model: when a new connection arrives
    (or is accepted after the previous one dropped), it becomes the active client.
    ZMQServer.on_state_change() must deliver the frame to the new client.
    """
    _, port = ipc_stack

    # First client connects then disconnects.
    first = FakeGodotClient(port)
    first.connect()
    time.sleep(_CONNECT_SETTLE_S)
    first.close()

    # Give _recv_loop time to detect the closure and clear _client_sock so
    # the accept loop is ready for the next connection.
    time.sleep(_STOP_SETTLE_S)

    # Second client connects.
    with FakeGodotClient(port) as second:
        time.sleep(_CONNECT_SETTLE_S)

        state_machine.transition_to(LumiState.LISTENING)

        msg = second.recv_message(timeout=2.0)

    assert msg["event"] == "state_change"
    assert msg["payload"]["state"] == "listening"
