"""
Tests for src/core/ipc_transport.py — Wave 0, Phase 5.

All tests use real loopback TCP sockets (127.0.0.1) on an OS-assigned port
(port 0) to avoid port-conflict flakiness.  Each test is expected to complete
in well under 200ms.

Fixture strategy:
- ``free_port`` asks the OS for an available port number.
- ``transport`` creates an IPCTransport bound to that port and stops it in
  teardown via ``try/finally``, even if the test body raises.
- ``raw_client`` opens a plain socket connected to the transport's port;
  it is closed in teardown.

Frame encoding helpers are module-level functions so tests stay readable.
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from typing import Generator

import pytest

from src.core.ipc_transport import IPCTransport, _HEADER_FORMAT, _HEADER_SIZE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_frame(payload: bytes) -> bytes:
    """Prepend a 4-byte big-endian length prefix to ``payload``."""
    return struct.pack(_HEADER_FORMAT, len(payload)) + payload


def _decode_frame(sock: socket.socket) -> bytes:
    """Read exactly one length-prefixed frame from ``sock`` (blocking)."""
    raw_len = _recv_exactly(sock, _HEADER_SIZE)
    (length,) = struct.unpack(_HEADER_FORMAT, raw_len)
    return _recv_exactly(sock, length)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock``, accumulating partial reads."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"Socket closed after {len(buf)}/{n} bytes")
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
def transport(free_port: int) -> Generator[IPCTransport, None, None]:
    """Create, start, and (in teardown) stop an IPCTransport on ``free_port``."""
    t = IPCTransport("127.0.0.1", free_port)
    t.start()
    # Brief wait for the accept loop to bind and start listening.
    time.sleep(0.05)
    try:
        yield t
    finally:
        t.stop()


def _connect(port: int) -> socket.socket:
    """Open a blocking TCP connection to 127.0.0.1:port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    return sock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_send_receive_roundtrip(transport: IPCTransport, free_port: int) -> None:
    """Server sends a message; raw client reads length prefix + payload."""
    payload = b'{"event": "state_change", "payload": {"state": "idle"}}'
    client = _connect(free_port)
    try:
        # Give _accept_loop time to accept the connection and set _client_sock.
        time.sleep(0.05)

        transport.send(payload)

        received = _decode_frame(client)
    finally:
        client.close()

    assert received == payload


@pytest.mark.integration
def test_receive_callback_fires(transport: IPCTransport, free_port: int) -> None:
    """Raw client sends a length-prefixed frame; on_message callback fires."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    client = _connect(free_port)
    try:
        time.sleep(0.05)  # Wait for accept

        payload = b'{"event": "interrupt", "payload": {}}'
        client.sendall(_encode_frame(payload))

        result = received_q.get(timeout=1.0)
    finally:
        client.close()

    assert result == payload


@pytest.mark.integration
def test_partial_read_handling(transport: IPCTransport, free_port: int) -> None:
    """Server reassembles a 100-byte payload sent one byte at a time."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    payload = b"x" * 100
    frame = _encode_frame(payload)

    client = _connect(free_port)
    try:
        time.sleep(0.05)  # Wait for accept

        # Send one byte at a time to exercise the accumulator in _recv_loop.
        for byte in frame:
            client.send(bytes([byte]))
            # Tiny sleep to prevent the OS from coalescing sends into one packet.
            time.sleep(0.001)

        result = received_q.get(timeout=2.0)
    finally:
        client.close()

    assert result == payload


@pytest.mark.integration
def test_client_disconnect_reconnect(transport: IPCTransport, free_port: int) -> None:
    """First client disconnects; second client can send and receive."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    # First client connects then disconnects.
    first = _connect(free_port)
    time.sleep(0.05)
    first.close()

    # Give _recv_loop time to detect the close and clear _client_sock.
    time.sleep(0.1)

    # Second client connects.
    second = _connect(free_port)
    try:
        time.sleep(0.05)  # Wait for accept

        payload = b'{"event": "user_text", "payload": {"text": "hello"}}'
        second.sendall(_encode_frame(payload))
        result = received_q.get(timeout=1.0)
        assert result == payload

        # Verify server can also send back to the second client.
        response = b'{"event": "state_change", "payload": {"state": "processing"}}'
        transport.send(response)
        received = _decode_frame(second)
        assert received == response
    finally:
        second.close()


@pytest.mark.integration
def test_stop_joins_threads(free_port: int) -> None:
    """After stop(), both internal threads are no longer alive."""
    t = IPCTransport("127.0.0.1", free_port)
    t.start()
    time.sleep(0.05)

    accept_thread = t._accept_thread
    recv_thread = t._recv_thread  # None until a client connects

    t.stop()

    if accept_thread is not None:
        assert not accept_thread.is_alive(), "accept thread still alive after stop()"
    # recv_thread is None (no client connected), which is also valid.
    if recv_thread is not None:
        assert not recv_thread.is_alive(), "recv thread still alive after stop()"


@pytest.mark.integration
def test_send_without_client_does_not_raise(free_port: int) -> None:
    """send() before any client has connected must not raise."""
    t = IPCTransport("127.0.0.1", free_port)
    t.start()
    time.sleep(0.05)
    try:
        # No exception must propagate.
        t.send(b'{"event": "tts_stop", "payload": {}}')
    finally:
        t.stop()


@pytest.mark.integration
def test_large_message(transport: IPCTransport, free_port: int) -> None:
    """1 MB payload survives a round-trip with byte-exact integrity."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    payload = b"A" * (1024 * 1024)  # 1 MiB

    client = _connect(free_port)
    try:
        time.sleep(0.05)  # Wait for accept

        client.sendall(_encode_frame(payload))

        result = received_q.get(timeout=5.0)
    finally:
        client.close()

    assert result == payload
    assert len(result) == 1024 * 1024
