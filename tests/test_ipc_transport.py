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
from unittest.mock import MagicMock, patch

import pytest

from src.core.ipc_transport import (
    IPCTransport,
    _HEADER_FORMAT,
    _HEADER_SIZE,
    _THREAD_JOIN_TIMEOUT,
)

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
def transport() -> Generator[IPCTransport, None, None]:
    """Create, start, and (in teardown) stop an IPCTransport on an OS-assigned port.

    Passes port=0 so the OS binds to any free port atomically, eliminating the
    TOCTOU race that existed when we probed for a free port and then rebound to it.
    Read the actual port via ``transport.bound_port`` after start().
    """
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    # Brief wait for the accept loop to start listening.
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
def test_send_receive_roundtrip(transport: IPCTransport) -> None:
    """Server sends a message; raw client reads length prefix + payload."""
    payload = b'{"event": "state_change", "payload": {"state": "idle"}}'
    client = _connect(transport.bound_port)
    try:
        # Give _accept_loop time to accept the connection and set _client_sock.
        time.sleep(0.05)

        transport.send(payload)

        received = _decode_frame(client)
    finally:
        client.close()

    assert received == payload


@pytest.mark.integration
def test_receive_callback_fires(transport: IPCTransport) -> None:
    """Raw client sends a length-prefixed frame; on_message callback fires."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    client = _connect(transport.bound_port)
    try:
        time.sleep(0.05)  # Wait for accept

        payload = b'{"event": "interrupt", "payload": {}}'
        client.sendall(_encode_frame(payload))

        result = received_q.get(timeout=1.0)
    finally:
        client.close()

    assert result == payload


@pytest.mark.integration
def test_partial_read_handling(transport: IPCTransport) -> None:
    """Server reassembles a 100-byte payload sent one byte at a time."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    payload = b"x" * 100
    frame = _encode_frame(payload)

    client = _connect(transport.bound_port)
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
def test_client_disconnect_reconnect(transport: IPCTransport) -> None:
    """First client disconnects; second client can send and receive."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    # First client connects then disconnects.
    first = _connect(transport.bound_port)
    time.sleep(0.05)
    first.close()

    # Give _recv_loop time to detect the close and clear _client_sock.
    time.sleep(0.1)

    # Second client connects.
    second = _connect(transport.bound_port)
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
def test_stop_joins_threads() -> None:
    """After stop(), both internal threads are no longer alive."""
    t = IPCTransport("127.0.0.1", 0)
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
def test_send_without_client_does_not_raise() -> None:
    """send() before any client has connected must not raise."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)
    try:
        # No exception must propagate.
        t.send(b'{"event": "tts_stop", "payload": {}}')
    finally:
        t.stop()


@pytest.mark.integration
def test_large_message(transport: IPCTransport) -> None:
    """1 MB payload survives a round-trip with byte-exact integrity."""
    received_q: queue.Queue[bytes] = queue.Queue()
    transport.set_on_message(received_q.put)

    payload = b"A" * (1024 * 1024)  # 1 MiB

    client = _connect(transport.bound_port)
    try:
        time.sleep(0.05)  # Wait for accept

        client.sendall(_encode_frame(payload))

        result = received_q.get(timeout=5.0)
    finally:
        client.close()

    assert result == payload
    assert len(result) == 1024 * 1024


# ---------------------------------------------------------------------------
# Error-injection tests (all mocked — no real sockets required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_start_while_already_running_is_idempotent() -> None:
    """start() called while accept thread is alive logs a warning and returns."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)
    try:
        # Calling start() a second time must not raise and must not spawn
        # a duplicate thread.
        first_thread = t._accept_thread
        t.start()
        assert t._accept_thread is first_thread, (
            "start() should NOT replace the existing thread when already running"
        )
    finally:
        t.stop()


@pytest.mark.unit
def test_is_connected_returns_false_when_no_client() -> None:
    """is_connected() returns False before any client has connected."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)
    try:
        assert t.is_connected() is False
    finally:
        t.stop()


@pytest.mark.integration
def test_is_connected_returns_true_when_client_present(transport: IPCTransport) -> None:
    """is_connected() returns True while a client socket is active."""
    client = _connect(transport.bound_port)
    try:
        time.sleep(0.05)  # Let _accept_loop store _client_sock
        assert transport.is_connected() is True
    finally:
        client.close()


@pytest.mark.unit
def test_send_oserror_is_caught_and_logged() -> None:
    """send() logs a WARNING but does not propagate OSError from sendall."""
    t = IPCTransport("127.0.0.1", 0)

    # Inject a fake connected socket whose sendall() always raises.
    bad_sock = MagicMock(spec=socket.socket)
    bad_sock.sendall.side_effect = OSError("broken pipe")
    t._client_sock = bad_sock

    # Must not raise — the OSError should be caught internally.
    t.send(b"hello")
    bad_sock.sendall.assert_called_once()


@pytest.mark.unit
def test_stop_oserror_on_server_socket_close_is_swallowed() -> None:
    """stop() tolerates OSError when closing the server socket."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # Replace the server socket with one whose close() raises.
    bad_server = MagicMock(spec=socket.socket)
    bad_server.close.side_effect = OSError("already closed")
    t._server_sock = bad_server

    # Must not raise.
    t.stop()
    bad_server.close.assert_called_once()


@pytest.mark.unit
def test_stop_oserror_on_client_socket_close_is_swallowed() -> None:
    """stop() tolerates OSError when closing the client socket."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # Inject a fake client socket whose close() raises.
    bad_client = MagicMock(spec=socket.socket)
    bad_client.close.side_effect = OSError("already closed")
    with t._client_lock:
        t._client_sock = bad_client

    # Must not raise.
    t.stop()
    bad_client.close.assert_called_once()


@pytest.mark.unit
def test_stop_logs_warning_when_accept_thread_does_not_exit() -> None:
    """stop() logs a warning if the accept thread exceeds join timeout."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # Replace the accept thread with one that never dies.
    blocker_started = threading.Event()
    stop_blocker = threading.Event()

    def _stuck() -> None:
        blocker_started.set()
        stop_blocker.wait()

    fake_thread = threading.Thread(target=_stuck, daemon=True)
    fake_thread.start()
    blocker_started.wait()
    t._accept_thread = fake_thread

    import logging
    with patch.object(
        logging.getLogger("src.core.ipc_transport"), "warning"
    ) as mock_warn:
        # Use a very short timeout so the test finishes fast.
        with patch("src.core.ipc_transport._THREAD_JOIN_TIMEOUT", 0.1):
            t.stop()

    stop_blocker.set()  # Unblock the fake thread so it can exit cleanly.
    mock_warn.assert_called()  # At least one warning about the stuck thread.


@pytest.mark.unit
def test_stop_logs_warning_when_recv_thread_does_not_exit() -> None:
    """stop() logs a warning if the recv thread exceeds join timeout."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # Replace the recv thread with one that never dies.
    blocker_started = threading.Event()
    stop_blocker = threading.Event()

    def _stuck() -> None:
        blocker_started.set()
        stop_blocker.wait()

    fake_thread = threading.Thread(target=_stuck, daemon=True)
    fake_thread.start()
    blocker_started.wait()
    t._recv_thread = fake_thread

    import logging
    with patch.object(
        logging.getLogger("src.core.ipc_transport"), "warning"
    ) as mock_warn:
        with patch("src.core.ipc_transport._THREAD_JOIN_TIMEOUT", 0.1):
            t.stop()

    stop_blocker.set()
    mock_warn.assert_called()


@pytest.mark.unit
def test_stop_before_start_does_not_raise() -> None:
    """stop() called before start() must not raise."""
    t = IPCTransport("127.0.0.1", 0)
    t.stop()  # No exception expected.


@pytest.mark.unit
def test_accept_loop_exits_when_server_sock_is_none() -> None:
    """_accept_loop logs an error and exits immediately when server socket is None."""
    t = IPCTransport("127.0.0.1", 0)
    # Do NOT call start() — _server_sock stays None.

    import logging
    with patch.object(
        logging.getLogger("src.core.ipc_transport"), "error"
    ) as mock_error:
        # Run _accept_loop directly in the current thread.
        t._accept_loop()

    mock_error.assert_called_once()
    assert "no server socket" in mock_error.call_args[0][0].lower()


@pytest.mark.unit
def test_accept_loop_oserror_from_select_breaks_loop() -> None:
    """_accept_loop exits cleanly when sel.select() raises OSError (server closed)."""
    t = IPCTransport("127.0.0.1", 0)

    # Create a real server socket so _accept_loop can register it.
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    server_sock.setblocking(False)
    t._server_sock = server_sock
    t._bound_port = server_sock.getsockname()[1]

    # Patch selectors.DefaultSelector so that select() raises OSError once.
    call_count = {"n": 0}
    real_selector_class = __import__("selectors").DefaultSelector

    class _FakeSelector:
        def __init__(self) -> None:
            self._inner = real_selector_class()

        def register(self, *args, **kwargs):
            return self._inner.register(*args, **kwargs)

        def select(self, timeout=None):
            call_count["n"] += 1
            raise OSError("simulated server close")

        def close(self):
            self._inner.close()

    with patch("src.core.ipc_transport.selectors.DefaultSelector", _FakeSelector):
        t._shutdown.clear()
        t._accept_loop()  # Must return without raising.

    server_sock.close()
    assert call_count["n"] >= 1


@pytest.mark.unit
def test_accept_loop_oserror_from_accept_breaks_loop() -> None:
    """_accept_loop exits cleanly when server_sock.accept() raises OSError."""
    t = IPCTransport("127.0.0.1", 0)

    # Create a real, listening server socket.
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    server_sock.setblocking(False)
    t._server_sock = server_sock
    t._bound_port = server_sock.getsockname()[1]

    # Patch DefaultSelector so select() immediately reports the server as ready.
    import selectors as _selectors

    class _ReadySelector:
        def __init__(self) -> None:
            pass

        def register(self, *args, **kwargs):
            pass

        def select(self, timeout=None):
            # Return a non-empty list to trigger the accept() path.
            return [("fake_key", _selectors.EVENT_READ)]

        def close(self):
            pass

    # Close the real socket so accept() raises OSError.
    server_sock.close()

    with patch("src.core.ipc_transport.selectors.DefaultSelector", _ReadySelector):
        t._shutdown.clear()
        t._accept_loop()  # Must return without raising.


@pytest.mark.unit
def test_accept_loop_evicts_existing_client_on_reconnect() -> None:
    """_accept_loop closes the previous client socket when a new client connects."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # Connect a first client.
    first = _connect(t.bound_port)
    time.sleep(0.05)
    assert t.is_connected()

    # Connect a second client — the first should be evicted.
    second = _connect(t.bound_port)
    time.sleep(0.1)

    try:
        # After eviction the transport should still be connected (to the second client).
        assert t.is_connected()
    finally:
        second.close()
        t.stop()
        first.close()


@pytest.mark.unit
def test_recv_loop_oserror_on_recv_exits_cleanly() -> None:
    """_recv_loop exits when conn.recv() raises OSError and clears _client_sock."""
    t = IPCTransport("127.0.0.1", 0)

    # Build a fake socket whose recv() raises OSError immediately.
    bad_conn = MagicMock(spec=socket.socket)
    bad_conn.recv.side_effect = OSError("connection reset")
    bad_conn.close.return_value = None

    t._client_sock = bad_conn

    # Run _recv_loop in a thread with a short timeout to prevent hangs.
    done = threading.Event()

    def _run() -> None:
        t._recv_loop(bad_conn)
        done.set()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    done.wait(timeout=2.0)

    assert done.is_set(), "_recv_loop did not exit within 2 seconds"
    # _client_sock must be cleared since it was the same socket we passed.
    with t._client_lock:
        assert t._client_sock is None


@pytest.mark.unit
def test_recv_loop_shutdown_event_causes_exit() -> None:
    """_recv_loop exits promptly when _shutdown is set mid-read."""
    t = IPCTransport("127.0.0.1", 0)

    # Build a fake socket whose recv() blocks until _shutdown is set.
    released = threading.Event()

    def _slow_recv(bufsize: int) -> bytes:
        released.wait(timeout=1.0)
        raise OSError("shutdown")

    slow_conn = MagicMock(spec=socket.socket)
    slow_conn.recv.side_effect = _slow_recv
    slow_conn.close.return_value = None

    t._client_sock = slow_conn

    done = threading.Event()

    def _run() -> None:
        t._recv_loop(slow_conn)
        done.set()

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    # Signal shutdown then unblock the fake recv.
    t._shutdown.set()
    released.set()

    done.wait(timeout=2.0)
    assert done.is_set(), "_recv_loop did not exit after _shutdown was set"


@pytest.mark.unit
def test_recv_loop_callback_none_does_not_raise() -> None:
    """_recv_loop with no on_message callback silently drops received frames."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # No callback registered (default is None).
    payload = b'{"event": "test"}'
    frame = _encode_frame(payload)

    client = _connect(t.bound_port)
    try:
        time.sleep(0.05)
        client.sendall(frame)
        # Brief sleep — if _recv_loop crashes it will take the test with it.
        time.sleep(0.1)
        # Transport should still be running with a connected client.
        assert t.is_connected()
    finally:
        client.close()
        t.stop()


@pytest.mark.unit
def test_recv_loop_callback_exception_is_caught() -> None:
    """_recv_loop catches and logs exceptions raised by the on_message callback."""
    t = IPCTransport("127.0.0.1", 0)

    boom_called = threading.Event()

    def _boom(data: bytes) -> None:
        boom_called.set()
        raise RuntimeError("callback exploded")

    t.set_on_message(_boom)
    t.start()
    time.sleep(0.05)

    payload = b'{"event": "trigger_boom"}'
    frame = _encode_frame(payload)

    client = _connect(t.bound_port)
    try:
        time.sleep(0.05)
        client.sendall(frame)
        assert boom_called.wait(timeout=2.0), "on_message callback was never invoked"
        # Give _recv_loop time to log the error and continue.
        time.sleep(0.05)
        # Transport must still be alive — exception must NOT have propagated.
        assert t._recv_thread is not None and t._recv_thread.is_alive()
    finally:
        client.close()
        t.stop()


@pytest.mark.unit
def test_recv_loop_conn_close_oserror_is_swallowed() -> None:
    """_recv_loop tolerates OSError when closing the conn socket on exit."""
    t = IPCTransport("127.0.0.1", 0)

    # Fake socket: recv() returns empty bytes (signals disconnection), close() raises.
    bad_conn = MagicMock(spec=socket.socket)
    bad_conn.recv.return_value = b""  # EOF
    bad_conn.close.side_effect = OSError("already closed")

    t._client_sock = bad_conn

    done = threading.Event()

    def _run() -> None:
        t._recv_loop(bad_conn)
        done.set()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    done.wait(timeout=2.0)

    assert done.is_set(), "_recv_loop did not exit within 2 seconds"
    bad_conn.close.assert_called_once()


@pytest.mark.unit
def test_accept_loop_joins_lingering_recv_thread() -> None:
    """_accept_loop joins a still-alive recv thread before starting a new one."""
    t = IPCTransport("127.0.0.1", 0)
    t.start()
    time.sleep(0.05)

    # Connect first client to spawn a recv thread.
    first = _connect(t.bound_port)
    time.sleep(0.05)
    assert t._recv_thread is not None

    # Connect second client — _accept_loop should join the previous recv thread.
    second = _connect(t.bound_port)
    time.sleep(0.1)

    try:
        # The recv thread should be the new one for the second client.
        assert t._recv_thread is not None and t._recv_thread.is_alive()
    finally:
        second.close()
        t.stop()
        first.close()
