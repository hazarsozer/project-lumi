"""
Minimal Python TCP client that mimics the Tauri/React frontend IPC client.

Wire format: 4-byte big-endian uint32 length prefix + UTF-8 JSON body.

JSON envelope (inbound to EventBridge / outbound from EventBridge):
    {
        "event":     str,
        "payload":   dict,
        "timestamp": float,
        "version":   "1.0"
    }

Usage
-----
    client = FakeTCPClient(port=54321)
    client.connect()
    client.send_frame("interrupt", {})
    msg = client.recv_frame(timeout=2.0)
    assert msg["event"] == "state_change"
    client.close()

    # Or as a context manager:
    with FakeTCPClient(port) as client:
        client.send_frame("user_text", {"text": "hello"})
        frame = client.recv_frame()

Notes
-----
- ``recv_frame`` sets a socket-level timeout to avoid hanging forever; the
  timeout is restored to blocking mode after each call.
- ``send_raw_frame`` lets tests inject arbitrary bytes as the frame body to
  exercise malformed-message handling in the server.
- All socket operations use stdlib ``socket`` only — no third-party deps.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any

# Matches IPCTransport's wire constants — import separately to avoid pulling
# in the full server module during test collection.
_HEADER_FORMAT: str = "!I"   # network byte order, unsigned 32-bit int
_HEADER_SIZE: int = 4


class FakeTCPClient:
    """Minimal synchronous TCP client speaking the Lumi IPC wire protocol.

    Args:
        port: TCP port of the server to connect to.
        host: Hostname or IP address (default ``"127.0.0.1"``).
    """

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a blocking TCP connection to ``host:port``."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._host, self._port))
        self._sock = sock

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

    def __enter__(self) -> "FakeTCPClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public send / receive
    # ------------------------------------------------------------------

    def send_frame(self, event: str, payload: dict[str, Any]) -> None:
        """Encode and send a length-prefixed JSON frame.

        Wraps ``event`` and ``payload`` in the full EventBridge wire envelope so
        that ``EventBridge._decode()`` accepts it.

        Args:
            event:   Wire event name (e.g. ``"interrupt"``, ``"user_text"``).
            payload: Arbitrary JSON-serialisable dict.

        Raises:
            RuntimeError: If ``connect()`` has not been called yet.
        """
        if self._sock is None:
            raise RuntimeError("FakeTCPClient: not connected — call connect() first")

        envelope: dict[str, Any] = {
            "event": event,
            "payload": payload,
            "timestamp": time.time(),
            "version": "1.0",
        }
        body: bytes = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        header: bytes = struct.pack(_HEADER_FORMAT, len(body))
        self._sock.sendall(header + body)

    def send_raw_frame(self, raw_body: bytes) -> None:
        """Send a length-prefixed frame whose body is arbitrary raw bytes.

        Useful for exercising malformed-message handling in the server.

        Args:
            raw_body: Frame body bytes (may be invalid JSON or binary data).

        Raises:
            RuntimeError: If ``connect()`` has not been called yet.
        """
        if self._sock is None:
            raise RuntimeError("FakeTCPClient: not connected — call connect() first")

        header: bytes = struct.pack(_HEADER_FORMAT, len(raw_body))
        self._sock.sendall(header + raw_body)

    def do_handshake(self, timeout: float = 1.0) -> None:
        """Complete the hello/hello_ack handshake after connecting.

        HandshakeHandler sends a ``hello`` frame immediately on connect.
        This reads it and replies with ``hello_ack`` so normal messages
        are forwarded downstream.

        Args:
            timeout: Seconds to wait for the hello frame.
        """
        hello = self.recv_frame(timeout=timeout)
        assert hello.get("type") == "hello", f"expected hello frame, got {hello!r}"
        ack: bytes = json.dumps(
            {"type": "hello_ack", "version": "1.0", "status": "ok"}
        ).encode("utf-8")
        header: bytes = struct.pack(_HEADER_FORMAT, len(ack))
        assert self._sock is not None
        self._sock.sendall(header + ack)

    def recv_frame(self, timeout: float = 2.0) -> dict[str, Any]:
        """Read one length-prefixed frame and return the decoded JSON dict.

        Temporarily sets a socket-level timeout to ``timeout`` seconds, then
        restores blocking mode after the read completes or fails.

        Args:
            timeout: Maximum seconds to wait for a complete frame.

        Returns:
            Decoded JSON object as a Python ``dict``.

        Raises:
            RuntimeError:        If ``connect()`` has not been called yet.
            TimeoutError:        If no complete frame arrives within ``timeout``.
            ConnectionError:     If the socket closes mid-read.
            json.JSONDecodeError: If the frame body is not valid JSON.
        """
        if self._sock is None:
            raise RuntimeError("FakeTCPClient: not connected — call connect() first")

        self._sock.settimeout(timeout)
        try:
            raw_len: bytes = self._recv_exactly(_HEADER_SIZE)
            (payload_len,) = struct.unpack(_HEADER_FORMAT, raw_len)
            body: bytes = self._recv_exactly(payload_len)
        except socket.timeout as exc:
            raise TimeoutError(
                f"FakeTCPClient: no frame received within {timeout}s"
            ) from exc
        finally:
            self._sock.settimeout(None)  # restore blocking mode

        return json.loads(body.decode("utf-8"))  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recv_exactly(self, n: int) -> bytes:
        """Read exactly ``n`` bytes, accumulating across partial ``recv()`` calls.

        Args:
            n: Exact number of bytes to read.

        Returns:
            Exactly ``n`` bytes.

        Raises:
            ConnectionError: If the socket is closed before ``n`` bytes arrive.
        """
        buf = bytearray()
        while len(buf) < n:
            assert self._sock is not None
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(
                    f"FakeTCPClient: socket closed after {len(buf)}/{n} bytes"
                )
            buf.extend(chunk)
        return bytes(buf)
