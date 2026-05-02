"""
Minimal synchronous WebSocket client that mimics the Tauri/React frontend.

Wire format: one UTF-8 JSON string per WebSocket message (no length prefix).

JSON envelope (inbound to EventBridge / outbound from EventBridge):
    {
        "event":     str,
        "payload":   dict,
        "timestamp": float,
        "version":   "1.0"
    }

Usage
─────
    client = FakeWSClient(port=54321)
    client.connect()
    client.do_handshake()
    client.send_frame("interrupt", {})
    msg = client.recv_frame(timeout=2.0)
    assert msg["event"] == "state_change"
    client.close()

    # Or as a context manager:
    with FakeWSClient(port) as client:
        client.do_handshake()
        client.send_frame("user_text", {"text": "hello"})
        frame = client.recv_frame()

Notes
─────
- Each FakeWSClient instance owns its own asyncio event loop that is created
  on construction and closed on close().  All async operations are driven via
  loop.run_until_complete() so the public API is fully synchronous.
- recv_frame() uses asyncio.wait_for() to enforce the timeout.
- send_raw_frame() sends bytes as a WebSocket binary frame; the server's
  on_message callback receives it as bytes and passes it to EventBridge for
  validation (exercising malformed-message handling).
- Backward-compat aliases (send_message, recv_message, send_raw) are provided
  so test_ipc_protocol_conformance.py can use this client without renaming
  every method call.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import websockets.asyncio.client as ws_client


class FakeWSClient:
    """Minimal synchronous WebSocket client speaking the Lumi IPC wire protocol.

    Args:
        port: WebSocket port of the server to connect to.
        host: Hostname or IP address (default ``"127.0.0.1"``).
    """

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._host = host
        self._port = port
        self._uri = f"ws://{host}:{port}"
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ws: Any | None = None  # websockets ClientConnection

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a WebSocket connection to ``host:port``."""
        self._ws = self._loop.run_until_complete(
            ws_client.connect(self._uri)
        )

    def close(self) -> None:
        """Close the WebSocket connection; safe to call multiple times."""
        if self._ws is not None:
            try:
                self._loop.run_until_complete(self._ws.close())
            except Exception:
                pass
            self._ws = None
        try:
            self._loop.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FakeWSClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public send / receive
    # ------------------------------------------------------------------

    def send_frame(self, event: str, payload: dict[str, Any]) -> None:
        """Encode and send a JSON frame as a WebSocket text message.

        Wraps ``event`` and ``payload`` in the full EventBridge wire envelope
        so that ``EventBridge._decode()`` accepts it.

        Args:
            event:   Wire event name (e.g. ``"interrupt"``, ``"user_text"``).
            payload: Arbitrary JSON-serialisable dict.

        Raises:
            RuntimeError: If ``connect()`` has not been called yet.
        """
        if self._ws is None:
            raise RuntimeError("FakeWSClient: not connected — call connect() first")

        envelope: dict[str, Any] = {
            "event": event,
            "payload": payload,
            "timestamp": time.time(),
            "version": "1.0",
        }
        body = json.dumps(envelope, ensure_ascii=False)
        self._loop.run_until_complete(self._ws.send(body))

    def send_raw_frame(self, raw_body: bytes) -> None:
        """Send arbitrary bytes as a WebSocket binary message.

        Useful for exercising malformed-message handling in the server.

        Args:
            raw_body: Frame body bytes (may be invalid JSON or binary data).

        Raises:
            RuntimeError: If ``connect()`` has not been called yet.
        """
        if self._ws is None:
            raise RuntimeError("FakeWSClient: not connected — call connect() first")

        self._loop.run_until_complete(self._ws.send(raw_body))

    def do_handshake(self, timeout: float = 1.0) -> None:
        """Complete the hello/hello_ack handshake after connecting.

        HandshakeHandler sends a ``hello`` frame immediately on connect.
        This reads it and replies with ``hello_ack`` so normal messages
        are forwarded downstream.

        Args:
            timeout: Seconds to wait for the hello frame.
        """
        async def _do() -> None:
            assert self._ws is not None
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            obj = json.loads(text)
            assert obj.get("type") == "hello", f"expected hello frame, got {obj!r}"
            ack = json.dumps({"type": "hello_ack", "version": "1.0", "status": "ok"})
            await self._ws.send(ack)

        self._loop.run_until_complete(_do())

    def recv_frame(self, timeout: float = 2.0) -> dict[str, Any]:
        """Read one WebSocket message and return the decoded JSON dict.

        Args:
            timeout: Maximum seconds to wait for a complete message.

        Returns:
            Decoded JSON object as a Python ``dict``.

        Raises:
            RuntimeError:        If ``connect()`` has not been called yet.
            TimeoutError:        If no message arrives within ``timeout``.
            json.JSONDecodeError: If the message body is not valid JSON.
        """
        if self._ws is None:
            raise RuntimeError("FakeWSClient: not connected — call connect() first")

        async def _do() -> dict[str, Any]:
            assert self._ws is not None
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            return json.loads(text)  # type: ignore[no-any-return]

        try:
            return self._loop.run_until_complete(_do())
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"FakeWSClient: no frame received within {timeout}s"
            ) from exc

    # ------------------------------------------------------------------
    # Backward-compat aliases (for test_ipc_protocol_conformance.py)
    # ------------------------------------------------------------------

    def send_message(self, event: str, payload: dict[str, Any]) -> None:
        """Alias for send_frame()."""
        self.send_frame(event, payload)

    def recv_message(self, timeout: float = 2.0) -> dict[str, Any]:
        """Alias for recv_frame()."""
        return self.recv_frame(timeout=timeout)

    def send_raw(self, raw_body: bytes) -> None:
        """Alias for send_raw_frame()."""
        self.send_raw_frame(raw_body)
