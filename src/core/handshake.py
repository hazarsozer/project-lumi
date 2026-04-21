"""
IPC hello/hello_ack handshake handler — Wave F6.

Protocol:
  1. Brain sends: {"type": "hello", "version": "1.0", "capabilities": [...]}
  2. Godot replies: {"type": "hello_ack", "version": "1.0", "status": "ok"}
     — or {"type": "hello_ack", "version": "X.Y", "status": "version_mismatch"}
  3. status != "ok" → Brain logs warning, stays connected (degrade gracefully).
  4. No hello_ack within HANDSHAKE_TIMEOUT_S → Brain logs warning, continues.

Design:
- HandshakeHandler wraps an IPCTransport (via duck typing — any object with
  .send(bytes) works).
- on_client_connected() sends the hello frame and arms a one-shot timeout Timer.
- on_message_received() inspects each incoming frame:
    * If it is a valid hello_ack → consume it, cancel the timer, mark complete.
    * Otherwise → forward to the downstream callback unchanged.
- All public methods are thread-safe (protected by a single lock).

Constraints:
- No asyncio — threading.Timer only.
- No print() — all output via logging.getLogger(__name__).
- stdlib only: json, logging, threading.
- All magic numbers are named constants.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

HELLO_VERSION: str = "1.0"
HELLO_CAPABILITIES: list[str] = ["tts", "rag", "tools"]
HANDSHAKE_TIMEOUT_S: float = 3.0   # Seconds to wait for hello_ack before warning


# ---------------------------------------------------------------------------
# HandshakeHandler
# ---------------------------------------------------------------------------


class HandshakeHandler:
    """Sends a hello frame when a client connects and waits for hello_ack.

    Usage::

        handler = HandshakeHandler(transport)
        handler.set_downstream_callback(my_on_message)

        # Wire into transport:
        transport.set_on_connect(handler.on_client_connected)
        transport.set_on_message(handler.on_message_received)

    The downstream callback receives all frames that are NOT hello_ack.

    Thread-safety:
        All public methods acquire ``_lock`` before mutating state.
    """

    def __init__(self, transport: object) -> None:
        """
        Args:
            transport: Any object that has a ``.send(bytes)`` method.
                       Typically an IPCTransport instance.
        """
        self._transport = transport
        self._downstream: Callable[[bytes], None] | None = None
        self._lock: threading.Lock = threading.Lock()

        # Handshake state — modified only under _lock.
        self._handshake_pending: bool = False
        self._handshake_done: bool = False
        self._timeout_timer: threading.Timer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_downstream_callback(self, callback: Callable[[bytes], None]) -> None:
        """Register the callback that receives non-handshake messages.

        Args:
            callback: Called with the raw payload bytes for every frame that
                      is not a hello_ack.
        """
        with self._lock:
            self._downstream = callback

    def on_client_connected(self) -> None:
        """Call when a new client connects.

        Sends the hello frame and arms the timeout timer.
        Safe to call from any thread.
        """
        with self._lock:
            # Reset per-connection state.
            self._handshake_done = False
            self._handshake_pending = True

            # Cancel any lingering timer from a previous connection.
            if self._timeout_timer is not None:
                self._timeout_timer.cancel()
                self._timeout_timer = None

            hello = self._build_hello()
            self._transport.send(hello)
            logger.debug("Sent hello frame to client.")

            timer = threading.Timer(HANDSHAKE_TIMEOUT_S, self._on_timeout)
            timer.daemon = True
            timer.start()
            self._timeout_timer = timer

    def on_message_received(self, raw: bytes) -> None:
        """Call for every raw incoming frame from the client.

        Inspects the frame:
        - If it is a valid hello_ack and handshake is pending → consume it.
        - Otherwise → forward to the downstream callback.

        Safe to call from any thread.

        Args:
            raw: Raw payload bytes (length prefix already stripped by transport).
        """
        with self._lock:
            if self._handshake_pending and not self._handshake_done:
                ack = self._try_parse_hello_ack(raw)
                if ack is not None:
                    self._consume_hello_ack(ack)
                    return
                # Not a hello_ack — fall through to downstream.

            downstream = self._downstream

        if downstream is not None:
            downstream(raw)

    def is_handshake_complete(self) -> bool:
        """Return True when the handshake has been resolved (ok, mismatch, or timeout)."""
        with self._lock:
            return self._handshake_done

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_hello(self) -> bytes:
        """Build the hello frame as UTF-8 JSON bytes (no length prefix)."""
        msg = {
            "type": "hello",
            "version": HELLO_VERSION,
            "capabilities": list(HELLO_CAPABILITIES),
        }
        return json.dumps(msg, ensure_ascii=False).encode("utf-8")

    def _try_parse_hello_ack(self, raw: bytes) -> dict | None:
        """Attempt to parse ``raw`` as a hello_ack frame.

        Returns the parsed dict if it is a hello_ack, otherwise None.
        Does not raise — JSON errors return None.
        """
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        if not isinstance(obj, dict):
            return None

        if obj.get("type") != "hello_ack":
            return None

        return obj

    def _consume_hello_ack(self, ack: dict) -> None:
        """Process a parsed hello_ack dict.

        Must be called with ``_lock`` held.

        Cancels the timeout timer and marks the handshake as done.
        Logs a warning if the status is not "ok".
        """
        # Cancel the timeout timer — ack arrived in time.
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

        self._handshake_pending = False
        self._handshake_done = True

        status = ack.get("status", "")
        remote_version = ack.get("version", "<unknown>")

        if status != "ok":
            logger.warning(
                "IPC handshake version mismatch: remote=%s status=%s; "
                "continuing with degraded compatibility.",
                remote_version,
                status,
            )
        else:
            logger.debug("IPC handshake complete (ok, remote=%s).", remote_version)

    def _on_timeout(self) -> None:
        """Called by the Timer when no hello_ack is received within the deadline."""
        with self._lock:
            if self._handshake_done:
                # Ack arrived concurrently with the timer firing — ignore.
                return

            self._handshake_pending = False
            self._handshake_done = True
            self._timeout_timer = None

        logger.warning(
            "IPC handshake timeout: no hello_ack received within %.1fs; "
            "continuing without handshake.",
            HANDSHAKE_TIMEOUT_S,
        )
