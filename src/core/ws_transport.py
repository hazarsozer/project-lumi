"""
WebSocket transport server for Project Lumi.

Drop-in replacement for IPCTransport — provides the same synchronous
public API (start / stop / send / set_on_message / set_on_connect /
is_connected / bound_port) while serving WebSocket clients instead of
raw TCP.

An asyncio event loop runs in a dedicated daemon thread.  All public
methods are synchronous and thread-safe; they communicate with the async
loop via threading.Event (ready gate), asyncio.Event (shutdown signal),
and asyncio.run_coroutine_threadsafe (outbound sends).

Wire format: one UTF-8 JSON string per WebSocket message.  The WebSocket
framing protocol handles message boundaries — no length prefix is needed.

Design notes
────────────
- start() blocks until the server socket is bound (mirrors IPCTransport's
  synchronous bind-before-return contract).
- Single-client policy: a second WebSocket connection is rejected with
  close code 1008 (policy violation).
- All magic numbers are named constants.
- No print() — all output via logging.getLogger(__name__).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

import websockets
from websockets.asyncio.server import ServerConnection

logger = logging.getLogger(__name__)

_START_TIMEOUT_S: float = 5.0
_STOP_TIMEOUT_S: float = 2.0


class WSTransport:
    """Single-client WebSocket server with the same API as IPCTransport.

    Thread-safety contract
    ──────────────────────
    - start() and stop() must be called from the same thread (usually the
      main/orchestrator thread).
    - send() is safe to call from any thread after start() returns.
    - set_on_message() and set_on_connect() must be called before start().
    - _ws_client is set and cleared from the asyncio thread; CPython's GIL
      makes simple attribute reads from the main thread safe without an
      additional lock.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._bound_port: int | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready: threading.Event = threading.Event()

        # Created inside _serve() — must live on the asyncio thread.
        self._shutdown: asyncio.Event | None = None
        self._ws_lock: asyncio.Lock | None = None

        # Current connected client; set/cleared from the asyncio thread.
        self._ws_client: ServerConnection | None = None

        self._on_message: Callable[[bytes], None] | None = None
        self._on_connect: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Public API (mirrors IPCTransport)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket server on a dedicated daemon thread.

        Blocks until the server socket is bound (or _START_TIMEOUT_S elapses).
        Idempotent: warns and returns if already running.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WSTransport.start() called while already running.")
            return

        self._ready.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ws-transport",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=_START_TIMEOUT_S):
            logger.warning(
                "WSTransport: server did not bind on %s:%d within %.1fs",
                self._host,
                self._port,
                _START_TIMEOUT_S,
            )

    def stop(self) -> None:
        """Signal the asyncio loop to stop and join the daemon thread.

        Safe to call before start() or after a previous stop().
        """
        loop = self._loop
        shutdown = self._shutdown

        if loop is not None and shutdown is not None:
            loop.call_soon_threadsafe(shutdown.set)

        if self._thread is not None:
            self._thread.join(timeout=_STOP_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "ws-transport thread did not exit within %.1fs", _STOP_TIMEOUT_S
                )
            self._thread = None

        self._loop = None
        self._bound_port = None
        logger.info("WSTransport stopped.")

    def send(self, message: bytes) -> None:
        """Send a frame to the connected client. Thread-safe.

        Drops silently (logs DEBUG) if no client is connected or the loop
        is not running.

        Args:
            message: Raw UTF-8 bytes to deliver as a WebSocket text message.
        """
        loop = self._loop
        ws = self._ws_client

        if loop is None or not loop.is_running():
            logger.debug("WSTransport.send(): loop not running; dropping.")
            return

        if ws is None:
            logger.debug("WSTransport.send(): no client connected; dropping.")
            return

        async def _do_send() -> None:
            try:
                await ws.send(message)
            except Exception as exc:
                logger.warning("WSTransport.send() failed (client gone?): %s", exc)

        asyncio.run_coroutine_threadsafe(_do_send(), loop)

    def set_on_message(self, callback: Callable[[bytes], None]) -> None:
        """Register the callback invoked when a complete frame is received.

        Args:
            callback: Called with raw payload bytes for every inbound message.
                      Invoked from the asyncio thread — must not block.
        """
        self._on_message = callback

    def set_on_connect(self, callback: Callable[[], None]) -> None:
        """Register the callback invoked when a new client connects.

        Args:
            callback: Called (no arguments) from the asyncio thread immediately
                      after the client socket is registered.
        """
        self._on_connect = callback

    def is_connected(self) -> bool:
        """Return True if a client WebSocket is currently active."""
        return self._ws_client is not None

    @property
    def bound_port(self) -> int | None:
        """Return the actual port the server is bound to, or None if not started.

        Useful when port=0 is passed (OS-assigned): call start(), then read
        bound_port to discover the assigned port.
        """
        return self._bound_port

    # ------------------------------------------------------------------
    # Private — asyncio thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Target for the daemon thread: owns and drives the asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("WSTransport loop exited with error: %s", exc)
        finally:
            # Ensure start() unblocks even if _serve raised before _ready.set().
            self._ready.set()

    async def _serve(self) -> None:
        """Bind the WebSocket server and wait for the shutdown signal."""
        self._shutdown = asyncio.Event()
        self._ws_lock = asyncio.Lock()

        try:
            async with websockets.serve(
                self._handle_ws, self._host, self._port
            ) as server:
                port = server.sockets[0].getsockname()[1]
                self._bound_port = port
                logger.info(
                    "WSTransport listening on ws://%s:%d", self._host, port
                )
                self._ready.set()
                await self._shutdown.wait()
        except Exception as exc:
            logger.error(
                "WSTransport failed to bind on %s:%d: %s",
                self._host,
                self._port,
                exc,
            )
            self._ready.set()

        logger.debug("WSTransport _serve exited.")

    async def _handle_ws(self, ws: ServerConnection) -> None:
        """Coroutine invoked by websockets for each new client connection."""
        assert self._ws_lock is not None

        async with self._ws_lock:
            if self._ws_client is not None:
                logger.warning("Second WS client rejected; only one allowed.")
                await ws.close(1008, "Only one client allowed")
                return
            self._ws_client = ws

        logger.info("WS client connected: %s", ws.remote_address)

        on_connect = self._on_connect
        if on_connect is not None:
            try:
                on_connect()
            except Exception as exc:
                logger.warning("on_connect callback raised: %s", exc)

        try:
            async for raw in ws:
                payload = raw if isinstance(raw, bytes) else raw.encode()
                callback = self._on_message
                if callback is not None:
                    try:
                        callback(payload)
                    except Exception as exc:
                        logger.error("on_message callback raised: %s", exc)
        except Exception as exc:
            logger.info("WS client disconnected: %s", exc)
        finally:
            assert self._ws_lock is not None
            async with self._ws_lock:
                if self._ws_client is ws:
                    self._ws_client = None
            logger.info("WS client disconnected; ready for new connection.")
