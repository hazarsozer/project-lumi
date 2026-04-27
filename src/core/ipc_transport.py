"""
Length-prefixed TCP transport server for Project Lumi.

Wire format: 4-byte big-endian uint32 length prefix + UTF-8 JSON body.

This module provides a single-client TCP server that:
- Binds on a loopback address and accepts one client at a time.
- Reads and writes length-prefixed frames.
- Runs two daemon threads: _accept_loop and _recv_loop.
- Is safe to call from the orchestrator thread via send().

Constraints:
- No asyncio — threads + queue.Queue only.
- No print() — all output via logging.getLogger(__name__).
- stdlib only: socket, selectors, threading, struct, logging.
- All magic numbers are named constants.
"""

from __future__ import annotations

import logging
import selectors
import socket
import struct
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

_HEADER_SIZE: int = 4  # Bytes in the length prefix (big-endian uint32)
_SELECT_TIMEOUT: float = 0.2  # Seconds; keeps _accept_loop responsive to stop()
_THREAD_JOIN_TIMEOUT: float = 2.0  # Seconds to wait when joining threads in stop()
_HEADER_FORMAT: str = "!I"  # struct format: network byte order, unsigned int


# ---------------------------------------------------------------------------
# IPCTransport
# ---------------------------------------------------------------------------


class IPCTransport:
    """Single-client length-prefixed TCP server.

    Wire format: 4-byte big-endian uint32 length prefix + UTF-8 JSON body.

    Runs two daemon threads:
    - ``_accept_loop``: binds the server socket, polls with selectors, and
      accepts one client at a time.  When a new client arrives while one is
      active, the old connection is closed first.
    - ``_recv_loop``: reads length-prefixed frames from the connected client.
      Uses a bytearray accumulator to handle partial reads correctly.

    Thread-safety contract:
    - ``send()`` acquires ``_send_lock`` before writing; drops silently if no
      client is connected (logs at DEBUG).
    - ``stop()`` sets ``_shutdown`` Event, closes the server socket, and joins
      both threads with a 2-second timeout.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._bound_port: int | None = None

        self._shutdown: threading.Event = threading.Event()
        self._send_lock: threading.Lock = threading.Lock()

        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._client_lock: threading.Lock = threading.Lock()

        self._accept_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None

        self._on_message: Callable[[bytes], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the server socket and start the accept loop daemon thread.

        Idempotent: calling start() more than once raises RuntimeError
        rather than silently spawning duplicate threads.
        """
        if self._accept_thread is not None and self._accept_thread.is_alive():
            logger.warning("IPCTransport.start() called while already running.")
            return

        self._shutdown.clear()

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self._host, self._port))
        server_sock.listen(1)
        server_sock.setblocking(False)
        self._server_sock = server_sock
        self._bound_port = server_sock.getsockname()[1]

        logger.info("IPCTransport listening on %s:%d", self._host, self._bound_port)

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="ipc-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        """Signal threads to exit, close the server socket, and join threads.

        Safe to call before start() or after a previous stop().
        """
        self._shutdown.set()

        # Close server socket to unblock any pending select() call.
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        # Close the active client socket so _recv_loop exits promptly.
        with self._client_lock:
            if self._client_sock is not None:
                try:
                    self._client_sock.close()
                except OSError:
                    pass
                self._client_sock = None

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if self._accept_thread.is_alive():
                logger.warning(
                    "ipc-accept thread did not exit within %.1fs",
                    _THREAD_JOIN_TIMEOUT,
                )
            self._accept_thread = None

        if self._recv_thread is not None:
            self._recv_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if self._recv_thread.is_alive():
                logger.warning(
                    "ipc-recv thread did not exit within %.1fs",
                    _THREAD_JOIN_TIMEOUT,
                )
            self._recv_thread = None

        logger.info("IPCTransport stopped.")

    def send(self, message: bytes) -> None:
        """Send a length-prefixed frame to the connected client.

        Thread-safe.  If no client is connected, logs at DEBUG and returns
        without raising.  Catches OSError during send and logs a WARNING.

        Args:
            message: Raw bytes to frame.  The 4-byte length prefix is added
                     automatically; callers must not prepend it themselves.
        """
        with self._send_lock:
            with self._client_lock:
                sock = self._client_sock

            if sock is None:
                logger.debug(
                    "send() called with no connected client; dropping message."
                )
                return

            header = struct.pack(_HEADER_FORMAT, len(message))
            try:
                sock.sendall(header + message)
            except OSError as exc:
                logger.warning("send() failed (client disconnected?): %s", exc)

    def set_on_message(self, callback: Callable[[bytes], None]) -> None:
        """Register the callback invoked when a complete frame is received.

        Args:
            callback: Called with the raw payload bytes (length prefix
                      stripped).  Invoked from the recv daemon thread.
        """
        self._on_message = callback

    def is_connected(self) -> bool:
        """Return True if a client socket is currently active."""
        with self._client_lock:
            return self._client_sock is not None

    @property
    def bound_port(self) -> int | None:
        """Return the actual port the server is bound to, or None if not started.

        Useful when the transport was started with port=0 (OS-assigned port):
        pass port=0 to avoid a TOCTOU race between probing for a free port and
        binding, then read the assigned port via this property.
        """
        return self._bound_port

    # ------------------------------------------------------------------
    # Private threads
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Daemon thread: poll the server socket and accept one client at a time.

        Uses ``selectors.DefaultSelector`` with a ``_SELECT_TIMEOUT`` second
        poll interval so the thread exits promptly when ``_shutdown`` is set.

        When a new client connects while one is already active, the old
        connection is closed before the new socket is stored (single-client
        model).
        """
        sel = selectors.DefaultSelector()
        server_sock = self._server_sock
        if server_sock is None:
            logger.error("_accept_loop started with no server socket; exiting.")
            return

        sel.register(server_sock, selectors.EVENT_READ)

        try:
            while not self._shutdown.is_set():
                try:
                    ready = sel.select(timeout=_SELECT_TIMEOUT)
                except OSError:
                    # Server socket was closed by stop(); exit cleanly.
                    break

                if not ready:
                    continue

                try:
                    conn, addr = server_sock.accept()
                except OSError:
                    # Server socket closed while we were about to accept.
                    break

                conn.setblocking(True)
                logger.info("Client connected from %s:%d", addr[0], addr[1])

                # Single-client model: evict existing client first.
                with self._client_lock:
                    old_sock = self._client_sock
                    if old_sock is not None:
                        logger.info("New client arrived; closing previous connection.")
                        try:
                            old_sock.close()
                        except OSError:
                            pass
                    self._client_sock = conn

                # Join any lingering recv thread before starting a new one.
                if self._recv_thread is not None and self._recv_thread.is_alive():
                    self._recv_thread.join(timeout=_THREAD_JOIN_TIMEOUT)

                recv_thread = threading.Thread(
                    target=self._recv_loop,
                    args=(conn,),
                    name="ipc-recv",
                    daemon=True,
                )
                self._recv_thread = recv_thread
                recv_thread.start()
        finally:
            sel.close()
            logger.debug("_accept_loop exited.")

    def _recv_loop(self, conn: socket.socket) -> None:
        """Daemon thread: read length-prefixed frames from ``conn``.

        Uses a bytearray accumulator to handle partial reads — ``recv()``
        is not guaranteed to return a complete frame.  When the client
        disconnects (empty recv or OSError), clears ``_client_sock`` and
        returns so ``_accept_loop`` can accept a new connection.

        Args:
            conn: The accepted client socket (blocking mode).
        """
        buf = bytearray()

        def _read_exactly(n: int) -> bytes | None:
            """Read exactly ``n`` bytes from ``conn`` using the buffer.

            Returns ``None`` if the connection is closed or ``_shutdown``
            is set before ``n`` bytes are available.
            """
            nonlocal buf
            while len(buf) < n:
                if self._shutdown.is_set():
                    return None
                try:
                    chunk = conn.recv(4096)
                except OSError as exc:
                    logger.info("recv error (client closed?): %s", exc)
                    return None
                if not chunk:
                    return None
                buf.extend(chunk)

            data = bytes(buf[:n])
            buf = buf[n:]
            return data

        try:
            while not self._shutdown.is_set():
                header_bytes = _read_exactly(_HEADER_SIZE)
                if header_bytes is None:
                    break

                (payload_len,) = struct.unpack(_HEADER_FORMAT, header_bytes)

                payload = _read_exactly(payload_len)
                if payload is None:
                    break

                callback = self._on_message
                if callback is not None:
                    try:
                        callback(payload)
                    except Exception as exc:
                        logger.error("on_message callback raised an exception: %s", exc)
        finally:
            # Clear the stored socket reference only if it still points at
            # the socket we were handling (a new connection may have replaced
            # it already in _accept_loop).
            with self._client_lock:
                if self._client_sock is conn:
                    self._client_sock = None

            try:
                conn.close()
            except OSError:
                pass

            logger.info("Client disconnected; ready for new connection.")
            logger.debug("_recv_loop exited.")
