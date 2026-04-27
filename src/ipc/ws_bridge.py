"""WebSocket-to-TCP bridge: relays between Tauri/React (WS) and Brain (TCP)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import websockets
from websockets.asyncio.server import ServerConnection

logger = logging.getLogger(__name__)

_HEADER_FORMAT = "!I"
_HEADER_SIZE = 4
_BACKOFF_INITIAL = 0.5
_BACKOFF_MAX = 8.0


# ---------------------------------------------------------------------------
# TCP helpers
# ---------------------------------------------------------------------------


async def _tcp_read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one length-prefixed frame from the TCP stream. Raises on EOF."""
    header = await reader.readexactly(_HEADER_SIZE)
    (length,) = struct.unpack(_HEADER_FORMAT, header)
    return await reader.readexactly(length)


def _tcp_frame(payload: bytes) -> bytes:
    """Wrap raw bytes in a 4-byte big-endian length prefix."""
    return struct.pack(_HEADER_FORMAT, len(payload)) + payload


# ---------------------------------------------------------------------------
# TCP connection with exponential backoff
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _connect_tcp_with_backoff(
    host: str, port: int
) -> AsyncIterator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    delay = _BACKOFF_INITIAL
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            logger.info("Connected to Brain at %s:%d", host, port)
            try:
                yield reader, writer
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                logger.info("TCP connection to Brain closed.")
            return
        except OSError as exc:
            logger.warning(
                "Cannot connect to Brain at %s:%d (%s); retrying in %.1fs",
                host,
                port,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _BACKOFF_MAX)


# ---------------------------------------------------------------------------
# Bridge core
# ---------------------------------------------------------------------------


class WsBridge:
    """Bidirectional relay between one WebSocket client and the Brain TCP server."""

    def __init__(self, tcp_host: str, tcp_port: int, ws_port: int) -> None:
        self._tcp_host = tcp_host
        self._tcp_port = tcp_port
        self._ws_port = ws_port
        self._ws_client: ServerConnection | None = None
        self._ws_lock = asyncio.Lock()
        self._tcp_writer: asyncio.StreamWriter | None = None
        self._tcp_write_lock = asyncio.Lock()
        self._ws_task: asyncio.Task[None] | None = None

    async def _handle_ws(self, ws: ServerConnection) -> None:
        async with self._ws_lock:
            if self._ws_client is not None:
                logger.warning("Second WS client rejected; only one allowed.")
                await ws.close(1008, "Only one client allowed")
                return
            self._ws_client = ws
        logger.info("React client connected: %s", ws.remote_address)

        try:
            async for raw in ws:
                payload = raw if isinstance(raw, bytes) else raw.encode()
                logger.debug("WS→TCP: %d bytes", len(payload))
                # Queued via a shared writer — the TCP→WS loop owns the reader;
                # we need access to the writer here.  The writer is stored on
                # the bridge instance while the TCP session is live.
                writer = self._tcp_writer
                if writer is None:
                    logger.debug("No TCP connection; dropping WS message.")
                    continue
                try:
                    async with self._tcp_write_lock:
                        writer.write(_tcp_frame(payload))
                        await writer.drain()
                except OSError as exc:
                    logger.warning("Failed to write to Brain: %s", exc)
        except Exception as exc:
            logger.info("WS client disconnected: %s", exc)
        finally:
            async with self._ws_lock:
                self._ws_client = None
            logger.info("React client disconnected.")

    async def _tcp_to_ws_loop(self, reader: asyncio.StreamReader) -> None:
        while True:
            try:
                frame = await _tcp_read_frame(reader)
            except (asyncio.IncompleteReadError, OSError, EOFError) as exc:
                logger.info("Brain TCP stream ended: %s", exc)
                break
            logger.debug("TCP→WS: %d bytes", len(frame))
            ws = self._ws_client
            if ws is None:
                logger.debug("No WS client connected; dropping Brain message.")
                continue
            try:
                # Send bytes directly if the frame is binary; decode to str only
                # for text payloads.  websockets.send() accepts both str and bytes,
                # so we let the type drive the wire format rather than assuming
                # all IPC frames are UTF-8 text (audio/image frames are not).
                await ws.send(frame if isinstance(frame, bytes) else frame.decode())
            except Exception as exc:
                logger.warning("Failed to send to WS client: %s", exc)

    async def run(self) -> None:
        async def _ws_serve() -> None:
            async with websockets.serve(
                self._handle_ws,
                "127.0.0.1",
                self._ws_port,
            ):
                logger.info("WS server listening on ws://127.0.0.1:%d", self._ws_port)
                await asyncio.get_running_loop().create_future()  # run forever

        self._ws_task = asyncio.create_task(_ws_serve())

        try:
            while True:
                async with _connect_tcp_with_backoff(
                    self._tcp_host, self._tcp_port
                ) as (reader, writer):
                    self._tcp_writer = writer
                    try:
                        await self._tcp_to_ws_loop(reader)
                    finally:
                        self._tcp_writer = None
                # After TCP session ends, loop back and reconnect with backoff.
        finally:
            if self._ws_task is not None:
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lumi WS↔TCP bridge")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=5555)
    parser.add_argument("--ws-port", type=int, default=5556)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    bridge = WsBridge(
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        ws_port=args.ws_port,
    )
    asyncio.run(bridge.run())
