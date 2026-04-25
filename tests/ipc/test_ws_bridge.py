"""
Tests for src/ipc/ws_bridge.py — WebSocket-to-TCP bridge.

## Architecture Under Test

WsBridge relays bidirectionally between:
  - WebSocket clients (Tauri/React frontend) at ws://127.0.0.1:<ws_port>
  - A TCP "Brain" server at 127.0.0.1:<tcp_port>
  - TCP wire format: 4-byte big-endian uint32 length prefix + UTF-8 JSON payload

## Async Test Runner

Tests use the ``anyio`` pytest plugin (already installed as a transitive dep).
``anyio_mode = "auto"`` in ``pyproject.toml`` means all ``async def test_*``
functions run automatically.  The module-level ``anyio_backend`` fixture
overrides the default parametrisation to restrict execution to asyncio only
(trio is not tested here because the bridge uses ``asyncio``-specific APIs).

## Mocking Strategy

Rather than calling ``WsBridge.run()`` (which loops forever over TCP reconnects),
tests decompose the bridge into three independently testable surfaces:

1. **Framing helpers** — ``_tcp_frame`` / ``_tcp_read_frame`` are pure functions;
   tested directly with in-memory ``asyncio.StreamReader`` objects.

2. **WS handler in isolation** — ``bridge._handle_ws(ws)`` is the coroutine that
   the websockets server calls for each new client.  Tests start the WS server
   manually via ``websockets.asyncio.server.serve``, then connect a real WS client
   using ``websockets.asyncio.client.connect``.  ``bridge._tcp_writer`` is set
   directly on the instance to simulate the TCP session being up or down.

3. **TCP→WS relay** — ``bridge._tcp_to_ws_loop(reader)`` is driven by feeding
   pre-framed bytes into an ``asyncio.StreamReader`` (in-memory; no real socket).

## Test Isolation

Every test gets its own ``WsBridge`` instance and its own WS server bound to an
OS-assigned port (port=0 via ``websockets.asyncio.server.serve``).  Mock TCP
endpoints (where needed) use ``asyncio.start_server(port=0)``.  All resources
are cleaned up in fixture teardown even when the test body raises.

## Known Bug (Bug 1 — uninitialized _tcp_writer)

``WsBridge.__init__`` does NOT set ``self._tcp_writer``.  The attribute is only
assigned inside ``WsBridge.run()``.  Any code path that calls ``_handle_ws``
without first calling ``run()`` will raise ``AttributeError`` on the
``writer = self._tcp_writer`` line.  The test ``test_clean_init_tcp_writer_is_none``
asserts the correct behaviour (attribute present, value ``None``) and will
**fail** until the bug is fixed in ``ws_bridge.py``.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import websockets.asyncio.client as ws_client
import websockets.asyncio.server as ws_server

from src.ipc.ws_bridge import WsBridge, _tcp_frame, _tcp_read_frame

# ---------------------------------------------------------------------------
# Restrict anyio to asyncio backend only (bridge uses asyncio-native APIs)
# ---------------------------------------------------------------------------

anyio_backend = "asyncio"

# ---------------------------------------------------------------------------
# Constants / wire-format helpers
# ---------------------------------------------------------------------------

_HEADER_FORMAT = "!I"
_HEADER_SIZE = 4


def _encode_frame(payload: bytes) -> bytes:
    """Prepend a 4-byte big-endian length prefix — mirrors bridge wire format."""
    return struct.pack(_HEADER_FORMAT, len(payload)) + payload


def _make_stream_reader(data: bytes) -> asyncio.StreamReader:
    """Return a StreamReader pre-loaded with ``data`` and an EOF marker."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


# ---------------------------------------------------------------------------
# Fixture: WsBridge instance with its own live WS server
# ---------------------------------------------------------------------------


class _BridgeHarness:
    """Holds a WsBridge instance together with its running WS server."""

    def __init__(
        self,
        bridge: WsBridge,
        ws_server_ctx: ws_server.Server,
        ws_port: int,
    ) -> None:
        self.bridge = bridge
        self.ws_server = ws_server_ctx
        self.ws_port = ws_port
        self.ws_uri = f"ws://127.0.0.1:{ws_port}"


@pytest.fixture()
async def bridge_harness() -> AsyncGenerator[_BridgeHarness, None]:
    """Start a WsBridge WS server on an OS-assigned port.

    The bridge's ``_tcp_writer`` is manually set to ``None`` so that tests
    that need it absent can work without depending on Bug 1 being present.
    The TCP reconnect loop (``run()``) is NOT started; tests drive the
    bridge's handler coroutines directly.

    Teardown closes the WS server and waits for it to finish.
    """
    bridge = WsBridge(tcp_host="127.0.0.1", tcp_port=0, ws_port=0)
    # Manually initialise the attribute that run() would set.
    # This sidesteps Bug 1 for tests that do NOT exercise the bug directly.
    bridge._tcp_writer = None  # type: ignore[attr-defined]

    # Start the WS server on an OS-assigned port (port=0).
    serve_ctx = await ws_server.serve(bridge._handle_ws, "127.0.0.1", 0)
    ws_port: int = serve_ctx.sockets[0].getsockname()[1]

    harness = _BridgeHarness(bridge, serve_ctx, ws_port)
    try:
        yield harness
    finally:
        serve_ctx.close()
        await serve_ctx.wait_closed()


# ---------------------------------------------------------------------------
# 1. TCP framing — pure unit tests (no network, no sockets)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_tcp_frame_encodes_length_prefix() -> None:
    """_tcp_frame prepends a correct 4-byte big-endian uint32 length prefix."""
    payload = b'{"event": "state_change"}'
    framed = _tcp_frame(payload)

    assert len(framed) == _HEADER_SIZE + len(payload)
    (length,) = struct.unpack(_HEADER_FORMAT, framed[:_HEADER_SIZE])
    assert length == len(payload)
    assert framed[_HEADER_SIZE:] == payload


@pytest.mark.unit
async def test_tcp_read_frame_round_trip() -> None:
    """_tcp_read_frame recovers the original payload from a framed stream."""
    payload = json.dumps({"event": "tts_chunk", "text": "hello"}).encode()
    reader = _make_stream_reader(_tcp_frame(payload))

    recovered = await _tcp_read_frame(reader)
    assert recovered == payload


@pytest.mark.unit
async def test_tcp_read_frame_raises_on_truncated_stream() -> None:
    """_tcp_read_frame raises asyncio.IncompleteReadError on a short stream."""
    # Feed only the header with length=100, but no body data.
    reader = _make_stream_reader(struct.pack(_HEADER_FORMAT, 100))

    with pytest.raises(asyncio.IncompleteReadError):
        await _tcp_read_frame(reader)


@pytest.mark.unit
async def test_framing_preserves_utf8_json() -> None:
    """Frame/unframe cycle is transparent for arbitrary UTF-8 JSON strings."""
    original = {"msg": "こんにちは Lumi!", "value": 42}
    payload = json.dumps(original).encode()
    reader = _make_stream_reader(_tcp_frame(payload))

    recovered = await _tcp_read_frame(reader)
    assert json.loads(recovered) == original


# ---------------------------------------------------------------------------
# 2. WS→TCP relay — message forwarded to the mock TCP writer
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_ws_message_forwarded_to_tcp(bridge_harness: _BridgeHarness) -> None:
    """WS message is encoded as a length-prefixed frame and written to TCP writer.

    Injects a lightweight mock StreamWriter that captures written bytes, connects
    a real WS client, sends a JSON string, then asserts the captured bytes match
    the expected length-prefixed frame.
    """
    written: list[bytes] = []

    class _MockWriter:
        def write(self, data: bytes) -> None:
            written.append(data)

        async def drain(self) -> None:
            pass

    bridge_harness.bridge._tcp_writer = _MockWriter()  # type: ignore[assignment]

    message = json.dumps({"event": "user_text", "payload": {"text": "hi lumi"}})
    expected_frame = _tcp_frame(message.encode())

    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        await ws.send(message)
        # Yield to the event loop so _handle_ws processes the message.
        await asyncio.sleep(0.05)

    assert written, "No bytes were written to the mock TCP writer"
    assert b"".join(written) == expected_frame


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_ws_bytes_frame_decoded_before_tcp_framing(
    bridge_harness: _BridgeHarness,
) -> None:
    """WS binary frames (bytes) are decoded to str before TCP framing.

    ``_handle_ws`` converts raw bytes to str via ``.decode()`` before encoding
    them as a TCP frame, ensuring the TCP payload is always UTF-8 text.
    """
    written: list[bytes] = []

    class _MockWriter:
        def write(self, data: bytes) -> None:
            written.append(data)

        async def drain(self) -> None:
            pass

    bridge_harness.bridge._tcp_writer = _MockWriter()  # type: ignore[assignment]

    message = json.dumps({"event": "ping"})
    # Send as bytes — _handle_ws must decode before framing.
    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        await ws.send(message.encode())
        await asyncio.sleep(0.05)

    assert b"".join(written) == _tcp_frame(message.encode())


# ---------------------------------------------------------------------------
# 3. Single-client policy — second connection rejected with code 1008
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_second_client_rejected_with_code_1008(
    bridge_harness: _BridgeHarness,
) -> None:
    """A second WS connection is rejected with close code 1008 (policy violation).

    The first client must remain connected (bridge._ws_client is not None) after
    the rejection.  Note: bridge._ws_client is a ServerConnection (server-side
    object), which is a different Python object from the client-side ClientConnection
    returned by ``ws_client.connect``; only close_code and non-None checks are valid.
    """
    async with ws_client.connect(bridge_harness.ws_uri):
        # Let _handle_ws register the first client.
        await asyncio.sleep(0.05)

        # _ws_client is now populated with the server-side connection object.
        assert bridge_harness.bridge._ws_client is not None

        # Second connection — bridge will close it with 1008.
        second_ws = await ws_client.connect(bridge_harness.ws_uri)
        try:
            await asyncio.wait_for(second_ws.wait_closed(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

        assert second_ws.close_code == 1008, (
            f"Expected close code 1008 (policy violation), got {second_ws.close_code}"
        )
        # First client is still registered — _ws_client must not have been cleared.
        assert bridge_harness.bridge._ws_client is not None


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_first_client_remains_registered_after_rejection(
    bridge_harness: _BridgeHarness,
) -> None:
    """After rejecting the second WS client, _ws_client is still set (non-None).

    We cannot use ``is first_ws`` because the bridge holds the server-side
    ServerConnection, not the client-side ClientConnection object.
    """
    async with ws_client.connect(bridge_harness.ws_uri):
        await asyncio.sleep(0.05)

        second_ws = await ws_client.connect(bridge_harness.ws_uri)
        try:
            await asyncio.wait_for(second_ws.wait_closed(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

        # First client's server-side connection remains registered.
        assert bridge_harness.bridge._ws_client is not None


# ---------------------------------------------------------------------------
# 4. WS-without-TCP: message dropped gracefully when _tcp_writer is None
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_ws_message_dropped_when_no_tcp(bridge_harness: _BridgeHarness) -> None:
    """WS message sent before TCP is up is dropped; no exception propagates.

    ``bridge._tcp_writer`` is already ``None`` from the fixture.  Sending a WS
    message must not crash the handler — the WS client must remain connected
    and able to send further messages without error.
    """
    assert bridge_harness.bridge._tcp_writer is None

    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        await asyncio.sleep(0.02)

        # This send should be silently dropped — no OSError propagated.
        await ws.send(json.dumps({"event": "user_text"}))
        await asyncio.sleep(0.05)

        # Connection is still open: we can send another message without error.
        await ws.send(json.dumps({"event": "ping"}))
        await asyncio.sleep(0.02)

    # No explicit assertion needed beyond "no exception raised" — that is the contract.


# ---------------------------------------------------------------------------
# 5. TCP→WS relay loop — in-memory stream drives WS client delivery
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_tcp_to_ws_loop_delivers_frame_to_ws_client(
    bridge_harness: _BridgeHarness,
) -> None:
    """_tcp_to_ws_loop reads a framed TCP message and sends it as text to the WS client.

    The TCP stream is simulated with an in-memory StreamReader pre-loaded with
    a single frame followed by EOF (which ends the loop naturally).
    """
    payload = json.dumps({"event": "tts_chunk", "text": "Hello, world!"}).encode()
    framed_stream = _make_stream_reader(_tcp_frame(payload))

    received: list[str] = []

    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        # Let _handle_ws register the client.
        await asyncio.sleep(0.05)

        # Run the TCP→WS relay loop until the stream hits EOF.
        loop_task = asyncio.create_task(
            bridge_harness.bridge._tcp_to_ws_loop(framed_stream)
        )

        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            received.append(msg if isinstance(msg, str) else msg.decode())
        except asyncio.TimeoutError:
            pass
        finally:
            await asyncio.wait_for(loop_task, timeout=2.0)

    assert received == [payload.decode()], (
        f"Expected [{payload.decode()!r}], got {received!r}"
    )


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_tcp_to_ws_loop_drops_when_no_ws_client(
    bridge_harness: _BridgeHarness,
) -> None:
    """_tcp_to_ws_loop drops TCP frames silently when no WS client is connected.

    ``_ws_client`` stays None (no WS connection opened in this test).  The loop
    must complete without error.
    """
    assert bridge_harness.bridge._ws_client is None

    payload = b'{"event": "state_change", "state": "idle"}'
    # Two frames followed by EOF.
    data = _tcp_frame(payload) + _tcp_frame(payload)
    reader = _make_stream_reader(data)

    # Must complete without raising.
    await asyncio.wait_for(
        bridge_harness.bridge._tcp_to_ws_loop(reader),
        timeout=3.0,
    )


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_tcp_to_ws_loop_exits_on_eof(bridge_harness: _BridgeHarness) -> None:
    """_tcp_to_ws_loop exits cleanly when the TCP stream reaches EOF immediately."""
    reader = asyncio.StreamReader()
    reader.feed_eof()

    # Must return without raising, well within 3 seconds.
    await asyncio.wait_for(
        bridge_harness.bridge._tcp_to_ws_loop(reader),
        timeout=3.0,
    )


# ---------------------------------------------------------------------------
# 6. TCP reconnect — bridge reconnects after TCP connection drops
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(15)
async def test_tcp_reconnect_after_disconnect() -> None:
    """After the TCP connection drops, the bridge reconnects automatically.

    Strategy:
    1. Start a mock TCP server on an OS-assigned port.
    2. Start ``bridge.run()`` as a background task (connects to mock server).
    3. Close the mock TCP server; the bridge's TCP session ends.
    4. Restart a new TCP server on the same port; verify the bridge reconnects.

    ``_BACKOFF_INITIAL = 0.5`` so the bridge waits 0.5 s before retrying.
    The 15 s test budget is far above the required time.
    """
    connected_event = asyncio.Event()
    connection_count = 0

    async def _tcp_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal connection_count
        connection_count += 1
        connected_event.set()
        # Hold the connection briefly, then close it.
        await asyncio.sleep(0.2)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    # Phase 1: start the first TCP server.
    first_server = await asyncio.start_server(_tcp_handler, "127.0.0.1", 0)
    tcp_port: int = first_server.sockets[0].getsockname()[1]

    bridge = WsBridge(tcp_host="127.0.0.1", tcp_port=tcp_port, ws_port=0)

    # Run the bridge in the background; it will connect to the TCP server.
    run_task = asyncio.create_task(bridge.run())

    try:
        # Wait for the first connection.
        await asyncio.wait_for(connected_event.wait(), timeout=5.0)
        assert connection_count >= 1, "Bridge never connected to TCP server"

        # Close the first server so the bridge's TCP session drops.
        first_server.close()
        await first_server.wait_closed()

        # Reset the event for the second connection attempt.
        connected_event.clear()

        # Phase 2: bring a new TCP server up on the same port.
        # The bridge will retry with 0.5 s backoff and reconnect.
        second_server = await asyncio.start_server(
            _tcp_handler, "127.0.0.1", tcp_port
        )
        try:
            await asyncio.wait_for(connected_event.wait(), timeout=8.0)
            assert connection_count >= 2, (
                "Bridge did not reconnect after TCP server came back up"
            )
        finally:
            second_server.close()
            await second_server.wait_closed()
    finally:
        run_task.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ---------------------------------------------------------------------------
# 7. Clean init — _tcp_writer must be None after __init__ (Bug 1 detection)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clean_init_tcp_writer_is_none() -> None:
    """WsBridge.__init__ must initialise _tcp_writer to None.

    EXPECTED FAILURE until Bug 1 is fixed in ws_bridge.py:
    Currently _tcp_writer is only assigned inside run(), so accessing it
    after __init__ raises AttributeError.  This test documents the correct
    contract: the attribute must exist and be None after construction.

    Fix required in ws_bridge.py:
        Add ``self._tcp_writer: asyncio.StreamWriter | None = None`` to __init__.
    """
    bridge = WsBridge(tcp_host="127.0.0.1", tcp_port=5555, ws_port=5556)

    assert hasattr(bridge, "_tcp_writer"), (
        "Bug 1: WsBridge.__init__ does not initialise _tcp_writer. "
        "The attribute is only set inside run(), causing AttributeError in "
        "_handle_ws when called before run() (e.g. in tests or if the WS "
        "server accepts a connection before the TCP loop starts). "
        "Fix: add `self._tcp_writer: asyncio.StreamWriter | None = None` to __init__."
    )
    assert bridge._tcp_writer is None, (  # type: ignore[attr-defined]
        f"_tcp_writer should be None after __init__, got {bridge._tcp_writer!r}"
    )


# ---------------------------------------------------------------------------
# 8. WS client disconnection clears _ws_client
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_ws_client_cleared_on_disconnect(bridge_harness: _BridgeHarness) -> None:
    """When the WS client disconnects, _ws_client is set back to None."""
    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        await asyncio.sleep(0.05)
        assert bridge_harness.bridge._ws_client is not None

    # After the context manager exits (connection closed), the handler's
    # finally block must clear _ws_client.
    await asyncio.sleep(0.15)
    assert bridge_harness.bridge._ws_client is None


# ---------------------------------------------------------------------------
# 9. Sequential clients — second client accepted after first disconnects
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_sequential_clients_both_accepted(
    bridge_harness: _BridgeHarness,
) -> None:
    """After first client disconnects, a second client is accepted normally."""
    # First client connects then leaves.
    async with ws_client.connect(bridge_harness.ws_uri):
        await asyncio.sleep(0.05)

    # Wait for _ws_client to be cleared by the finally block.
    await asyncio.sleep(0.15)
    assert bridge_harness.bridge._ws_client is None

    # Second client should now be accepted without rejection.
    # We cannot use ``is ws2`` because the bridge holds the ServerConnection
    # (server-side), not the ClientConnection (client-side) object.
    # We verify acceptance by confirming _ws_client is populated.
    async with ws_client.connect(bridge_harness.ws_uri):
        await asyncio.sleep(0.05)
        assert bridge_harness.bridge._ws_client is not None


# ---------------------------------------------------------------------------
# 10. Error injection — OSError from writer.drain() is caught (lines 116-117)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_tcp_writer_oserror_is_caught(bridge_harness: _BridgeHarness) -> None:
    """OSError raised by the TCP writer's drain() is caught and logged.

    Covers lines 116-117 in _handle_ws: the ``except OSError`` path.
    The WS client must remain connected after the write failure.
    """

    class _FailingWriter:
        def write(self, data: bytes) -> None:
            pass  # write() itself succeeds

        async def drain(self) -> None:
            raise OSError("broken pipe")

    bridge_harness.bridge._tcp_writer = _FailingWriter()  # type: ignore[assignment]

    # Send a message — drain() will raise OSError, which must be caught.
    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        await asyncio.sleep(0.02)
        await ws.send(json.dumps({"event": "test"}))
        await asyncio.sleep(0.05)

        # WS connection is still usable after the OSError was swallowed.
        await ws.send(json.dumps({"event": "still_alive"}))
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# 11. Error injection — ws.send() exception in _tcp_to_ws_loop (lines 139-140)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_tcp_to_ws_loop_send_exception_is_caught(
    bridge_harness: _BridgeHarness,
) -> None:
    """Exception raised by ws.send() inside _tcp_to_ws_loop is caught and logged.

    Covers lines 139-140: the ``except Exception`` path in _tcp_to_ws_loop.
    The loop must continue after the failed send (with the next frame succeeding).
    """
    # Two frames in the stream: the first send will fail; the second should succeed.
    payload1 = b'{"event": "first"}'
    payload2 = b'{"event": "second"}'
    data = _tcp_frame(payload1) + _tcp_frame(payload2)
    reader = _make_stream_reader(data)

    sent_successfully: list[str] = []
    send_call_count = 0

    class _FaultyServerConn:
        """Mimics a ServerConnection whose first send() raises."""

        async def send(self, msg: str) -> None:
            nonlocal send_call_count
            send_call_count += 1
            if send_call_count == 1:
                raise RuntimeError("simulated send failure")
            sent_successfully.append(msg)

    bridge_harness.bridge._ws_client = _FaultyServerConn()  # type: ignore[assignment]

    # Run the loop — it must not propagate the exception from send().
    await asyncio.wait_for(
        bridge_harness.bridge._tcp_to_ws_loop(reader),
        timeout=3.0,
    )

    # The second frame must have been delivered despite the first send failing.
    assert sent_successfully == [payload2.decode()], (
        f"Expected second frame to be sent after first failed; got {sent_successfully!r}"
    )


# ---------------------------------------------------------------------------
# 12. Backoff retry — _connect_tcp_with_backoff retries on OSError (lines 65-73)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(15)
async def test_bridge_retries_on_initial_connection_failure() -> None:
    """Bridge retries connecting to TCP when the server is not yet up.

    ``_connect_tcp_with_backoff`` catches OSError and sleeps before retrying.
    This test starts the mock TCP server *after* the bridge has begun trying,
    causing at least one retry (covering lines 65-73).
    """
    connected_event = asyncio.Event()

    async def _tcp_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        connected_event.set()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    # Find a free port without holding it open so the bridge fails the first attempt.
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        tcp_port = s.getsockname()[1]
    # Port is now free but no server is listening.

    bridge = WsBridge(tcp_host="127.0.0.1", tcp_port=tcp_port, ws_port=0)
    run_task = asyncio.create_task(bridge.run())

    try:
        # Wait a bit so the bridge makes at least one failed attempt.
        await asyncio.sleep(0.3)

        # Start the TCP server — the bridge should reconnect.
        server = await asyncio.start_server(_tcp_handler, "127.0.0.1", tcp_port)
        try:
            await asyncio.wait_for(connected_event.wait(), timeout=8.0)
            assert connected_event.is_set(), "Bridge never connected after retry"
        finally:
            server.close()
            await server.wait_closed()
    finally:
        run_task.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ---------------------------------------------------------------------------
# 13. WS abrupt disconnect — _handle_ws outer except clause (lines 118-119)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_ws_abrupt_close_clears_client(bridge_harness: _BridgeHarness) -> None:
    """_handle_ws catches exceptions when the WS connection closes abruptly.

    Covers lines 118-119: the outer ``except Exception`` block in ``_handle_ws``.
    An abrupt transport close forces the ``async for raw in ws:`` iterator to raise,
    triggering the except clause.  After the exception is handled, ``_ws_client``
    must be cleared to None by the finally block.
    """
    async with ws_client.connect(bridge_harness.ws_uri) as ws:
        await asyncio.sleep(0.05)
        assert bridge_harness.bridge._ws_client is not None

        # Abruptly close the underlying transport — forces the server-side
        # async iterator to raise an exception, hitting line 118.
        ws.transport.close()

    # Give the server-side handler time to catch the exception and clear _ws_client.
    await asyncio.sleep(0.2)
    assert bridge_harness.bridge._ws_client is None
