"""
IPC seam contract tests — CR-24 / Issue #25 seam (3) + auth-on full turn (4).

Covers:
  3a. WS handshake: hello → hello_ack completes and an event round-trips Brain↔client.
  3b. Malformed frame post-handshake does NOT crash the server.
  4.  Full turn with bearer-token auth ENABLED (overrides the session-scoped
      ``_disable_ipc_token_auth`` autouse fixture from conftest.py).

Anti-tautology argument (R3 regression):
    R3 was: the IPC handshake had a race — the server accepted the WS connection
    before the HandshakeHandler was ready to send the hello frame, so the client
    never received it and the handshake blocked indefinitely.

    ✗ FAILS on pre-fix (R3) code:
    - test_ipc_handshake_and_round_trip: do_handshake() would block/timeout
      waiting for the hello frame that never arrived.
    - test_malformed_frame_does_not_crash_server: the follow-up state_change
      frame would never arrive (server crashed or handshake never completed).

    ✓ PASSES on fixed code: WSTransport calls on_connect immediately after
    accept(), which fires HandshakeHandler.on_client_connected() and sends
    the hello frame before any user data arrives.

Auth-on test (4):
    The session-scoped autouse ``_disable_ipc_token_auth`` in conftest.py patches
    ``src.core.event_bridge.read_token`` to always return None for the ENTIRE test
    session.  The auth-on test overrides this by using a local patch context that
    restores read_token to return a known token, then constructs a fresh EventBridge
    (which reads the token at construction time) inside that context.

    The FakeWSClient.do_handshake_with_token() method sends the correct token in the
    hello_ack frame.  The test verifies that:
    a. The handshake completes successfully (no disconnect).
    b. A subsequent user_text frame round-trips to the event queue.

    This is the ONLY test in the suite that exercises the full IPC stack with auth ON.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Generator

import pytest

from src.core.config import IPCConfig
from src.core.events import (
    LLMResponseReadyEvent,
    UserTextEvent,
)
from src.core.state_machine import LumiState, StateMachine
from src.core.event_bridge import EventBridge
from tests.integration.fake_ws_client import FakeWSClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECT_SETTLE_S: float = 0.08
_RECV_SETTLE_S: float = 0.05
_RECV_TIMEOUT_S: float = 3.0

# The known token used by the auth-on tests.
_TEST_TOKEN = "seam_test_token_abc123"


# ---------------------------------------------------------------------------
# Shared fixtures (local to this module — reuse pattern from test_ipc_full_turn)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _event_queue() -> queue.Queue[Any]:
    return queue.Queue()


@pytest.fixture()
def _state_machine() -> StateMachine:
    return StateMachine()


@pytest.fixture()
def ipc_stack(
    _event_queue: queue.Queue[Any],
    _state_machine: StateMachine,
) -> Generator[tuple[EventBridge, int], None, None]:
    """Real WSTransport + EventBridge on OS-assigned port, no auth."""
    config = IPCConfig(address="127.0.0.1", port=0)
    server = EventBridge(
        config=config,
        event_queue=_event_queue,
        state_machine=_state_machine,
    )
    try:
        server.start()
        time.sleep(0.05)
        port = server.bound_port
        assert port is not None
        yield server, port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Test 3a — IPC seam: hello/hello_ack handshake + event round-trip (R3)
#
# Anti-tautology: pre-fix (R3 race), do_handshake() would block forever waiting
# for a hello frame that was sent BEFORE the WS accept loop was ready, so the
# client never received it.  TimeoutError → test fails.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_ipc_handshake_and_round_trip(
    ipc_stack: tuple[EventBridge, int],
    _state_machine: StateMachine,
) -> None:
    """WS hello/hello_ack completes and a state_change event round-trips to the client.

    Proves:
    1. Brain sends hello frame immediately on connect (no race, R3 fixed).
    2. Client can complete hello_ack handshake.
    3. A subsequent Brain→client event (state_change) is correctly delivered.

    R3 anti-tautology: pre-fix, the handshake raced — hello was sent before the
    recv loop registered the client, so the frame was lost.  do_handshake() would
    timeout → TimeoutError → test fails.
    """
    _, port = ipc_stack

    with FakeWSClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)

        # This is the R3 regression check: do_handshake() must complete within 1 s.
        client.do_handshake()

        # Round-trip: Brain posts state_change → client receives it.
        _state_machine.transition_to(LumiState.LISTENING)
        frame = client.recv_frame(timeout=_RECV_TIMEOUT_S)

    assert frame["event"] == "state_change", (
        f"Expected state_change frame; got {frame!r}. "
        "R3 regression: IPC handshake or event round-trip broken."
    )
    assert frame["payload"]["state"] == "listening"
    assert frame["version"] == "1.0"


# ---------------------------------------------------------------------------
# Test 3b — IPC seam: malformed frame post-handshake does NOT crash server (R3)
#
# Anti-tautology: if the server crashes on a malformed frame the follow-up
# state_change will never arrive.  TimeoutError → test fails.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_malformed_frame_does_not_crash_server(
    ipc_stack: tuple[EventBridge, int],
    _state_machine: StateMachine,
) -> None:
    """Malformed WS frame after handshake must not crash the server.

    EventBridge._on_raw_message catches json.JSONDecodeError and logs a WARNING,
    returning None.  The server must remain operational for subsequent events.

    R3 anti-tautology: if the server crashed, the follow-up state_change frame
    would never arrive.  The recv_frame call would timeout → TimeoutError.
    """
    _, port = ipc_stack

    with FakeWSClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        # Send a frame whose body is not valid JSON.
        client.send_raw_frame(b"INVALID_JSON_NOT_A_DICT}{")

        # Give the recv callback time to process the bad frame.
        time.sleep(_RECV_SETTLE_S)

        # Server must still be alive — trigger a state change.
        _state_machine.transition_to(LumiState.LISTENING)
        frame = client.recv_frame(timeout=_RECV_TIMEOUT_S)

    assert frame["event"] == "state_change", (
        f"Expected state_change; got {frame!r}. "
        "Server crashed or did not recover after malformed frame."
    )
    assert frame["payload"]["state"] == "listening"


# ---------------------------------------------------------------------------
# Test 3c — IPC seam: tts_start Brain→client round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_tts_start_round_trip(
    ipc_stack: tuple[EventBridge, int],
) -> None:
    """EventBridge.on_tts_start() delivers a tts_start frame to the connected client.

    Exercises the Brain→client half of the IPC seam: an internal event posted by
    the Orchestrator (LLMResponseReadyEvent) must be correctly serialised and
    delivered over the wire.
    """
    server, port = ipc_stack

    with FakeWSClient(port) as client:
        time.sleep(_CONNECT_SETTLE_S)
        client.do_handshake()

        server.on_tts_start(LLMResponseReadyEvent(text="Hello from Lumi."))
        frame = client.recv_frame(timeout=_RECV_TIMEOUT_S)

    assert frame["event"] == "tts_start"
    assert frame["payload"]["text"] == "Hello from Lumi."
    assert frame["payload"]["duration_ms"] == 0


# ---------------------------------------------------------------------------
# Test 4 — Full turn with bearer-token auth ENABLED (overrides autouse fixture)
#
# The session-scoped autouse ``_disable_ipc_token_auth`` in conftest.py patches
# read_token → None for ALL tests.  This test restores auth by constructing a
# fresh EventBridge inside a ``patch`` context that returns _TEST_TOKEN, then
# sends the correct token in the hello_ack frame.
#
# This is the only test in the suite that covers the authenticated path end-to-end.
# ---------------------------------------------------------------------------


def _do_handshake_with_token(client: FakeWSClient, token: str, timeout: float = 2.0) -> None:
    """Complete hello/hello_ack with a bearer token in the ack frame.

    Reads the hello frame, then replies with hello_ack carrying *token*.
    Uses client._loop.run_until_complete() exactly like FakeWSClient.do_handshake().

    Args:
        client: A connected FakeWSClient.
        token:  The bearer token to include in hello_ack.
        timeout: Seconds to wait for the hello frame.
    """
    import asyncio
    import json as _json

    async def _do() -> None:
        assert client._ws is not None
        raw = await asyncio.wait_for(client._ws.recv(), timeout=timeout)
        text = raw if isinstance(raw, str) else raw.decode("utf-8")
        obj = _json.loads(text)
        assert obj.get("type") == "hello", f"expected hello frame, got {obj!r}"
        ack = _json.dumps({
            "type": "hello_ack",
            "version": "1.0",
            "status": "ok",
            "token": token,
        })
        await client._ws.send(ack)

    client._loop.run_until_complete(_do())


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_authenticated_handshake_and_round_trip() -> None:
    """Full turn with bearer-token auth ON: hello + correct token → round-trip succeeds.

    Overrides the session-scoped ``_disable_ipc_token_auth`` autouse fixture:
    - Patches ``src.core.event_bridge.read_token`` to return ``_TEST_TOKEN``.
    - Constructs a fresh EventBridge inside that context so it reads the token.
    - Client sends hello_ack with the correct token via ``_do_handshake_with_token``.
    - A follow-up user_text frame must arrive in the event queue (proving the
      handshake completed and the pipeline accepted the authenticated client).

    Why this is NOT covered elsewhere:
    The session-scoped ``_disable_ipc_token_auth`` fixture disables auth globally.
    Every other integration test runs with auth OFF.  Without this test there would
    be NO coverage of the authenticated path end-to-end in the test suite.
    """
    from unittest.mock import patch as _patch

    event_queue_auth: queue.Queue[Any] = queue.Queue()
    state_machine_auth = StateMachine()

    # Override the autouse fixture: make read_token return our test token.
    # EventBridge reads the token at __init__ time, so the patch must be active
    # when EventBridge is constructed below.
    with _patch("src.core.event_bridge.read_token", return_value=_TEST_TOKEN):
        config = IPCConfig(address="127.0.0.1", port=0)
        server = EventBridge(
            config=config,
            event_queue=event_queue_auth,
            state_machine=state_machine_auth,
        )

        try:
            server.start()
            time.sleep(0.05)
            port = server.bound_port
            assert port is not None, "EventBridge did not bind"

            with FakeWSClient(port) as client:
                time.sleep(_CONNECT_SETTLE_S)

                # Perform hello/hello_ack with the correct token.
                _do_handshake_with_token(client, _TEST_TOKEN)

                # Allow the handshake handler time to process the ack and
                # unblock the downstream message path.
                time.sleep(_RECV_SETTLE_S)

                # Send user_text — must reach the event queue because the
                # authenticated handshake completed and downstream is unblocked.
                client.send_frame("user_text", {"text": "auth test message"})

                try:
                    event = event_queue_auth.get(timeout=_RECV_TIMEOUT_S)
                except queue.Empty:
                    event = None

        finally:
            server.stop()

    assert event is not None, (
        "No event received after authenticated handshake. "
        "Authenticated IPC path is broken."
    )
    assert isinstance(event, UserTextEvent), (
        f"Expected UserTextEvent; got {type(event).__name__!r}"
    )
    assert event.text == "auth test message"
