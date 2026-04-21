"""
Tests for IPC hello/hello_ack handshake — Wave F6.

Protocol:
  1. Brain sends: {"type": "hello", "version": "1.0", "capabilities": [...]}
  2. Godot replies: {"type": "hello_ack", "version": "1.0", "status": "ok"}
     — or {"type": "hello_ack", "version": "X.Y", "status": "version_mismatch"}
  3. Status != "ok" → Brain logs warning, stays connected.
  4. No hello_ack within 3 s → Brain logs warning, continues.

All tests mock the IPC transport layer; no real sockets are used.
Tests are isolated: each creates its own HandshakeHandler with fresh state.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.handshake import (
    HELLO_CAPABILITIES,
    HELLO_VERSION,
    HANDSHAKE_TIMEOUT_S,
    HandshakeHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(connected: bool = True) -> MagicMock:
    """Return a mock IPCTransport with configurable is_connected()."""
    transport = MagicMock()
    transport.is_connected.return_value = connected
    transport.send = MagicMock()
    return transport


def _encode_frame(msg: dict[str, Any]) -> bytes:
    """Encode a dict as UTF-8 JSON bytes (without the length prefix)."""
    return json.dumps(msg, ensure_ascii=False).encode("utf-8")


def _ack_frame(status: str = "ok", version: str = HELLO_VERSION) -> bytes:
    return _encode_frame(
        {"type": "hello_ack", "version": version, "status": status}
    )


# ---------------------------------------------------------------------------
# Test 1: Brain sends hello frame immediately on connect
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brain_sends_hello_on_connect() -> None:
    """After on_client_connected(), HandshakeHandler sends a hello frame."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    handler.on_client_connected()

    # send() must have been called exactly once.
    assert transport.send.call_count == 1

    raw = transport.send.call_args[0][0]
    frame = json.loads(raw.decode("utf-8"))

    assert frame["type"] == "hello"
    assert "version" in frame
    assert "capabilities" in frame


# ---------------------------------------------------------------------------
# Test 2: hello frame contains version and capabilities
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hello_frame_contains_version_and_capabilities() -> None:
    """Hello frame must carry version == '1.0' and a non-empty capabilities list."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    handler.on_client_connected()

    raw = transport.send.call_args[0][0]
    frame = json.loads(raw.decode("utf-8"))

    assert frame["version"] == HELLO_VERSION
    assert isinstance(frame["capabilities"], list)
    assert len(frame["capabilities"]) > 0
    assert "tts" in frame["capabilities"]
    assert "rag" in frame["capabilities"]
    assert "tools" in frame["capabilities"]


# ---------------------------------------------------------------------------
# Test 3: Normal handshake — hello_ack ok, no warning logged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brain_accepts_hello_ack_ok() -> None:
    """hello_ack with status='ok' completes silently — no warnings logged."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    logger_name = "src.core.handshake"
    with patch.object(logging.getLogger(logger_name), "warning") as mock_warn:
        handler.on_client_connected()
        handler.on_message_received(_ack_frame(status="ok"))

    mock_warn.assert_not_called()
    assert handler.is_handshake_complete()


# ---------------------------------------------------------------------------
# Test 4: Version mismatch — warning logged, NOT disconnected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brain_logs_warning_on_version_mismatch() -> None:
    """hello_ack with status='version_mismatch' logs a warning; transport stays open."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    logger_name = "src.core.handshake"
    with patch.object(logging.getLogger(logger_name), "warning") as mock_warn:
        handler.on_client_connected()
        handler.on_message_received(_ack_frame(status="version_mismatch", version="2.0"))

    # Warning must have been logged.
    assert mock_warn.call_count >= 1
    warning_text = " ".join(str(a) for a in mock_warn.call_args[0])
    assert "version" in warning_text.lower() or "mismatch" in warning_text.lower()

    # Transport must NOT be closed.
    transport.stop.assert_not_called()
    # Handshake is considered "done" (degraded) so the pipeline can continue.
    assert handler.is_handshake_complete()


# ---------------------------------------------------------------------------
# Test 5: Handshake timeout — warning logged, pipeline continues
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brain_logs_warning_on_handshake_timeout() -> None:
    """If no hello_ack arrives within timeout, Brain logs a warning and continues.

    The timeout is patched to a very small value so the test stays fast.
    """
    transport = _make_transport()

    with patch("src.core.handshake.HANDSHAKE_TIMEOUT_S", 0.05):
        handler = HandshakeHandler(transport)

        logger_name = "src.core.handshake"
        with patch.object(logging.getLogger(logger_name), "warning") as mock_warn:
            handler.on_client_connected()
            # Do NOT call on_message_received — simulate missing ack.
            # Wait for the background timeout thread to fire.
            time.sleep(0.2)

        assert mock_warn.call_count >= 1
        warning_text = " ".join(str(a) for a in mock_warn.call_args[0])
        assert "timeout" in warning_text.lower() or "handshake" in warning_text.lower()

    # After timeout, handshake is considered complete (degraded) so pipeline continues.
    assert handler.is_handshake_complete()
    # Transport must still be open.
    transport.stop.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Non-handshake messages before ack are passed to downstream callback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_handshake_messages_forwarded_to_downstream() -> None:
    """Messages that are not hello_ack must be forwarded to the downstream callback."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    received: list[bytes] = []
    handler.set_downstream_callback(received.append)

    handler.on_client_connected()

    regular_msg = _encode_frame({"event": "user_text", "payload": {"text": "hi"}})
    handler.on_message_received(regular_msg)

    assert received == [regular_msg]


# ---------------------------------------------------------------------------
# Test 7: hello_ack is NOT forwarded to downstream callback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hello_ack_not_forwarded_to_downstream() -> None:
    """hello_ack frames must be consumed by the handshake handler, not passed downstream."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    received: list[bytes] = []
    handler.set_downstream_callback(received.append)

    handler.on_client_connected()
    handler.on_message_received(_ack_frame(status="ok"))

    # The hello_ack must NOT appear in the downstream queue.
    assert len(received) == 0


# ---------------------------------------------------------------------------
# Test 8: Duplicate hello_ack is ignored gracefully
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_duplicate_hello_ack_ignored() -> None:
    """A second hello_ack after handshake is already complete must not crash or re-log."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    logger_name = "src.core.handshake"
    with patch.object(logging.getLogger(logger_name), "warning") as mock_warn:
        handler.on_client_connected()
        handler.on_message_received(_ack_frame(status="ok"))
        # Second ack — must be ignored silently.
        handler.on_message_received(_ack_frame(status="ok"))

    # No warnings from the normal ok path.
    mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Test 9: on_message_received before on_client_connected — no crash
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_message_before_connect_does_not_crash() -> None:
    """on_message_received() before on_client_connected() must not raise."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    # Should not raise.
    handler.on_message_received(_ack_frame(status="ok"))


# ---------------------------------------------------------------------------
# Test 10: Malformed hello_ack (invalid JSON) is forwarded to downstream
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_malformed_ack_forwarded_to_downstream() -> None:
    """A frame that cannot be decoded as a valid hello_ack is passed to downstream."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    received: list[bytes] = []
    handler.set_downstream_callback(received.append)

    handler.on_client_connected()

    bad_frame = b"not json }{{}}"
    handler.on_message_received(bad_frame)

    # Invalid JSON → not a hello_ack → forward downstream.
    assert received == [bad_frame]


# ---------------------------------------------------------------------------
# Test 11: Timeout cancelled when ack received before deadline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_timeout_does_not_fire_when_ack_received() -> None:
    """If hello_ack is received before the timeout, no warning is logged."""
    transport = _make_transport()

    with patch("src.core.handshake.HANDSHAKE_TIMEOUT_S", 0.2):
        handler = HandshakeHandler(transport)

        logger_name = "src.core.handshake"
        with patch.object(logging.getLogger(logger_name), "warning") as mock_warn:
            handler.on_client_connected()
            # Deliver ack immediately — well before the 0.2 s timeout.
            handler.on_message_received(_ack_frame(status="ok"))
            # Wait past what would have been the timeout window.
            time.sleep(0.35)

        mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Test 12: Thread-safety — concurrent on_message_received calls
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests 13-15: Branch coverage for edge paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_timer_cancelled_on_reconnect() -> None:
    """A second on_client_connected() cancels the previous timer before starting a new one."""
    transport = _make_transport()

    with patch("src.core.handshake.HANDSHAKE_TIMEOUT_S", 5.0):
        handler = HandshakeHandler(transport)

        handler.on_client_connected()
        first_timer = handler._timeout_timer
        assert first_timer is not None

        # Simulate reconnect — second call should cancel the first timer.
        handler.on_client_connected()

    # The previous timer object must have been cancelled (is_alive → False)
    # and a new timer is now armed.
    assert not first_timer.is_alive(), "First timer should have been cancelled on reconnect"
    # Clean up
    if handler._timeout_timer is not None:
        handler._timeout_timer.cancel()


@pytest.mark.unit
def test_non_dict_json_frame_forwarded_to_downstream() -> None:
    """A JSON array frame (non-dict) must be forwarded downstream, not treated as hello_ack."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    received: list[bytes] = []
    handler.set_downstream_callback(received.append)

    handler.on_client_connected()

    # JSON array parses successfully but is not a dict.
    array_frame = json.dumps(["not", "a", "dict"]).encode("utf-8")
    handler.on_message_received(array_frame)

    assert received == [array_frame]
    # Cancel background timer to avoid side effects.
    if handler._timeout_timer is not None:
        handler._timeout_timer.cancel()


@pytest.mark.unit
def test_timeout_noop_when_ack_already_received() -> None:
    """_on_timeout() is a no-op when handshake was already completed by hello_ack."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    logger_name = "src.core.handshake"
    with patch.object(logging.getLogger(logger_name), "warning") as mock_warn:
        handler.on_client_connected()
        # Complete the handshake first.
        handler.on_message_received(_ack_frame(status="ok"))
        # Now fire the timeout manually, simulating a race.
        handler._on_timeout()

    # The timeout warning must NOT be logged because handshake was already done.
    mock_warn.assert_not_called()


@pytest.mark.unit
def test_concurrent_message_delivery_is_safe() -> None:
    """Concurrent calls to on_message_received() must not raise or deadlock."""
    transport = _make_transport()
    handler = HandshakeHandler(transport)

    received: list[bytes] = []
    handler.set_downstream_callback(received.append)

    handler.on_client_connected()

    regular = _encode_frame({"event": "ping", "payload": {}})

    errors: list[Exception] = []

    def _send_many() -> None:
        try:
            for _ in range(50):
                handler.on_message_received(regular)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_send_many, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Thread errors: {errors}"
