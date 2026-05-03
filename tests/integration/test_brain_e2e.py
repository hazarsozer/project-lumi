"""
E2E smoke test — full Brain turn over a live WebSocket connection.

Starts a real Orchestrator (with mocked LLM model and no TTS) on an OS-assigned
port, connects a FakeWSClient, sends a user_text frame, and asserts that the
full pipeline produces llm_token + tts_start + tts_stop wire frames.

Mocking strategy
----------------
- ModelLoader      → mock with is_loaded=True; create_completion yields canned tokens.
- ConversationMemory → mock (no filesystem I/O).
- SpeakerThread    → mock (no sounddevice / audio hardware needed).
- TTS              → None (orchestrator immediately posts SpeechCompletedEvent,
                    which drives tts_stop without real synthesis).
- ReasoningRouter  → NOT mocked; the real token loop runs, exercising the C4
                    on_sentence streaming path added in Ring 2.

IPC stack
---------
EventBridge (WSTransport on port=0) + Orchestrator.run() in a daemon thread.
FakeWSClient connects on the OS-assigned port.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import (
    IPCConfig,
    LumiConfig,
    RAGConfig,
    ToolsConfig,
    VisionConfig,
)
from src.core.events import ShutdownEvent
from tests.integration.fake_ws_client import FakeWSClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tokens that form a single sentence ending in ". " so the C4 sentence-streaming
# path fires on_sentence exactly once, producing one tts_start frame.
_CANNED_TOKENS = ["Paris", " is", " the", " capital. "]

# Non-reflex text — bypasses ReflexRouter and goes through the real LLM path.
_USER_QUERY = "what is the capital of France"

_BIND_SETTLE_S = 0.15    # wait for WSTransport to bind
_CONNECT_SETTLE_S = 0.08  # wait for accept loop to register the client
_FRAME_TIMEOUT_S = 2.0   # per-frame recv timeout
_TOTAL_TIMEOUT_S = 12.0  # wall-clock limit for collecting expected frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_speaker() -> MagicMock:
    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()
    speaker.flush = MagicMock()
    return speaker


def _make_model_loader_cls(tokens: list[str]) -> MagicMock:
    """Return a mock ModelLoader class whose instance yields *tokens* then stops."""
    call_count = 0

    def _create_completion(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        if call_count < len(tokens):
            tok = tokens[call_count]
            call_count += 1
            finish = "stop" if call_count == len(tokens) else None
            return {"choices": [{"text": tok, "finish_reason": finish}]}
        return {"choices": [{"text": "", "finish_reason": "stop"}]}

    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.load = MagicMock()
    mock_model.model = MagicMock()
    mock_model.model.create_completion.side_effect = _create_completion

    return MagicMock(return_value=mock_model)


def _make_memory_cls() -> MagicMock:
    mem = MagicMock()
    mem.load = MagicMock()
    mem.get_history.return_value = []
    mem.add_turn = MagicMock()
    mem.save = MagicMock()
    return MagicMock(return_value=mem)


def _collect_frames(
    client: FakeWSClient,
    stop_on_event: str,
    timeout_s: float = _TOTAL_TIMEOUT_S,
    per_frame_timeout: float = _FRAME_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Drain frames until *stop_on_event* arrives or wall-clock *timeout_s* elapses."""
    received: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            frame = client.recv_frame(timeout=min(per_frame_timeout, remaining))
            received.append(frame)
            if frame.get("event") == stop_on_event:
                break
        except TimeoutError:
            pass
    return received


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(15)
def test_user_text_e2e_produces_llm_token_tts_start_tts_stop() -> None:
    """Full E2E: user_text frame → LLM streaming → llm_token + tts_start + tts_stop.

    What this test proves:
    1. EventBridge correctly accepts a user_text WS frame and routes it to the
       Orchestrator event queue.
    2. The real ReasoningRouter token loop runs and posts LLMTokenEvent per token,
       which EventBridge forwards as llm_token WS frames (on_llm_token handler).
    3. The C4 sentence-streaming path fires on_sentence after ". " boundary,
       posting LLMResponseReadyEvent → forwarded as tts_start (on_tts_start).
    4. With tts=None the orchestrator immediately posts SpeechCompletedEvent,
       which EventBridge forwards as tts_stop (on_tts_stop).
    """
    config = LumiConfig(
        ipc=IPCConfig(enabled=True, port=0),
        rag=RAGConfig(enabled=False),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
    )

    mock_loader_cls = _make_model_loader_cls(_CANNED_TOKENS)
    mock_memory_cls = _make_memory_cls()
    speaker = _mock_speaker()
    brain_exc: list[BaseException] = []

    # Import before patching so 'src.core.orchestrator' is in sys.modules,
    # which is required for patch() to resolve the target attribute path.
    from src.core.orchestrator import Orchestrator  # noqa: PLC0415

    with (
        patch("src.core.orchestrator.ModelLoader", mock_loader_cls),
        patch("src.core.orchestrator.ConversationMemory", mock_memory_cls),
    ):
        orch = Orchestrator(config, speaker=speaker, tts=None)

        # Wait for WSTransport to bind so bound_port is available.
        time.sleep(_BIND_SETTLE_S)
        port = orch._event_bridge.bound_port  # type: ignore[union-attr]
        assert port is not None, "EventBridge did not bind to a port"

        def _run() -> None:
            try:
                orch.run()
            except Exception as exc:
                brain_exc.append(exc)

        run_thread = threading.Thread(target=_run, daemon=True, name="BrainE2EThread")
        run_thread.start()

        try:
            with FakeWSClient(port) as client:
                time.sleep(_CONNECT_SETTLE_S)
                client.do_handshake()

                client.send_frame("user_text", {"text": _USER_QUERY})

                frames = _collect_frames(client, stop_on_event="tts_stop")

        finally:
            orch._event_queue.put(ShutdownEvent())
            run_thread.join(timeout=3.0)
            speaker.stop()

    assert not brain_exc, f"Brain thread raised: {brain_exc[0]}"

    event_names = [f.get("event") for f in frames]

    assert "llm_token" in event_names, (
        f"Expected at least one llm_token frame. Received events: {event_names}"
    )
    assert "tts_start" in event_names, (
        f"Expected at least one tts_start frame. Received events: {event_names}"
    )
    assert "tts_stop" in event_names, (
        f"Expected a tts_stop frame. Received events: {event_names}"
    )

    # Verify at least one llm_token frame has a non-empty token payload.
    token_frames = [f for f in frames if f.get("event") == "llm_token"]
    assert any(
        f.get("payload", {}).get("token") for f in token_frames
    ), f"llm_token frames had empty token payloads: {token_frames}"
