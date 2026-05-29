"""
LLM seam contract tests — CR-24 / Issue #25 seam (2).

Covers: TranscriptReadyEvent → ReasoningRouter.generate → LLMResponseReadyEvent
        (non-empty text) → forwarded as tts_start.

Anti-tautology argument (R2 regression):
    R2 was: the reasoning router emitted empty text (either the model returned
    an empty first token and the router bailed out, or the empty-token guard
    swallowed all content).

    ✗ FAILS on pre-fix (R2) code:
    - test_transcript_drives_nonempty_llm_response: the LLMResponseReadyEvent
      would carry empty text ("") or be missing entirely, causing the assertion
      ``event.text != ""`` to fail.
    - test_llm_response_carries_nonempty_tts_start: the tts_start payload would
      carry an empty "text" field or the frame would never arrive.

    ✓ PASSES on fixed code: the router correctly skips empty tokens and flushes
    the remaining partial sentence at the end, so at least one non-empty
    LLMResponseReadyEvent is always produced.

Mocking strategy
----------------
- llama_cpp.Llama is patched via the ``mock_llama_cpp`` conftest fixture.
- ReasoningRouter uses the real token loop; only the model itself is mocked.
- The mock create_completion returns a stream of canned tokens that form a
  single complete sentence so on_sentence fires exactly once.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import (
    IPCConfig,
    LLMConfig,
    LumiConfig,
    RAGConfig,
    ToolsConfig,
    VisionConfig,
)
from src.core.events import LLMResponseReadyEvent, TranscriptReadyEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Include a leading empty token to exercise the empty-token-skip path (R2 guard).
# Pre-fix code had `if not token: break` which would exit on the first empty
# chunk and produce no output.  The fixed code has `if not token: continue`
# (when nothing collected yet) so it skips the prefix and reaches real content.
_CANNED_TOKENS = ["", "Hello", " world", ". "]  # empty prefix + sentence ending in ". "
_EVENT_TIMEOUT_S = 8.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> LumiConfig:
    return LumiConfig(
        ipc=IPCConfig(enabled=False),
        rag=RAGConfig(enabled=False),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
    )


def _make_token_stream(tokens: list[str]) -> Any:
    """Return a create_completion side-effect that streams tokens via stream=True."""

    def _side_effect(*args: Any, **kwargs: Any) -> Any:
        stream = kwargs.get("stream", False)
        if stream:
            def _gen() -> Any:
                for idx, tok in enumerate(tokens):
                    finish = "stop" if idx == len(tokens) - 1 else None
                    yield {"choices": [{"text": tok, "finish_reason": finish}]}
            return _gen()
        # Non-streaming fallback (not used by ReasoningRouter but needed for safety).
        return {"choices": [{"text": "".join(tokens), "finish_reason": "stop"}]}

    return _side_effect


# ---------------------------------------------------------------------------
# Test 1 — LLM seam: transcript → non-empty LLMResponseReadyEvent (R2)
#
# Anti-tautology: pre-fix (R2), the router emitted empty text, so
# LLMResponseReadyEvent.text would be "" and the assertion fails.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(12)
def test_transcript_drives_nonempty_llm_response(
    mock_llama_cpp: MagicMock,
) -> None:
    """TranscriptReadyEvent drives ReasoningRouter → non-empty LLMResponseReadyEvent.

    The real ReasoningRouter token loop must:
    1. Receive the canned token stream from the mock model.
    2. Accumulate tokens into sentence_buf.
    3. Fire on_sentence with the non-empty sentence text.
    4. Post LLMResponseReadyEvent with text != "".

    R2 anti-tautology: pre-fix, the router swallowed all content (empty first
    token caused premature break) so LLMResponseReadyEvent.text was "".
    The assertion ``event.text`` would be falsy → test fails.
    """
    from src.core.orchestrator import Orchestrator
    from src.core.state_machine import LumiState

    # Configure the mock Llama instance to stream canned tokens.
    mock_llama_instance = mock_llama_cpp.return_value
    mock_llama_instance.create_completion.side_effect = _make_token_stream(
        _CANNED_TOKENS
    )
    # Ensure is_loaded is True so reasoning_router.generate() doesn't bail out.
    mock_llama_instance.is_loaded = True

    config = _minimal_config()

    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()

    with (
        patch("src.core.orchestrator.ModelLoader") as mock_loader_cls,
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
    ):
        # Build the mock ModelLoader so the reasoning router uses our streaming mock.
        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_loader.load = MagicMock()
        mock_loader.model = mock_llama_instance
        mock_loader.get_identity_guard_logit_bias.return_value = {}
        mock_loader_cls.return_value = mock_loader

        mock_mem_cls.return_value.load = MagicMock()
        mock_mem_cls.return_value.get_history.return_value = []
        mock_mem_cls.return_value.add_turn = MagicMock()
        mock_mem_cls.return_value.save = MagicMock()

        orch = Orchestrator(config, speaker=speaker)

    # Drive to LISTENING → PROCESSING so _handle_transcript accepts the event.
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)

    # Directly invoke _handle_transcript (the seam between Scribe and LLM).
    orch._handle_transcript(TranscriptReadyEvent(text="what is the capital of France"))

    # Collect LLMResponseReadyEvent from the event queue.
    deadline = time.monotonic() + _EVENT_TIMEOUT_S
    llm_event: LLMResponseReadyEvent | None = None
    while time.monotonic() < deadline:
        try:
            ev = orch._event_queue.get_nowait()
            if isinstance(ev, LLMResponseReadyEvent):
                llm_event = ev
                break
        except queue.Empty:
            time.sleep(0.02)

    assert llm_event is not None, (
        "LLMResponseReadyEvent was never posted. "
        "R2 regression: reasoning router produced no output."
    )
    assert llm_event.text, (
        "LLMResponseReadyEvent.text is empty. "
        "R2 regression: reasoning router emitted empty text."
    )


# ---------------------------------------------------------------------------
# Test 2 — LLM seam: non-empty text propagated in tts_start wire frame (R2)
#
# Anti-tautology: pre-fix (R2), tts_start would carry empty "text" in its
# payload (or the event would be missing), so the assertion fails.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(15)
def test_llm_response_carries_nonempty_tts_start(
    mock_llama_cpp: MagicMock,
) -> None:
    """Full LLM turn: user_text → LLMResponseReadyEvent → tts_start with non-empty text.

    This exercises the seam between the reasoning router output and the IPC layer:
    1. user_text frame sent via FakeWSClient.
    2. Orchestrator runs the real ReasoningRouter token loop.
    3. LLMResponseReadyEvent is posted with non-empty text.
    4. EventBridge forwards it as tts_start with payload.text non-empty.

    R2 anti-tautology: pre-fix, tts_start.payload.text was "" so the assertion
    ``payload["text"]`` would be falsy → test fails.
    """
    import threading

    from src.core.events import ShutdownEvent
    from tests.integration.fake_ws_client import FakeWSClient

    # Configure the mock Llama instance with a sentence that ends in ". ".
    mock_llama_instance = mock_llama_cpp.return_value
    mock_llama_instance.create_completion.side_effect = _make_token_stream(
        _CANNED_TOKENS
    )
    mock_llama_instance.is_loaded = True

    config = LumiConfig(
        ipc=IPCConfig(enabled=True, port=0),
        rag=RAGConfig(enabled=False),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
    )

    with (
        patch("src.core.orchestrator.ModelLoader") as mock_loader_cls,
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
    ):
        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_loader.load = MagicMock()
        mock_loader.model = mock_llama_instance
        mock_loader.get_identity_guard_logit_bias.return_value = {}
        mock_loader_cls.return_value = mock_loader

        mock_mem_cls.return_value.load = MagicMock()
        mock_mem_cls.return_value.get_history.return_value = []
        mock_mem_cls.return_value.add_turn = MagicMock()
        mock_mem_cls.return_value.save = MagicMock()

        speaker = MagicMock()
        speaker.start = MagicMock()
        speaker.stop = MagicMock()

        from src.core.orchestrator import Orchestrator  # noqa: PLC0415

        orch = Orchestrator(config, speaker=speaker, tts=None)

    # WSTransport.start() (called inside Orchestrator.__init__) blocks until bound.
    port = orch._event_bridge.bound_port  # type: ignore[union-attr]
    assert port is not None, "EventBridge did not bind"

    brain_exc: list[BaseException] = []

    def _run() -> None:
        try:
            orch.run()
        except Exception as exc:
            brain_exc.append(exc)

    thread = threading.Thread(target=_run, daemon=True, name="LLMSeamBrainThread")
    thread.start()

    tts_start_frame: dict[str, Any] | None = None
    try:
        with FakeWSClient(port) as client:
            # do_handshake() reads the hello frame — synchronises on connection accept.
            client.do_handshake()
            client.send_frame("user_text", {"text": "what is the capital of France"})

            # Collect frames until tts_start or timeout.
            deadline = time.monotonic() + _EVENT_TIMEOUT_S
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    frame = client.recv_frame(timeout=min(1.0, remaining))
                    if frame.get("event") == "tts_start":
                        tts_start_frame = frame
                        break
                except TimeoutError:
                    pass
    finally:
        orch._event_queue.put(ShutdownEvent())
        thread.join(timeout=3.0)
        speaker.stop()

    assert not brain_exc, f"Brain thread raised: {brain_exc[0]}"
    assert tts_start_frame is not None, (
        "tts_start frame was never received. "
        "R2 regression: reasoning router did not produce a tts_start event."
    )
    text_payload = tts_start_frame.get("payload", {}).get("text", "")
    assert text_payload, (
        f"tts_start payload.text is empty: {tts_start_frame!r}. "
        "R2 regression: tts_start carries empty text."
    )
