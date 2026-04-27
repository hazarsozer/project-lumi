"""Tests for Phase 6 Wave 2: tool executor and LLM token streaming wired into
the Orchestrator.

All tests mock ModelLoader, PromptEngine, ConversationMemory, ReasoningRouter,
ToolExecutor, SpeakerThread, and KokoroTTS so no real models or hardware are
required.

Threading strategy: since orchestrator inference runs in daemon threads, tests
use threading.Event to wait for completion rather than time.sleep(), except
where the orchestrator's event loop must also be running, in which case run()
is called in a background thread.
"""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig, ToolsConfig
from src.core.events import (
    LLMResponseReadyEvent,
    LLMTokenEvent,
    ShutdownEvent,
    TranscriptReadyEvent,
    UserTextEvent,
)
from src.core.orchestrator import Orchestrator
from src.core.state_machine import LumiState
from src.core.event_bridge import EventBridge
from src.tools.base import ToolResult


def _make_orchestrator(
    tools_enabled: bool = True,
    zmq_server: EventBridge | None = None,
) -> Orchestrator:
    """Create an Orchestrator with mock SpeakerThread and no audio device."""
    mock_speaker = MagicMock(spec=SpeakerThread)
    config = LumiConfig()
    # Replace the ToolsConfig with the desired enabled state. LumiConfig is
    # frozen, so we use object.__setattr__ for the replacement.
    tools_config = ToolsConfig(
        enabled=tools_enabled,
        allowed_tools=("launch_app", "clipboard", "file_info", "window_list"),
        execution_timeout_s=10.0,
    )
    object.__setattr__(config, "tools", tools_config)
    return Orchestrator(config=config, speaker=mock_speaker, zmq_server=zmq_server)


# ---------------------------------------------------------------------------
# Test 1: tool call in response triggers a second generate() call
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(8)
def test_tool_call_in_response_triggers_second_generate(
    mock_llama_cpp: MagicMock,
) -> None:
    """When the LLM returns a tool-call block, the orchestrator executes the
    tool and makes a second generate() call with the tool results injected.
    """
    orch = _make_orchestrator()

    tool_call_response = (
        '<tool_call>{"tool":"file_info","args":{"path":"/tmp"}}</tool_call>'
    )
    final_response = "Done."

    call_count: list[int] = [0]
    response_received: threading.Event = threading.Event()
    received_responses: list[str] = []

    def _fake_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return tool_call_response
        return final_response

    mock_tool_result = ToolResult(success=True, output="file: /tmp (0 bytes)", data={})

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        received_responses.append(e.text)
        response_received.set()
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_fake_generate),
        patch.object(orch._tool_executor, "execute", return_value=[mock_tool_result]),
    ):

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="what is /tmp?"))

        response_received.wait(timeout=6.0)
        loop_thread.join(timeout=3.0)

    assert call_count[0] == 2, f"Expected 2 generate() calls, got {call_count[0]}"
    assert received_responses == [final_response]


# ---------------------------------------------------------------------------
# Test 2: no tool calls — single generate() call
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(8)
def test_no_tool_calls_in_response_single_generate(mock_llama_cpp: MagicMock) -> None:
    """When the LLM returns plain text with no tool-call blocks, generate() is
    called exactly once.
    """
    orch = _make_orchestrator()

    call_count: list[int] = [0]
    response_received = threading.Event()

    def _fake_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        call_count[0] += 1
        return "Hello, I am Lumi."

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        response_received.set()
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_fake_generate),
    ):

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="hello"))

        response_received.wait(timeout=6.0)
        loop_thread.join(timeout=3.0)

    assert call_count[0] == 1, f"Expected 1 generate() call, got {call_count[0]}"


# ---------------------------------------------------------------------------
# Test 3: tools disabled → no tools registered
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tool_call_disabled_config_no_tools_registered() -> None:
    """When config.tools.enabled is False, the tool registry has no tools."""
    orch = _make_orchestrator(tools_enabled=False)
    assert orch._tool_registry.list_tools() == []


# ---------------------------------------------------------------------------
# Test 4: LLMTokenEvent handler registered when ZMQServer is injected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_llm_token_event_handler_registered_with_zmq() -> None:
    """When a ZMQServer is injected, LLMTokenEvent is registered as a handler
    type in the orchestrator's handler map.
    """
    mock_zmq = MagicMock(spec=EventBridge)
    orch = _make_orchestrator(zmq_server=mock_zmq)
    assert LLMTokenEvent in orch._handlers


# ---------------------------------------------------------------------------
# Test 5: utterance_id is passed to generate()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(8)
def test_utterance_id_passed_to_generate(mock_llama_cpp: MagicMock) -> None:
    """The orchestrator passes a non-empty utterance_id kwarg to generate()."""
    orch = _make_orchestrator()

    captured_utterance_ids: list[str] = []
    response_received = threading.Event()

    def _fake_generate(
        text: str,
        cancel_flag: threading.Event,
        utterance_id: str = "",
        **kwargs: object,
    ) -> str:
        captured_utterance_ids.append(utterance_id)
        return "response"

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        response_received.set()
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_fake_generate),
    ):

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="test utterance id"))

        response_received.wait(timeout=6.0)
        loop_thread.join(timeout=3.0)

    assert len(captured_utterance_ids) >= 1
    uid = captured_utterance_ids[0]
    assert uid != "", "utterance_id must be non-empty"
    # Verify it is a valid UUID string.
    parsed = uuid.UUID(uid)
    assert str(parsed) == uid


# ---------------------------------------------------------------------------
# Test 6: cancel flag is passed to tool_executor.execute()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(8)
def test_tool_executor_cancel_flag_passed(mock_llama_cpp: MagicMock) -> None:
    """When the LLM returns a tool-call block, execute() is called with the
    orchestrator's _llm_cancel_flag threading.Event.
    """
    orch = _make_orchestrator()

    tool_call_response = (
        '<tool_call>{"tool":"file_info","args":{"path":"/tmp"}}</tool_call>'
    )
    captured_cancel_flags: list[threading.Event] = []
    response_received = threading.Event()

    call_count: list[int] = [0]

    def _fake_generate(
        text: str, cancel_flag: threading.Event, **kwargs: object
    ) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return tool_call_response
        return "All done."

    def _fake_execute(
        tool_calls: list, cancel_flag: threading.Event
    ) -> list[ToolResult]:
        captured_cancel_flags.append(cancel_flag)
        return [ToolResult(success=True, output="file: /tmp (0 bytes)", data={})]

    def _on_llm_response(e: LLMResponseReadyEvent) -> None:
        response_received.set()
        orch.post_event(ShutdownEvent())

    orch.register_handler(LLMResponseReadyEvent, _on_llm_response)

    with (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reasoning_router, "generate", side_effect=_fake_generate),
        patch.object(orch._tool_executor, "execute", side_effect=_fake_execute),
    ):

        loop_thread = threading.Thread(target=orch.run, daemon=True)
        loop_thread.start()

        orch.state_machine.transition_to(LumiState.LISTENING)
        orch.post_event(TranscriptReadyEvent(text="check /tmp please"))

        response_received.wait(timeout=6.0)
        loop_thread.join(timeout=3.0)

    assert len(captured_cancel_flags) == 1
    assert captured_cancel_flags[0] is orch._llm_cancel_flag
