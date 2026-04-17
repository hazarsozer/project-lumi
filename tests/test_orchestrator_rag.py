"""Tests for Orchestrator RAG wiring — RAGSetEnabledEvent and use_rag dispatch."""

import queue
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from src.core.events import (
    LLMResponseReadyEvent,
    RAGSetEnabledEvent,
    TranscriptReadyEvent,
    UserTextEvent,
)
from src.core.orchestrator import Orchestrator


def _make_config(*, rag_enabled=False, tools_enabled=False, vision_enabled=False, ipc_enabled=False):
    cfg = MagicMock()
    cfg.rag.enabled = rag_enabled
    cfg.tools.enabled = tools_enabled
    cfg.vision.enabled = vision_enabled
    cfg.ipc.enabled = ipc_enabled
    cfg.llm.memory_dir = "/tmp/lumi_test_memory"
    cfg.llm.max_tokens = 5
    cfg.llm.temperature = 0.7
    cfg.llm.context_length = 512
    return cfg


def _make_orchestrator(rag_enabled=False):
    config = _make_config(rag_enabled=rag_enabled)
    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()

    with (
        patch("src.core.orchestrator.ConversationMemory") as mock_mem,
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.PromptEngine"),
        patch("src.core.orchestrator.ReasoningRouter") as mock_rr,
        patch("src.core.orchestrator.ReflexRouter") as mock_reflex,
        patch("src.core.orchestrator.ToolRegistry"),
        patch("src.core.orchestrator.ToolExecutor"),
    ):
        mock_mem.return_value.load = MagicMock()
        mock_mem.return_value.get_history = MagicMock(return_value=[])
        mock_mem.return_value.add_turn = MagicMock()
        mock_mem.return_value.save = MagicMock()

        reflex_instance = MagicMock()
        reflex_instance.route.return_value = None
        reflex_instance.route_rag_intent.return_value = False
        mock_reflex.return_value = reflex_instance

        rr_instance = MagicMock()
        rr_instance.generate.return_value = "response"
        mock_rr.return_value = rr_instance

        orch = Orchestrator(config, speaker=speaker)

    return orch


class TestRAGSetEnabledEvent:
    def test_no_retriever_logs_warning_does_not_raise(self):
        orch = _make_orchestrator(rag_enabled=False)
        assert orch._rag_retriever is None
        # Should not raise
        orch._handle_rag_set_enabled(RAGSetEnabledEvent(enabled=True))

    def test_with_retriever_sets_flag_true(self):
        orch = _make_orchestrator(rag_enabled=False)
        orch._rag_retriever = MagicMock()  # inject fake retriever
        orch._rag_runtime_enabled = False
        orch._handle_rag_set_enabled(RAGSetEnabledEvent(enabled=True))
        assert orch._rag_runtime_enabled is True

    def test_with_retriever_sets_flag_false(self):
        orch = _make_orchestrator(rag_enabled=False)
        orch._rag_retriever = MagicMock()
        orch._rag_runtime_enabled = True
        orch._handle_rag_set_enabled(RAGSetEnabledEvent(enabled=False))
        assert orch._rag_runtime_enabled is False

    def test_handler_registered_for_event(self):
        orch = _make_orchestrator(rag_enabled=False)
        assert RAGSetEnabledEvent in orch._handlers


class TestRAGIntentDispatch:
    def test_use_rag_false_when_rag_disabled(self):
        orch = _make_orchestrator(rag_enabled=False)
        orch._rag_runtime_enabled = False
        orch._reflex_router.route_rag_intent.return_value = True

        # Wake-word pipeline puts state machine in LISTENING before transcript fires.
        from src.core.state_machine import LumiState
        orch._state_machine.transition_to(LumiState.LISTENING)

        # Trigger _handle_transcript
        event = TranscriptReadyEvent(text="search my docs")
        orch._handle_transcript(event)

        # Wait briefly for daemon thread
        import time; time.sleep(0.05)

        generate_call = orch._reasoning_router.generate.call_args
        if generate_call:
            assert generate_call.kwargs.get("use_rag", False) is False

    def test_use_rag_true_when_rag_enabled_and_intent_detected(self):
        orch = _make_orchestrator(rag_enabled=False)
        orch._rag_runtime_enabled = True
        orch._rag_retriever = MagicMock()
        orch._reflex_router.route_rag_intent.return_value = True

        from src.core.state_machine import LumiState
        orch._state_machine.transition_to(LumiState.LISTENING)

        event = TranscriptReadyEvent(text="search my docs for notes")
        orch._handle_transcript(event)

        import time; time.sleep(0.1)

        generate_call = orch._reasoning_router.generate.call_args
        assert generate_call is not None
        assert generate_call.kwargs.get("use_rag") is True

    def test_user_text_event_also_checks_rag_intent(self):
        orch = _make_orchestrator(rag_enabled=False)
        orch._rag_runtime_enabled = True
        orch._rag_retriever = MagicMock()
        orch._reflex_router.route_rag_intent.return_value = True

        event = UserTextEvent(text="find my notes")
        orch._handle_user_text(event)

        import time; time.sleep(0.1)

        generate_call = orch._reasoning_router.generate.call_args
        assert generate_call is not None
        assert generate_call.kwargs.get("use_rag") is True
