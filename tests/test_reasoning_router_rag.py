"""Tests for ReasoningRouter RAG integration (use_rag=True path)."""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.llm.reasoning_router import ReasoningRouter


def _make_config(max_tokens=10, temperature=0.7, context_length=512):
    cfg = MagicMock()
    cfg.max_tokens = max_tokens
    cfg.temperature = temperature
    cfg.context_length = context_length
    return cfg


def _make_model_loader(response_tokens=("Hello", " world")):
    loader = MagicMock()
    loader.is_loaded = True

    tokens = list(response_tokens)
    call_count = [0]

    def fake_complete(prompt, max_tokens=1, temperature=0.7):
        idx = call_count[0]
        call_count[0] += 1
        if idx >= len(tokens):
            return {"choices": [{"text": "", "finish_reason": "stop"}]}
        finish = "stop" if idx == len(tokens) - 1 else None
        return {"choices": [{"text": tokens[idx], "finish_reason": finish}]}

    loader.model.create_completion.side_effect = fake_complete
    return loader


def _make_router(retriever=None, event_queue=None):
    loader = _make_model_loader()
    engine = MagicMock()
    engine.build_prompt.return_value = "<prompt>"
    engine.truncate_history.return_value = []
    memory = MagicMock()
    memory.get_history.return_value = []
    config = _make_config()
    return ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=config,
        event_queue=event_queue,
        retriever=retriever,
    )


class TestReasoningRouterRAG:
    def test_use_rag_false_does_not_call_retriever(self):
        retriever = MagicMock()
        router = _make_router(retriever=retriever)
        router.generate("hello", threading.Event(), use_rag=False)
        retriever.retrieve.assert_not_called()

    def test_use_rag_true_calls_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(context="doc passage")
        router = _make_router(retriever=retriever)
        router.generate("search my docs", threading.Event(), use_rag=True)
        retriever.retrieve.assert_called_once()

    def test_retriever_context_injected_into_prompt(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(context="key passage")
        router = _make_router(retriever=retriever)
        router.generate("find my notes", threading.Event(), use_rag=True)
        build_call = router._prompt_engine.build_prompt.call_args
        assert build_call.kwargs.get("rag_context") == "key passage"

    def test_no_retriever_use_rag_true_still_works(self):
        router = _make_router(retriever=None)
        response = router.generate("query", threading.Event(), use_rag=True)
        assert isinstance(response, str)

    def test_retriever_exception_does_not_abort_generation(self):
        retriever = MagicMock()
        retriever.retrieve.side_effect = Exception("db error")
        router = _make_router(retriever=retriever)
        # Should not raise — _maybe_retrieve catches and logs
        response = router.generate("search my docs", threading.Event(), use_rag=True)
        assert isinstance(response, str)

    def test_cancel_before_retrieve_raises_interrupted(self):
        retriever = MagicMock()
        router = _make_router(retriever=retriever)
        flag = threading.Event()
        flag.set()
        with pytest.raises(InterruptedError):
            router.generate("query", flag, use_rag=True)
        retriever.retrieve.assert_not_called()

    def test_default_use_rag_is_false(self):
        retriever = MagicMock()
        router = _make_router(retriever=retriever)
        router.generate("hello", threading.Event())
        retriever.retrieve.assert_not_called()
