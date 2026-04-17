"""Tests for LLMTokenEvent streaming in ReasoningRouter.

Validates that the event_queue and utterance_id parameters added in
Phase 6 Wave 1 correctly post LLMTokenEvent instances during generation.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.config import LLMConfig
from src.core.events import LLMTokenEvent
from src.llm.memory import ConversationMemory
from src.llm.model_loader import ModelLoader
from src.llm.prompt_engine import PromptEngine
from src.llm.reasoning_router import ReasoningRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_router(
    mock_llama_cpp: MagicMock,
    tmp_path: Path,
    event_queue: queue.Queue[Any] | None = None,
) -> ReasoningRouter:
    """Construct a fully wired ReasoningRouter with mocked llama_cpp."""
    cfg = LLMConfig()
    loader = ModelLoader()
    loader.load(cfg)
    engine = PromptEngine()
    memory = ConversationMemory(memory_dir=str(tmp_path))
    return ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=cfg,
        event_queue=event_queue,
    )


def _setup_multi_token_mock(mock_llama_cpp: MagicMock) -> None:
    """Configure mock to return three tokens then stop."""
    tokens = ["Hello", " ", "world"]
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        if call_count < len(tokens):
            token = tokens[call_count]
            call_count += 1
            finish = "stop" if call_count == len(tokens) else None
            return {"choices": [{"text": token, "finish_reason": finish}]}
        return {"choices": [{"text": "", "finish_reason": "stop"}]}

    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_posts_llm_token_events(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """With event_queue and utterance_id set, each token posts an LLMTokenEvent."""
    _setup_multi_token_mock(mock_llama_cpp)

    event_q: queue.Queue[Any] = queue.Queue()
    router = _build_router(mock_llama_cpp, tmp_path, event_queue=event_q)
    cancel = threading.Event()

    router.generate("Hi", cancel, utterance_id="utt-1")

    events: list[LLMTokenEvent] = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    assert len(events) == 3
    assert all(isinstance(e, LLMTokenEvent) for e in events)
    assert [e.token for e in events] == ["Hello", " ", "world"]
    assert all(e.utterance_id == "utt-1" for e in events)


@pytest.mark.unit
def test_generate_no_queue_no_events(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """Without event_queue, no LLMTokenEvent is posted; response still returned."""
    _setup_multi_token_mock(mock_llama_cpp)

    router = _build_router(mock_llama_cpp, tmp_path, event_queue=None)
    cancel = threading.Event()

    result = router.generate("Hi", cancel, utterance_id="utt-2")
    assert result == "Hello world"


@pytest.mark.unit
def test_generate_empty_utterance_id_no_events(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """utterance_id='' must suppress LLMTokenEvent posting even with a queue."""
    _setup_multi_token_mock(mock_llama_cpp)

    event_q: queue.Queue[Any] = queue.Queue()
    router = _build_router(mock_llama_cpp, tmp_path, event_queue=event_q)
    cancel = threading.Event()

    result = router.generate("Hi", cancel, utterance_id="")
    assert result == "Hello world"
    assert event_q.empty(), "No events should be posted when utterance_id is empty"


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_generate_cancel_stops_streaming(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """cancel_flag set mid-loop raises InterruptedError; partial tokens are posted."""
    event_q: queue.Queue[Any] = queue.Queue()
    cancel = threading.Event()
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            cancel.set()
        return {"choices": [{"text": f"tok{call_count}", "finish_reason": None}]}

    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect

    router = _build_router(mock_llama_cpp, tmp_path, event_queue=event_q)

    with pytest.raises(InterruptedError):
        router.generate("Long query", cancel, utterance_id="utt-cancel")

    # At least 1 token event should have been posted before cancel
    events: list[LLMTokenEvent] = []
    while not event_q.empty():
        events.append(event_q.get_nowait())
    assert len(events) >= 1
    assert all(isinstance(e, LLMTokenEvent) for e in events)
