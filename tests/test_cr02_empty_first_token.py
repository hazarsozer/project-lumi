"""CR-02 regression test: empty first token must not silence generation.

Before the fix, create_completion(max_tokens=1) broke immediately on an empty
first token (EOS-class special token from llama.cpp), producing tts_start.text=""
and completing in <500 ms with zero output.

The architectural fix (this version): replace the per-token
create_completion(max_tokens=1) loop (~200 C-API round-trips/response) with a
single create_completion(max_tokens=N, stream=True) call.  The router iterates
the returned generator of chunk dicts, preserving per-sentence TTS streaming and
cancel-flag checks between chunks.

The empty-token-skip logic (the original CR-02 bug fix) is preserved under the
streaming model:
  - Empty chunks that arrive before any content is collected are skipped (prefix
    special tokens / BOS artefacts from llama.cpp).
  - Empty chunks that arrive after content is collected signal EOS and stop the
    loop.
  - finish_reason in ("stop", "length") also stops the loop.

These tests were RED on the pre-fix loop and GREEN after the fix.
They are also GREEN under the streaming refactor (single call, stream=True).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from src.core.config import LLMConfig
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
    config: LLMConfig | None = None,
) -> ReasoningRouter:
    cfg = config or LLMConfig()
    loader = ModelLoader()
    loader.load(cfg)
    engine = PromptEngine()
    memory = ConversationMemory(memory_dir=str(tmp_path))
    return ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=cfg,
    )


def _streaming_side_effect(chunks: list[dict[str, Any]]) -> Any:
    """Return a create_completion side_effect that, when called with stream=True,
    yields *chunks* as a generator (the llama.cpp streaming protocol).

    This replaces the old per-call _side_effect_factory: with the streaming
    router, create_completion is called ONCE and must return a generator that
    yields all token chunks.
    """

    def _side_effect(*args: object, **kwargs: object) -> Generator[dict[str, Any], None, None]:
        yield from chunks

    return _side_effect


# ---------------------------------------------------------------------------
# CR-02 regression: empty first token must not produce empty output
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_first_token_is_skipped_not_silencing(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """An empty token as the FIRST chunk must be skipped, not treated as EOS.

    This is the core CR-02 regression: llama.cpp sometimes returns an empty
    string as the very first token (prefix special token / BOS artefact).
    The old code broke immediately on ``if not token: break``, yielding an
    empty response.  The fix skips empty chunks before any output is collected.

    Under the streaming refactor, create_completion is called ONCE and the mock
    yields all chunks from the returned generator.

    Sequence: ["", "Hello", " world", ""]  (last "" has finish_reason="stop")
    Expected: "Hello world" (not "")
    """
    chunks = [
        {"choices": [{"text": "", "finish_reason": None}]},    # prefix special token
        {"choices": [{"text": "Hello", "finish_reason": None}]},
        {"choices": [{"text": " world", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "stop"}]},   # EOS
    ]
    mock_llama_cpp.return_value.create_completion.side_effect = _streaming_side_effect(chunks)

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()

    result = router.generate("What is the capital of France?", cancel)

    assert result == "Hello world", (
        f"Expected 'Hello world' but got {result!r}. "
        "Empty first token was not skipped — the CR-02 bug is still present."
    )


@pytest.mark.unit
def test_empty_first_token_fires_on_sentence_callback(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """on_sentence callback must fire even when the first chunk is empty.

    Validates that the sentence-streaming TTS path (on_sentence) is not
    bypassed when a prefix special token precedes the real content.
    """
    chunks = [
        {"choices": [{"text": "", "finish_reason": None}]},      # prefix special token
        {"choices": [{"text": "Paris", "finish_reason": None}]},
        {"choices": [{"text": " is great", "finish_reason": None}]},
        {"choices": [{"text": ". ", "finish_reason": None}]},     # sentence boundary
        {"choices": [{"text": "", "finish_reason": "stop"}]},
    ]
    mock_llama_cpp.return_value.create_completion.side_effect = _streaming_side_effect(chunks)

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    sentences: list[str] = []

    result = router.generate("Tell me about Paris.", cancel, on_sentence=sentences.append)

    assert result != "", "Result must be non-empty when tokens follow the prefix token"
    assert len(sentences) >= 1, "on_sentence must fire at least once for the sentence boundary"
    assert "Paris" in sentences[0], f"Expected 'Paris' in first sentence, got {sentences!r}"


@pytest.mark.unit
def test_multiple_empty_prefix_tokens_all_skipped(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """Multiple consecutive empty chunks at the start are all skipped.

    Some llama.cpp quantised models emit several empty tokens before real
    content starts.  All of them must be skipped gracefully.
    """
    chunks = [
        {"choices": [{"text": "", "finish_reason": None}]},   # prefix 1
        {"choices": [{"text": "", "finish_reason": None}]},   # prefix 2
        {"choices": [{"text": "", "finish_reason": None}]},   # prefix 3
        {"choices": [{"text": "Answer", "finish_reason": None}]},
        {"choices": [{"text": " here", "finish_reason": "stop"}]},
    ]
    mock_llama_cpp.return_value.create_completion.side_effect = _streaming_side_effect(chunks)

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()

    result = router.generate("Question?", cancel)

    assert result == "Answer here", (
        f"Expected 'Answer here' but got {result!r}. "
        "Multiple consecutive empty prefix tokens were not all skipped."
    )


@pytest.mark.unit
def test_empty_mid_stream_token_still_stops_loop(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """An empty chunk that arrives AFTER output has been collected must stop generation.

    This preserves the pre-fix behaviour for the case where an empty token
    appears after real content: it signals EOS and the loop must stop.
    """
    chunks = [
        {"choices": [{"text": "hello", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": None}]},    # mid-stream empty = EOS
        # If the loop continued, it would return this token — which we assert
        # must NOT appear in the result.
        {"choices": [{"text": " SHOULD_NOT_APPEAR", "finish_reason": None}]},
    ]
    mock_llama_cpp.return_value.create_completion.side_effect = _streaming_side_effect(chunks)

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()

    result = router.generate("short?", cancel)

    assert result == "hello", (
        f"Expected 'hello' but got {result!r}. "
        "Mid-stream empty token did not stop the loop."
    )
    assert "SHOULD_NOT_APPEAR" not in result


@pytest.mark.unit
def test_create_completion_called_exactly_once(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """create_completion must be called exactly ONCE per generate() invocation.

    This proves the O(1) C-API call contract: the streaming refactor replaced
    the ~200 per-token create_completion(max_tokens=1) calls with a single
    create_completion(max_tokens=N, stream=True) call whose iterator drives
    the generation loop.
    """
    chunks = [
        {"choices": [{"text": "One", "finish_reason": None}]},
        {"choices": [{"text": " two", "finish_reason": None}]},
        {"choices": [{"text": " three", "finish_reason": "stop"}]},
    ]
    mock_llama_cpp.return_value.create_completion.side_effect = _streaming_side_effect(chunks)

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()

    router.generate("Count to three.", cancel)

    call_count = mock_llama_cpp.return_value.create_completion.call_count
    assert call_count == 1, (
        f"Expected create_completion to be called exactly once (O(1) C-API calls) "
        f"but it was called {call_count} time(s). "
        "The per-token loop may still be active."
    )


@pytest.mark.unit
@pytest.mark.timeout(3)
def test_all_empty_stream_returns_empty_string_without_hanging(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """A generator that yields only empty-token chunks must return "" without hanging.

    Guards against an infinite-loop regression if the production code ever
    re-introduces a polling fallback: an all-empty stream must exhaust the
    iterator and return the empty string, not spin indefinitely.
    """
    chunks = [
        {"choices": [{"text": "", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": None}]},
    ]
    mock_llama_cpp.return_value.create_completion.side_effect = _streaming_side_effect(chunks)

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()

    result = router.generate("all empty stream", cancel)

    assert result == "", (
        f"Expected '' for an all-empty stream but got {result!r}. "
        "Empty-only generators must return the empty string, not partial content."
    )
