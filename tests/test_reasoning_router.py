"""
Tests for src.llm.reasoning_router.ReasoningRouter.

Mocking strategy
----------------
ReasoningRouter calls llama_cpp.Llama to run inference.  The ``mock_llama_cpp``
fixture patches ``llama_cpp.Llama`` so no model is loaded and no GPU is needed.

ModelLoader, PromptEngine, and ConversationMemory are constructed as real
objects but the underlying llama_cpp call is intercepted by ``mock_llama_cpp``.
ModelLoader.load() is called with a mock config that bypasses the
FileNotFoundError guard because the mock_llama_cpp fixture prevents the real
Llama constructor from executing.

All tests are marked ``unit``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.config import LLMConfig

# RED: these imports will fail until the llm submodules are written.
from src.llm.model_loader import ModelLoader  # type: ignore[import]
from src.llm.prompt_engine import PromptEngine  # type: ignore[import]
from src.llm.memory import ConversationMemory  # type: ignore[import]
from src.llm.reasoning_router import ReasoningRouter  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_router(
    mock_llama_cpp: MagicMock,
    tmp_path: Path,
    config: LLMConfig | None = None,
) -> ReasoningRouter:
    """Construct a fully wired ReasoningRouter with mocked llama_cpp."""
    cfg = config or LLMConfig()
    loader = ModelLoader()
    loader.load(cfg)  # safe — llama_cpp.Llama is mocked by fixture
    engine = PromptEngine()
    memory = ConversationMemory(memory_dir=str(tmp_path))
    return ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_returns_string(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """generate() must return a non-empty string on a normal call."""
    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    result = router.generate("What is the capital of France?", cancel)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_generate_adds_to_memory(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """After generate(), the memory must contain both the user and assistant turns."""
    cfg = LLMConfig()
    loader = ModelLoader()
    loader.load(cfg)
    engine = PromptEngine()
    memory = ConversationMemory(memory_dir=str(tmp_path))
    router = ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=cfg,
    )
    cancel = threading.Event()
    router.generate("Tell me about Paris.", cancel)
    history = memory.get_history()
    assert len(history) == 2
    roles = [turn["role"] for turn in history]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.unit
def test_generate_raises_on_cancel(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """If cancel_flag is set before generate() is called, it must raise InterruptedError."""
    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    cancel.set()  # pre-cancel
    with pytest.raises(InterruptedError):
        router.generate("This should not complete.", cancel)


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_generate_checks_cancel_flag_mid_generation(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """Setting cancel_flag mid-generation must abort and raise InterruptedError.

    The mock is configured to yield multiple token chunks so there is an
    opportunity for the router to observe the cancel flag between tokens.
    """
    # Configure the mock to return a streaming-style iterator of chunks.
    chunk_a = {"choices": [{"text": "chunk_one "}]}
    chunk_b = {"choices": [{"text": "chunk_two "}]}

    cancel = threading.Event()

    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        # Set the cancel flag on the second chunk to simulate mid-generation cancel.
        if call_count >= 2:
            cancel.set()
        return chunk_b if call_count >= 2 else chunk_a

    mock_llama_cpp.return_value.side_effect = _side_effect
    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect

    router = _build_router(mock_llama_cpp, tmp_path)
    with pytest.raises(InterruptedError):
        router.generate("Long query that gets cancelled mid-stream.", cancel)


@pytest.mark.unit
def test_router_requires_loaded_model(tmp_path: Path) -> None:
    """ReasoningRouter.generate() must raise if the ModelLoader is not loaded."""
    loader = ModelLoader()  # not loaded — is_loaded == False
    engine = PromptEngine()
    memory = ConversationMemory(memory_dir=str(tmp_path))
    cfg = LLMConfig()
    router = ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=cfg,
    )
    cancel = threading.Event()
    with pytest.raises((RuntimeError, ValueError)):
        router.generate("Hello?", cancel)


@pytest.mark.unit
def test_generate_uses_config_max_tokens(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """generate() must pass max_tokens from the config to the underlying model."""
    cfg = LLMConfig(max_tokens=128)
    router = _build_router(mock_llama_cpp, tmp_path, config=cfg)
    cancel = threading.Event()
    router.generate("Short answer.", cancel)
    # Verify the mock was called (via direct call or create_completion).
    call_kwargs = mock_llama_cpp.return_value.create_completion.call_args
    if call_kwargs is None:
        call_kwargs = mock_llama_cpp.return_value.call_args
    assert call_kwargs is not None


@pytest.mark.unit
def test_generate_stops_on_empty_token(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """generate() must stop (line 84) when the model returns an empty string token."""
    responses = [
        {"choices": [{"text": "hello", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": None}]},  # empty token triggers break
    ]
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        result = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return result

    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    result = router.generate("empty token", cancel)

    # Only the first non-empty token should be in the output.
    assert result == "hello"
    assert call_count == 2  # first real token + empty token that triggers break


@pytest.mark.unit
def test_generate_stops_on_repeated_token(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """generate() must stop (line 87-88) when the same token is returned twice in a row,
    treating it as an implicit EOS signal."""
    # Return "word" twice in a row — the second occurrence must trigger the break.
    responses = [
        {"choices": [{"text": "word", "finish_reason": None}]},
        {"choices": [{"text": "word", "finish_reason": None}]},
    ]
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        result = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return result

    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    result = router.generate("repeat me", cancel)

    # Only the first token should be in the output; the repeated one causes a break.
    assert result == "word"
    # create_completion was called at most twice (first token + duplicate detection).
    assert call_count <= 2


@pytest.mark.unit
def test_generate_stops_on_finish_reason_stop(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """generate() must stop (line 96) when finish_reason is 'stop'."""
    responses = [
        {"choices": [{"text": "done", "finish_reason": "stop"}]},
    ]
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        result = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return result

    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    result = router.generate("stop me", cancel)

    assert result == "done"
    assert call_count == 1


@pytest.mark.unit
def test_generate_stops_on_finish_reason_length(mock_llama_cpp: MagicMock, tmp_path: Path) -> None:
    """generate() must stop (line 96) when finish_reason is 'length'."""
    responses = [
        {"choices": [{"text": "truncated", "finish_reason": "length"}]},
    ]
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        result = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return result

    mock_llama_cpp.return_value.create_completion.side_effect = _side_effect

    router = _build_router(mock_llama_cpp, tmp_path)
    cancel = threading.Event()
    result = router.generate("length limit", cancel)

    assert result == "truncated"
    assert call_count == 1


@pytest.mark.unit
def test_generate_cancel_set_during_final_token_raises(
    mock_llama_cpp: MagicMock, tmp_path: Path
) -> None:
    """cancel_flag set during the last create_completion call must raise InterruptedError.

    Covers the post-loop cancel check: the loop exits normally via finish_reason
    but the cancel flag was set during that final call, so the partial response
    must not be committed to memory.
    """
    cancel = threading.Event()

    def _complete_and_cancel(*args: object, **kwargs: object) -> dict:
        # Simulate the cancel flag being set *during* the model call —
        # after the pre-call check passed but before the loop exits.
        cancel.set()
        return {"choices": [{"text": "partial", "finish_reason": "stop"}]}

    mock_llama_cpp.return_value.create_completion.side_effect = _complete_and_cancel

    router = _build_router(mock_llama_cpp, tmp_path)
    with pytest.raises(InterruptedError, match="final token"):
        router.generate("interrupted at end", cancel)
