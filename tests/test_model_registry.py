"""Tests for src.llm.model_registry.ModelRegistry."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.core.config import LLMConfig
from src.llm.model_registry import ModelRegistry


def _cfg(path: str = "models/llm/fake.gguf") -> LLMConfig:
    return LLMConfig(model_path=path)


@pytest.mark.unit
def test_list_empty_on_init(mock_llama_cpp: MagicMock) -> None:
    assert ModelRegistry().list_registered() == []


@pytest.mark.unit
def test_register_appears_in_list(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    registry.register("base", _cfg())
    assert "base" in registry.list_registered()


@pytest.mark.unit
def test_register_overwrites_silently(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    registry.register("base", _cfg("a.gguf"))
    registry.register("base", _cfg("b.gguf"))
    assert registry.list_registered() == ["base"]


@pytest.mark.unit
def test_load_unknown_raises_key_error(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="unknown"):
        registry.load("unknown")


@pytest.mark.unit
def test_load_sets_is_loaded_and_current_name(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    registry.register("base", _cfg())
    registry.load("base")
    assert registry.is_loaded is True
    assert registry.current_name == "base"


@pytest.mark.unit
def test_load_swap_updates_current_name(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    registry.register("a", _cfg("a.gguf"))
    registry.register("b", _cfg("b.gguf"))
    registry.load("a")
    registry.load("b")
    assert registry.current_name == "b"
    assert registry.is_loaded is True


@pytest.mark.unit
def test_unload_clears_state(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    registry.register("base", _cfg())
    registry.load("base")
    registry.unload()
    assert registry.is_loaded is False
    assert registry.current_name is None


@pytest.mark.unit
def test_model_raises_when_not_loaded(mock_llama_cpp: MagicMock) -> None:
    with pytest.raises(RuntimeError):
        _ = ModelRegistry().model


@pytest.mark.unit
def test_model_returns_instance_when_loaded(mock_llama_cpp: MagicMock) -> None:
    registry = ModelRegistry()
    registry.register("base", _cfg())
    registry.load("base")
    assert registry.model is not None


@pytest.mark.unit
def test_vram_lock_acquired_per_load(mock_llama_cpp: MagicMock) -> None:
    """VRAM lock is acquired once per load() call (enforced inside ModelLoader)."""
    registry = ModelRegistry()
    registry.register("a", _cfg("a.gguf"))
    registry.register("b", _cfg("b.gguf"))

    lock_mock = MagicMock()
    lock_mock.__enter__ = MagicMock(return_value=None)
    lock_mock.__exit__ = MagicMock(return_value=False)

    # Patch before constructing the registry so ModelLoader.__init__ binds
    # self._vram_lock to the mock instead of the real threading.Lock.
    with patch("src.llm.model_loader._VRAM_LOCK", lock_mock):
        registry = ModelRegistry()
        registry.register("a", _cfg("a.gguf"))
        registry.register("b", _cfg("b.gguf"))
        registry.load("a")
        registry.load("b")

    assert lock_mock.__enter__.call_count == 2
