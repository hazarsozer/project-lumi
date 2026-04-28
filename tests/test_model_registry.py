"""Tests for src.llm.model_registry.ModelRegistry."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.core.config import LLMConfig
from src.llm.model_registry import AdapterSpec, ModelRegistry


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


# ---------------------------------------------------------------------------
# AdapterSpec tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adapter_spec_fields() -> None:
    """AdapterSpec stores persona, task, lora_path, and lora_scale correctly."""
    spec = AdapterSpec(persona="assistant", task="coding", lora_path="lora/code.gguf", lora_scale=0.9)
    assert spec.persona == "assistant"
    assert spec.task == "coding"
    assert spec.lora_path == "lora/code.gguf"
    assert spec.lora_scale == 0.9


@pytest.mark.unit
def test_adapter_spec_default_scale() -> None:
    """AdapterSpec.lora_scale defaults to 1.0 when omitted."""
    spec = AdapterSpec(persona="assistant", task=None, lora_path="lora/base.gguf")
    assert spec.lora_scale == 1.0


@pytest.mark.unit
def test_adapter_spec_is_frozen() -> None:
    """AdapterSpec is a frozen dataclass — mutation must raise FrozenInstanceError."""
    spec = AdapterSpec(persona="assistant", task=None, lora_path="lora/base.gguf")
    with pytest.raises(Exception):
        spec.lora_scale = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# register_adapter / resolve tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_and_resolve_exact(mock_llama_cpp: MagicMock) -> None:
    """register_adapter + resolve roundtrip returns the same spec."""
    registry = ModelRegistry()
    spec = AdapterSpec(persona="assistant", task="coding", lora_path="lora/code.gguf")
    registry.register_adapter(spec)
    result = registry.resolve("assistant", "coding")
    assert result == spec


@pytest.mark.unit
def test_resolve_falls_back_to_persona_none(mock_llama_cpp: MagicMock) -> None:
    """resolve falls back to (persona, None) when exact (persona, task) is absent."""
    registry = ModelRegistry()
    fallback = AdapterSpec(persona="assistant", task=None, lora_path="lora/base.gguf")
    registry.register_adapter(fallback)
    # Ask for a specific task that was never registered.
    result = registry.resolve("assistant", "summarise")
    assert result == fallback


@pytest.mark.unit
def test_resolve_exact_preferred_over_fallback(mock_llama_cpp: MagicMock) -> None:
    """When both an exact match and a (persona, None) fallback exist,
    the exact match is returned."""
    registry = ModelRegistry()
    fallback = AdapterSpec(persona="assistant", task=None, lora_path="lora/base.gguf")
    exact = AdapterSpec(persona="assistant", task="coding", lora_path="lora/code.gguf")
    registry.register_adapter(fallback)
    registry.register_adapter(exact)
    result = registry.resolve("assistant", "coding")
    assert result == exact


@pytest.mark.unit
def test_resolve_returns_none_when_no_match(mock_llama_cpp: MagicMock) -> None:
    """resolve returns None when no entry exists for the persona."""
    registry = ModelRegistry()
    assert registry.resolve("unknown_persona") is None


@pytest.mark.unit
def test_resolve_returns_none_when_persona_missing_task_fallback(mock_llama_cpp: MagicMock) -> None:
    """resolve returns None when only a different persona is registered."""
    registry = ModelRegistry()
    registry.register_adapter(AdapterSpec(persona="other", task=None, lora_path="x.gguf"))
    assert registry.resolve("assistant") is None


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
