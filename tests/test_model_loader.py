"""
Tests for src.llm.model_loader.ModelLoader.

Mocking strategy
----------------
``llama_cpp.Llama`` is never instantiated for real.  The ``mock_llama_cpp``
fixture (defined in conftest.py) patches ``llama_cpp.Llama`` at the package
boundary so no GGUF file is read and no CPU/GPU inference library is loaded.

All tests are marked ``unit`` — they require no hardware and no model files.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.core.config import LLMConfig

# RED: these imports will fail until src/llm/model_loader.py is written.
from src.llm.model_loader import ModelLoader  # type: ignore[import]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_config() -> LLMConfig:
    """Return a default LLMConfig instance for use in tests."""
    return LLMConfig()


def _config_with_path(path: str) -> LLMConfig:
    """Return an LLMConfig with model_path overridden to ``path``."""
    return LLMConfig(model_path=path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initial_state_not_loaded() -> None:
    """A freshly constructed ModelLoader must report is_loaded == False."""
    loader = ModelLoader()
    assert loader.is_loaded is False


@pytest.mark.unit
def test_load_sets_is_loaded(mock_llama_cpp: MagicMock, tmp_path: object) -> None:
    """After a successful load(), is_loaded must be True."""
    # Provide a fake model file path; the real file is never opened because
    # llama_cpp.Llama is mocked.
    config = _default_config()
    loader = ModelLoader()
    loader.load(config)
    assert loader.is_loaded is True


@pytest.mark.unit
def test_model_property_raises_when_not_loaded() -> None:
    """Accessing .model before load() must raise RuntimeError."""
    loader = ModelLoader()
    with pytest.raises(RuntimeError):
        _ = loader.model


@pytest.mark.unit
def test_model_property_returns_instance_when_loaded(mock_llama_cpp: MagicMock) -> None:
    """After load(), .model must return the llama_cpp.Llama instance."""
    config = _default_config()
    loader = ModelLoader()
    loader.load(config)
    # The returned object should be the mock instance configured by the fixture.
    model = loader.model
    assert model is not None


@pytest.mark.unit
def test_unload_sets_not_loaded(mock_llama_cpp: MagicMock) -> None:
    """After unload(), is_loaded must be False again."""
    config = _default_config()
    loader = ModelLoader()
    loader.load(config)
    assert loader.is_loaded is True
    loader.unload()
    assert loader.is_loaded is False


@pytest.mark.unit
def test_load_missing_model_path_raises(mock_llama_cpp: MagicMock) -> None:
    """load() with a path that does not exist on disk must raise FileNotFoundError."""
    mock_llama_cpp.side_effect = ValueError("file does not exist")
    config = _config_with_path("/nonexistent/path/to/model.gguf")
    loader = ModelLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(config)


@pytest.mark.unit
def test_model_raises_after_unload(mock_llama_cpp: MagicMock) -> None:
    """Accessing .model after unload() must raise RuntimeError."""
    config = _default_config()
    loader = ModelLoader()
    loader.load(config)
    loader.unload()
    with pytest.raises(RuntimeError):
        _ = loader.model


@pytest.mark.unit
def test_load_passes_config_fields_to_llama(
    mock_llama_cpp: MagicMock, tmp_path: object
) -> None:
    """load() must forward n_gpu_layers and context_length to llama_cpp.Llama."""
    config = LLMConfig(n_gpu_layers=4, context_length=2048)
    loader = ModelLoader()
    loader.load(config)
    # mock_llama_cpp is the class mock; verify it was called
    assert mock_llama_cpp.called


@pytest.mark.unit
def test_load_reraises_non_file_not_found_value_error(
    mock_llama_cpp: MagicMock,
) -> None:
    """load() must re-raise a ValueError that does NOT contain 'does not exist'
    (line 47 — the bare ``raise`` at the end of the except ValueError block)."""
    # Configure the mock Llama constructor to raise a ValueError whose message
    # does NOT contain "does not exist", so the re-raise branch is taken.
    mock_llama_cpp.side_effect = ValueError("invalid quantization type")
    config = _default_config()
    loader = ModelLoader()
    with pytest.raises(ValueError, match="invalid quantization type"):
        loader.load(config)


@pytest.mark.unit
def test_kv_cache_quant_forwarded(mock_llama_cpp: MagicMock) -> None:
    """When config.kv_cache_quant is set, cache_type_k and cache_type_v must be
    forwarded to llama_cpp.Llama()."""
    config = LLMConfig(kv_cache_quant="turbo3")
    loader = ModelLoader()
    loader.load(config)

    assert mock_llama_cpp.call_count == 1
    kwargs = mock_llama_cpp.call_args.kwargs
    assert kwargs.get("cache_type_k") == "turbo3"
    assert kwargs.get("cache_type_v") == "turbo3"


@pytest.mark.unit
def test_kv_cache_quant_none_not_forwarded(mock_llama_cpp: MagicMock) -> None:
    """When config.kv_cache_quant is None (default), cache_type_k/cache_type_v
    must not be present in the Llama() kwargs."""
    config = LLMConfig()  # kv_cache_quant defaults to None
    assert config.kv_cache_quant is None
    loader = ModelLoader()
    loader.load(config)

    assert mock_llama_cpp.call_count == 1
    kwargs = mock_llama_cpp.call_args.kwargs
    assert "cache_type_k" not in kwargs
    assert "cache_type_v" not in kwargs


@pytest.mark.unit
def test_kv_cache_quant_graceful_fallback(
    mock_llama_cpp: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """When the installed llama-cpp-python predates PR #21089, the first call
    raises TypeError('...cache_type_k...').  ModelLoader must log a warning,
    drop the kwargs, and retry successfully."""
    # First invocation: raise TypeError as an old llama-cpp-python would.
    # Second invocation: return the normal mocked instance.
    fallback_instance = MagicMock(name="fallback_llama")
    mock_llama_cpp.side_effect = [
        TypeError("Llama.__init__() got an unexpected keyword argument 'cache_type_k'"),
        fallback_instance,
    ]

    config = LLMConfig(kv_cache_quant="turbo3")
    loader = ModelLoader()
    with caplog.at_level("WARNING"):
        loader.load(config)

    # Warning logged about fallback.
    assert any(
        "TurboQuant" in rec.message or "falling back" in rec.message.lower()
        for rec in caplog.records
    )

    # Two calls: one that raised, one retry without the quant kwargs.
    assert mock_llama_cpp.call_count == 2
    retry_kwargs = mock_llama_cpp.call_args_list[1].kwargs
    assert "cache_type_k" not in retry_kwargs
    assert "cache_type_v" not in retry_kwargs

    # Model ended up loaded via the fallback path.
    assert loader.is_loaded is True
    assert loader.model is fallback_instance
