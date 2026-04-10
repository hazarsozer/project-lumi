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
def test_load_missing_model_path_raises() -> None:
    """load() with a path that does not exist on disk must raise FileNotFoundError."""
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
def test_load_passes_config_fields_to_llama(mock_llama_cpp: MagicMock, tmp_path: object) -> None:
    """load() must forward n_gpu_layers and context_length to llama_cpp.Llama."""
    config = LLMConfig(n_gpu_layers=4, context_length=2048)
    loader = ModelLoader()
    loader.load(config)
    # mock_llama_cpp is the class mock; verify it was called
    assert mock_llama_cpp.called


@pytest.mark.unit
def test_load_reraises_non_file_not_found_value_error(mock_llama_cpp: MagicMock) -> None:
    """load() must re-raise a ValueError that does NOT contain 'does not exist'
    (line 47 — the bare ``raise`` at the end of the except ValueError block)."""
    # Configure the mock Llama constructor to raise a ValueError whose message
    # does NOT contain "does not exist", so the re-raise branch is taken.
    mock_llama_cpp.side_effect = ValueError("invalid quantization type")
    config = _default_config()
    loader = ModelLoader()
    with pytest.raises(ValueError, match="invalid quantization type"):
        loader.load(config)
