"""
Tests for src/core/startup_check.py — Wave A2 promoted hard-failure checks.

Covers:
- _check_llm_model raises RuntimeError when model file is missing.
- _check_llm_model passes when model file exists.
- _check_tts_package raises RuntimeError when enabled=True and kokoro-onnx missing.
- _check_tts_package is a no-op when enabled=False.
- _check_tts_package passes when enabled=True and kokoro-onnx is installed.
- _check_tts_model raises RuntimeError when model file is missing (TTS enabled).
- _check_tts_model raises RuntimeError when voices file is missing (TTS enabled).
- _check_tts_model passes when both files exist.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# _check_llm_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_llm_model_raises_when_file_missing(tmp_path):
    """_check_llm_model raises RuntimeError when the GGUF file is absent."""
    from src.core.startup_check import _check_llm_model

    missing = str(tmp_path / "model.gguf")
    with pytest.raises(RuntimeError, match="not found"):
        _check_llm_model(missing)


@pytest.mark.unit
def test_check_llm_model_passes_when_file_exists(tmp_path):
    """_check_llm_model does not raise when the GGUF file is present."""
    from src.core.startup_check import _check_llm_model

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"\x00" * 16)
    _check_llm_model(str(model_file))  # must not raise


# ---------------------------------------------------------------------------
# _check_tts_package
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_tts_package_raises_when_enabled_and_missing():
    """_check_tts_package raises RuntimeError when TTS is enabled but kokoro-onnx absent."""
    from src.core.startup_check import _check_tts_package

    with patch.dict("sys.modules", {"kokoro_onnx": None}):
        with pytest.raises(RuntimeError, match="kokoro-onnx"):
            _check_tts_package(enabled=True)


@pytest.mark.unit
def test_check_tts_package_noop_when_disabled():
    """_check_tts_package is silent when TTS is disabled, even if package missing."""
    from src.core.startup_check import _check_tts_package

    with patch.dict("sys.modules", {"kokoro_onnx": None}):
        _check_tts_package(enabled=False)  # must not raise


@pytest.mark.unit
def test_check_tts_package_passes_when_enabled_and_installed():
    """_check_tts_package does not raise when TTS enabled and kokoro-onnx installed."""
    from src.core.startup_check import _check_tts_package

    fake_kokoro = MagicMock()
    with patch.dict("sys.modules", {"kokoro_onnx": fake_kokoro}):
        _check_tts_package(enabled=True)  # must not raise


# ---------------------------------------------------------------------------
# _check_tts_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_tts_model_raises_when_model_missing(tmp_path):
    """_check_tts_model raises RuntimeError when the Kokoro ONNX file is absent."""
    from src.core.startup_check import _check_tts_model

    voices = tmp_path / "voices.bin"
    voices.write_bytes(b"\x00" * 8)
    missing_model = str(tmp_path / "kokoro.onnx")

    with pytest.raises(RuntimeError, match="TTS model file not found"):
        _check_tts_model(missing_model, str(voices))


@pytest.mark.unit
def test_check_tts_model_raises_when_voices_missing(tmp_path):
    """_check_tts_model raises RuntimeError when voices.bin is absent."""
    from src.core.startup_check import _check_tts_model

    model = tmp_path / "kokoro.onnx"
    model.write_bytes(b"\x00" * 8)
    missing_voices = str(tmp_path / "voices.bin")

    with pytest.raises(RuntimeError, match="TTS voices file not found"):
        _check_tts_model(str(model), missing_voices)


@pytest.mark.unit
def test_check_tts_model_passes_when_both_files_exist(tmp_path):
    """_check_tts_model does not raise when both model and voices files exist."""
    from src.core.startup_check import _check_tts_model

    model = tmp_path / "kokoro.onnx"
    voices = tmp_path / "voices.bin"
    model.write_bytes(b"\x00" * 8)
    voices.write_bytes(b"\x00" * 8)

    _check_tts_model(str(model), str(voices))  # must not raise
