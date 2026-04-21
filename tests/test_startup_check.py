"""
Tests for src/core/startup_check.py — Wave A2 promoted hard-failure checks.

Covers:
- _check_openwakeword_version raises on mismatch, passes on match.
- _check_wake_word_model raises when file missing, passes when present.
- _check_stt_model warns (not raises) when directory missing.
- _check_stt_model logs info when directory exists.
- _check_llm_package raises when llama_cpp import fails.
- _check_llm_model raises RuntimeError when model file is missing.
- _check_llm_model passes when model file exists.
- _check_tts_package raises RuntimeError when enabled=True and kokoro-onnx missing.
- _check_tts_package is a no-op when enabled=False.
- _check_tts_package passes when enabled=True and kokoro-onnx is installed.
- _check_tts_model raises RuntimeError when model file is missing.
- _check_tts_model raises RuntimeError when voices file is missing.
- _check_tts_model passes when both files exist.
- _check_rag_packages raises when enabled and packages missing.
- _check_rag_packages is a no-op when disabled.
- _check_rag_packages passes when enabled and all packages present.
- _check_microphone raises when no input devices found.
- _check_microphone passes when at least one input device is found.
- _check_microphone handles dict return (single device) from query_devices.
- run_startup_checks raises TypeError on bad config type.
- run_startup_checks orchestrates all checks and completes successfully.
- run_startup_checks propagates RuntimeError from any hard-failure check.
- run_startup_checks skips _check_tts_model when TTS is disabled.
- run_startup_checks calls _check_tts_model when TTS is enabled.
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


# ---------------------------------------------------------------------------
# _check_openwakeword_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_oww_version_raises_on_mismatch():
    """_check_openwakeword_version raises when the installed version is wrong."""
    from src.core.startup_check import _check_openwakeword_version

    with patch("importlib.metadata.version", return_value="0.6.0"):
        with pytest.raises(RuntimeError, match="version mismatch"):
            _check_openwakeword_version()


@pytest.mark.unit
def test_check_oww_version_passes_on_correct_version():
    """_check_openwakeword_version does not raise when version matches."""
    from src.core.startup_check import _check_openwakeword_version, _REQUIRED_OWW_VERSION

    with patch("importlib.metadata.version", return_value=_REQUIRED_OWW_VERSION):
        _check_openwakeword_version()  # must not raise


@pytest.mark.unit
def test_check_oww_version_raises_on_import_error():
    """_check_openwakeword_version raises when importlib.metadata fails."""
    from src.core.startup_check import _check_openwakeword_version

    with patch("importlib.metadata.version", side_effect=Exception("not found")):
        with pytest.raises(RuntimeError, match="Cannot determine"):
            _check_openwakeword_version()


# ---------------------------------------------------------------------------
# _check_wake_word_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_wake_word_model_raises_when_missing(tmp_path):
    """_check_wake_word_model raises when the ONNX file is absent."""
    from src.core.startup_check import _check_wake_word_model

    with pytest.raises(RuntimeError, match="Wake word model not found"):
        _check_wake_word_model(str(tmp_path / "hey_lumi.onnx"))


@pytest.mark.unit
def test_check_wake_word_model_passes_when_present(tmp_path):
    """_check_wake_word_model does not raise when the ONNX file exists."""
    from src.core.startup_check import _check_wake_word_model

    model = tmp_path / "hey_lumi.onnx"
    model.write_bytes(b"\x00" * 8)
    _check_wake_word_model(str(model))  # must not raise


# ---------------------------------------------------------------------------
# _check_stt_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_stt_model_warns_but_does_not_raise_when_missing(tmp_path, caplog):
    """_check_stt_model warns (not raises) when the STT model directory is absent."""
    import logging
    from src.core.startup_check import _check_stt_model

    missing = str(tmp_path / "whisper-model")
    with caplog.at_level(logging.WARNING, logger="src.core.startup_check"):
        _check_stt_model(missing)  # must not raise

    assert any("STT model" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _check_llm_package
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_llm_package_raises_when_missing():
    """_check_llm_package raises RuntimeError when llama_cpp cannot be imported."""
    from src.core.startup_check import _check_llm_package

    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            _check_llm_package()


@pytest.mark.unit
def test_check_llm_package_passes_when_installed():
    """_check_llm_package does not raise when llama_cpp is importable."""
    from src.core.startup_check import _check_llm_package

    fake = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": fake}):
        _check_llm_package()  # must not raise


# ---------------------------------------------------------------------------
# _check_rag_packages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_rag_packages_noop_when_disabled():
    """_check_rag_packages is silent when RAG is disabled."""
    from src.core.startup_check import _check_rag_packages

    with patch.dict("sys.modules", {"sqlite_vec": None, "sentence_transformers": None}):
        _check_rag_packages(enabled=False)  # must not raise


@pytest.mark.unit
def test_check_rag_packages_raises_when_enabled_and_missing():
    """_check_rag_packages raises when RAG is enabled and packages are absent."""
    from src.core.startup_check import _check_rag_packages

    with patch.dict("sys.modules", {"sqlite_vec": None, "sentence_transformers": None, "pypdf": None}):
        with pytest.raises(RuntimeError, match="RAG packages are missing"):
            _check_rag_packages(enabled=True)


# ---------------------------------------------------------------------------
# _check_microphone
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_microphone_raises_when_no_input_devices():
    """_check_microphone raises RuntimeError when no input devices are detected."""
    from src.core.startup_check import _check_microphone

    with patch("sounddevice.query_devices", return_value=[]):
        with pytest.raises(RuntimeError, match="No microphone"):
            _check_microphone()


@pytest.mark.unit
def test_check_microphone_passes_when_input_device_present():
    """_check_microphone does not raise when at least one input device exists."""
    from src.core.startup_check import _check_microphone

    fake_device = {"max_input_channels": 2, "name": "USB Mic"}
    with patch("sounddevice.query_devices", return_value=[fake_device]):
        _check_microphone()  # must not raise


@pytest.mark.unit
def test_check_microphone_raises_on_portaudio_error():
    """_check_microphone raises RuntimeError when sounddevice itself throws."""
    from src.core.startup_check import _check_microphone

    with patch("sounddevice.query_devices", side_effect=Exception("no portaudio")):
        with pytest.raises(RuntimeError, match="Failed to query audio devices"):
            _check_microphone()


@pytest.mark.unit
def test_check_microphone_handles_single_dict_device():
    """_check_microphone accepts a single dict (not list) from query_devices.

    sounddevice.query_devices() returns a dict when there is exactly one
    device on the system.  The branch at line 265 converts it to a list so
    that the rest of the logic can iterate uniformly.
    """
    from src.core.startup_check import _check_microphone

    single_device = {"max_input_channels": 1, "name": "Built-in Mic"}
    with patch("sounddevice.query_devices", return_value=single_device):
        _check_microphone()  # must not raise


# ---------------------------------------------------------------------------
# _check_stt_model — happy path (directory exists)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_stt_model_logs_info_when_directory_exists(tmp_path, caplog):
    """_check_stt_model logs INFO (not WARNING) when the STT directory exists."""
    import logging
    from src.core.startup_check import _check_stt_model

    stt_dir = tmp_path / "whisper-tiny"
    stt_dir.mkdir()

    with caplog.at_level(logging.INFO, logger="src.core.startup_check"):
        _check_stt_model(str(stt_dir))

    assert any("STT model directory found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _check_rag_packages — happy path (all packages present when enabled)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_rag_packages_passes_when_enabled_and_all_present():
    """_check_rag_packages does not raise when RAG is enabled and all packages importable."""
    from src.core.startup_check import _check_rag_packages

    fake_sqlite_vec = MagicMock()
    fake_sentence_transformers = MagicMock()
    fake_pypdf = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "sqlite_vec": fake_sqlite_vec,
            "sentence_transformers": fake_sentence_transformers,
            "pypdf": fake_pypdf,
        },
    ):
        _check_rag_packages(enabled=True)  # must not raise


# ---------------------------------------------------------------------------
# run_startup_checks — orchestrator
# ---------------------------------------------------------------------------


def _make_fake_lumi_config(
    tmp_path,
    *,
    tts_enabled: bool = False,
    rag_enabled: bool = False,
) -> "MagicMock":
    """Return a MagicMock shaped like LumiConfig with realistic path attributes.

    All model paths default to non-existent paths unless the caller provides
    real tmp_path children.  The individual _check_* functions are mocked at
    the orchestrator level, so path existence does not matter for these tests.
    """
    from src.core.config import LumiConfig, AudioConfig, ScribeConfig, LLMConfig, TTSConfig, RAGConfig

    cfg = MagicMock(spec=LumiConfig)
    cfg.audio = MagicMock(spec=AudioConfig)
    cfg.audio.wake_word_model_path = str(tmp_path / "hey_lumi.onnx")
    cfg.scribe = MagicMock(spec=ScribeConfig)
    cfg.scribe.model_path = str(tmp_path / "whisper-tiny")
    cfg.llm = MagicMock(spec=LLMConfig)
    cfg.llm.model_path = str(tmp_path / "phi.gguf")
    cfg.tts = MagicMock(spec=TTSConfig)
    cfg.tts.enabled = tts_enabled
    cfg.tts.model_path = str(tmp_path / "kokoro.onnx")
    cfg.tts.voices_path = str(tmp_path / "voices.bin")
    cfg.rag = MagicMock(spec=RAGConfig)
    cfg.rag.enabled = rag_enabled
    return cfg


@pytest.mark.unit
def test_run_startup_checks_raises_type_error_on_bad_input():
    """run_startup_checks raises TypeError when passed a non-LumiConfig object."""
    from src.core.startup_check import run_startup_checks

    with pytest.raises(TypeError, match="run_startup_checks expects LumiConfig"):
        run_startup_checks("not-a-config")  # type: ignore[arg-type]


@pytest.mark.unit
def test_run_startup_checks_raises_type_error_on_none():
    """run_startup_checks raises TypeError when passed None."""
    from src.core.startup_check import run_startup_checks

    with pytest.raises(TypeError, match="run_startup_checks expects LumiConfig"):
        run_startup_checks(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_run_startup_checks_completes_successfully_tts_disabled(tmp_path):
    """run_startup_checks runs all checks without error when TTS and RAG are disabled."""
    from src.core.startup_check import run_startup_checks

    cfg = _make_fake_lumi_config(tmp_path, tts_enabled=False, rag_enabled=False)

    with (
        patch("src.core.startup_check._check_openwakeword_version"),
        patch("src.core.startup_check._check_wake_word_model"),
        patch("src.core.startup_check._check_microphone"),
        patch("src.core.startup_check._check_llm_package"),
        patch("src.core.startup_check._check_stt_model"),
        patch("src.core.startup_check._check_llm_model"),
        patch("src.core.startup_check._check_tts_package"),
        patch("src.core.startup_check._check_tts_model") as mock_tts_model,
        patch("src.core.startup_check._check_rag_packages"),
    ):
        run_startup_checks(cfg)
        # TTS disabled — _check_tts_model must NOT be called.
        mock_tts_model.assert_not_called()


@pytest.mark.unit
def test_run_startup_checks_calls_tts_model_check_when_tts_enabled(tmp_path):
    """run_startup_checks calls _check_tts_model when config.tts.enabled is True."""
    from src.core.startup_check import run_startup_checks

    cfg = _make_fake_lumi_config(tmp_path, tts_enabled=True, rag_enabled=False)

    with (
        patch("src.core.startup_check._check_openwakeword_version"),
        patch("src.core.startup_check._check_wake_word_model"),
        patch("src.core.startup_check._check_microphone"),
        patch("src.core.startup_check._check_llm_package"),
        patch("src.core.startup_check._check_stt_model"),
        patch("src.core.startup_check._check_llm_model"),
        patch("src.core.startup_check._check_tts_package"),
        patch("src.core.startup_check._check_tts_model") as mock_tts_model,
        patch("src.core.startup_check._check_rag_packages"),
    ):
        run_startup_checks(cfg)
        mock_tts_model.assert_called_once_with(cfg.tts.model_path, cfg.tts.voices_path)


@pytest.mark.unit
def test_run_startup_checks_propagates_oww_error(tmp_path):
    """run_startup_checks lets RuntimeError from _check_openwakeword_version propagate."""
    from src.core.startup_check import run_startup_checks

    cfg = _make_fake_lumi_config(tmp_path)

    with patch(
        "src.core.startup_check._check_openwakeword_version",
        side_effect=RuntimeError("version mismatch"),
    ):
        with pytest.raises(RuntimeError, match="version mismatch"):
            run_startup_checks(cfg)


@pytest.mark.unit
def test_run_startup_checks_calls_each_check_with_correct_args(tmp_path):
    """run_startup_checks passes the right config fields to every sub-check."""
    from src.core.startup_check import run_startup_checks

    cfg = _make_fake_lumi_config(tmp_path, tts_enabled=False, rag_enabled=False)

    with (
        patch("src.core.startup_check._check_openwakeword_version") as mock_oww,
        patch("src.core.startup_check._check_wake_word_model") as mock_ww,
        patch("src.core.startup_check._check_microphone") as mock_mic,
        patch("src.core.startup_check._check_llm_package") as mock_llm_pkg,
        patch("src.core.startup_check._check_stt_model") as mock_stt,
        patch("src.core.startup_check._check_llm_model") as mock_llm,
        patch("src.core.startup_check._check_tts_package") as mock_tts_pkg,
        patch("src.core.startup_check._check_tts_model"),
        patch("src.core.startup_check._check_rag_packages") as mock_rag,
    ):
        run_startup_checks(cfg)

    mock_oww.assert_called_once_with()
    mock_ww.assert_called_once_with(cfg.audio.wake_word_model_path)
    mock_mic.assert_called_once_with()
    mock_llm_pkg.assert_called_once_with()
    mock_stt.assert_called_once_with(cfg.scribe.model_path)
    mock_llm.assert_called_once_with(cfg.llm.model_path)
    mock_tts_pkg.assert_called_once_with(cfg.tts.enabled)
    mock_rag.assert_called_once_with(cfg.rag.enabled)
