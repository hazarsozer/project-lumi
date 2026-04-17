"""
Startup validation for Project Lumi.

run_startup_checks() is called once during application boot (before any
subsystem is initialized) to surface missing dependencies and model files
early with clear, human-readable error messages.

Failure policy:
    HARD failures (raise RuntimeError — caller decides whether to abort):
        - openwakeword version mismatch
        - Wake word ONNX model file not found
        - No microphone device detected
        - llama-cpp-python not installed (required for LLM inference)
        - RAG packages missing when config.rag.enabled is True

    SOFT failures (log WARNING — application can continue degraded):
        - STT model directory not found
        - LLM model file not found  (model is optional until PROCESSING)
        - kokoro-onnx not installed when config.tts.enabled is True
        - TTS model or voices file not found

Constraints:
    - No print() calls — all output via logging.getLogger(__name__)
    - No sys.exit() — raise RuntimeError and let main.py decide
    - No imports from src/audio/, src/llm/, or src/interface/
"""

from __future__ import annotations

import importlib.metadata as _meta
import logging
from pathlib import Path

from src.core.config import LumiConfig

logger = logging.getLogger(__name__)

# Exact openwakeword version required by the monkey-patch in ears.py.
# 0.6.0 has no Python 3.12 wheels; any other version breaks the patch.
_REQUIRED_OWW_VERSION: str = "0.4.0"


def _check_openwakeword_version() -> None:
    """Raise RuntimeError if the installed openwakeword version is wrong.

    The monkey-patch applied in ears.py targets the internal API of exactly
    version 0.4.0.  A version mismatch would cause silent misbehaviour or
    a hard crash at runtime rather than a clear error at startup.
    """
    try:
        installed = _meta.version("openwakeword")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot determine installed openwakeword version: {exc}\n"
            "Install the required version with:\n"
            f"  uv add openwakeword=={_REQUIRED_OWW_VERSION}"
        ) from exc

    if installed != _REQUIRED_OWW_VERSION:
        raise RuntimeError(
            f"openwakeword version mismatch: found '{installed}', "
            f"required '{_REQUIRED_OWW_VERSION}'.\n"
            "The ears.py monkey-patch only targets this exact version.\n"
            "Fix with:\n"
            f"  uv add openwakeword=={_REQUIRED_OWW_VERSION}"
        )

    logger.info(
        "openwakeword version check passed: %s == %s",
        installed,
        _REQUIRED_OWW_VERSION,
    )


def _check_wake_word_model(model_path: str) -> None:
    """Raise RuntimeError if the wake word ONNX file is absent.

    Without this file the wake word detector cannot start at all, so this
    is a hard failure regardless of edition.
    """
    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(
            f"Wake word model not found: '{model_path}'\n"
            "Place the hey_lumi.onnx file at the expected path or update\n"
            "config.yaml → audio.wake_word_model_path."
        )

    logger.info("Wake word model found: %s", model_path)


def _check_stt_model(model_path: str) -> None:
    """Log a warning if the STT model directory is missing.

    faster-whisper can download the model on first use, so a missing local
    directory is not a hard failure — it will cause a slow first startup.
    """
    path = Path(model_path)
    if not path.is_dir():
        logger.warning(
            "STT model directory not found: '%s'. "
            "faster-whisper will attempt to download the model on first use. "
            "Pre-download with: "
            "uv run python -c \"from faster_whisper import WhisperModel; "
            "WhisperModel('%s', device='cpu', compute_type='int8')\"",
            model_path,
            model_path,
        )
    else:
        logger.info("STT model directory found: %s", model_path)


def _check_llm_model(model_path: str) -> None:
    """Log a warning if the LLM GGUF file is missing.

    The LLM is only loaded during PROCESSING (not at IDLE), so a missing
    model file is acceptable at startup — the assistant can still do wake
    word detection and STT.
    """
    path = Path(model_path)
    if not path.is_file():
        logger.warning(
            "LLM model file not found: '%s'. "
            "The assistant will not be able to generate responses until "
            "the model is present. Place the GGUF file at the configured "
            "path or update config.yaml → llm.model_path.",
            model_path,
        )
    else:
        logger.info("LLM model file found: %s", model_path)


def _check_tts_model(model_path: str, voices_path: str) -> None:
    """Log a warning if TTS model files are missing.

    KokoroTTS falls back to silent mode when model files are absent, so
    this is a soft failure — the assistant can still wake, transcribe, and
    route commands without TTS playback.
    """
    model = Path(model_path)
    voices = Path(voices_path)

    if not model.is_file():
        logger.warning(
            "TTS model file not found: '%s'. "
            "KokoroTTS will run in silent mode until the file is present. "
            "Download from the kokoro-onnx releases page and place at the "
            "configured path, or update config.yaml → tts.model_path.",
            model_path,
        )
    else:
        logger.info("TTS model file found: %s", model_path)

    if not voices.is_file():
        logger.warning(
            "TTS voices file not found: '%s'. "
            "KokoroTTS will run in silent mode until the file is present. "
            "Download voices.bin from the kokoro-onnx releases page and "
            "place at the configured path, or update config.yaml → tts.voices_path.",
            voices_path,
        )
    else:
        logger.info("TTS voices file found: %s", voices_path)


def _check_llm_package() -> None:
    """Raise RuntimeError if llama-cpp-python is not installed.

    llama-cpp-python is required for all LLM inference.  It lives in the
    optional ``llm`` extra and is not installed by a plain ``uv sync``.
    Failing here gives the user a clear remediation path before the
    pipeline reaches the LLM load step.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "llama-cpp-python is not installed but is required for LLM inference.\n"
            "Install with:\n"
            "  uv sync --extra llm\n"
            "Note: building llama-cpp-python requires CMake and a C++ compiler.\n"
            "  sudo apt install cmake build-essential  # Debian/Ubuntu"
        )

    logger.info("llama-cpp-python package check passed.")


def _check_tts_package(enabled: bool) -> None:
    """Log a warning if kokoro-onnx is missing and TTS is enabled.

    TTS is treated as a soft dependency — the assistant can still wake,
    transcribe, and route commands without audio playback.  Only emit the
    warning when the user has explicitly set ``config.tts.enabled = True``
    so that users running in silent/test mode are not warned unnecessarily.
    """
    if not enabled:
        return
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        logger.warning(
            "kokoro-onnx is not installed but config.tts.enabled is True. "
            "TTS playback will be unavailable. "
            "Install with: uv sync --extra tts"
        )
        return

    logger.info("kokoro-onnx package check passed.")


def _check_rag_packages(enabled: bool) -> None:
    """Raise RuntimeError if RAG packages are missing and RAG is enabled.

    sqlite-vec, sentence-transformers, and pypdf are grouped under the
    optional ``rag`` extra.  When ``config.rag.enabled`` is False the check
    is skipped entirely so users who have not installed the extra are not
    affected at all.  When enabled, all three must be present or the
    RAGRetriever will fail at init time anyway — better to surface the full
    list upfront.
    """
    if not enabled:
        return

    missing: list[str] = []
    for pkg, import_name in [
        ("sqlite-vec (uv sync --extra rag)", "sqlite_vec"),
        ("sentence-transformers (uv sync --extra rag)", "sentence_transformers"),
        ("pypdf (uv sync --extra rag)", "pypdf"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise RuntimeError(
            "config.rag.enabled is True but the following RAG packages are missing:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nInstall with:\n  uv sync --extra rag"
        )

    logger.info("RAG package check passed: all required packages present.")


def _check_microphone() -> None:
    """Raise RuntimeError if no input (microphone) device is available.

    Uses sounddevice to query the host audio API.  Any exception from
    sounddevice (PortAudio not found, no devices, etc.) is treated as a
    hard failure because the entire pipeline depends on microphone input.
    """
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query audio devices: {exc}\n"
            "Ensure PortAudio is installed and a microphone is connected.\n"
            "On Debian/Ubuntu: sudo apt install portaudio19-dev"
        ) from exc

    # query_devices() returns either a dict (single device) or a list.
    if isinstance(devices, dict):
        device_list = [devices]
    else:
        device_list = list(devices)

    input_devices = [d for d in device_list if d.get("max_input_channels", 0) > 0]

    if not input_devices:
        raise RuntimeError(
            "No microphone (input audio device) detected.\n"
            "Connect a microphone and ensure the operating system grants\n"
            "application access to audio input."
        )

    logger.info(
        "Microphone check passed: %d input device(s) available.",
        len(input_devices),
    )


def run_startup_checks(config: LumiConfig) -> None:
    """Run all startup validation checks for Project Lumi.

    Args:
        config: The fully loaded LumiConfig instance produced by
                load_config().

    Raises:
        TypeError: If *config* is not a LumiConfig instance.
        RuntimeError: On any hard failure (see module docstring for the
                      complete list).  The error message is human-readable
                      and includes remediation instructions.
    """
    if not isinstance(config, LumiConfig):
        raise TypeError(
            f"run_startup_checks expects LumiConfig, got {type(config).__name__}"
        )

    logger.info("--- Project Lumi startup checks ---")

    # Hard failures — raise immediately on error.
    _check_openwakeword_version()
    _check_wake_word_model(config.audio.wake_word_model_path)
    _check_microphone()
    _check_llm_package()

    # Soft failures — warn but continue.
    _check_stt_model(config.scribe.model_path)
    _check_llm_model(config.llm.model_path)
    _check_tts_package(config.tts.enabled)
    if config.tts.enabled:
        _check_tts_model(config.tts.model_path, config.tts.voices_path)

    # Hard failure when RAG feature is explicitly enabled without its extras.
    _check_rag_packages(config.rag.enabled)

    logger.info("--- Startup checks complete ---")
