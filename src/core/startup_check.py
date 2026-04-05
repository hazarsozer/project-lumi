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

    SOFT failures (log WARNING — application can continue degraded):
        - STT model directory not found
        - LLM model file not found  (model is optional until PROCESSING)

Constraints:
    - No print() calls — all output via logging.getLogger(__name__)
    - No sys.exit() — raise RuntimeError and let main.py decide
    - No imports from src/audio/, src/llm/, or src/interface/
"""

from __future__ import annotations

import logging
from pathlib import Path

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
        import importlib.metadata as meta

        installed = meta.version("openwakeword")
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


def run_startup_checks(config: "LumiConfig") -> None:  # noqa: F821
    """Run all startup validation checks for Project Lumi.

    Args:
        config: The fully loaded LumiConfig instance produced by
                load_config().

    Raises:
        RuntimeError: On any hard failure (see module docstring for the
                      complete list).  The error message is human-readable
                      and includes remediation instructions.
    """
    # Import here (not at module level) to avoid the circular-import
    # constraint — startup_check.py must not import config.py at module
    # level if config.py might import startup_check.py in the future.
    # Using a forward-reference string annotation on the parameter type
    # keeps mypy happy without a runtime import.
    from src.core.config import LumiConfig  # noqa: PLC0415

    assert isinstance(config, LumiConfig), (
        f"run_startup_checks expects LumiConfig, got {type(config).__name__}"
    )

    logger.info("--- Project Lumi startup checks ---")

    # Hard failures — raise immediately on error.
    _check_openwakeword_version()
    _check_wake_word_model(config.audio.wake_word_model_path)
    _check_microphone()

    # Soft failures — warn but continue.
    _check_stt_model(config.scribe.model_path)
    _check_llm_model(config.llm.model_path)

    logger.info("--- Startup checks complete ---")
