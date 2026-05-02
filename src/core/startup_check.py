"""
Startup validation for Project Lumi.

run_startup_checks() is called once during application boot (before any
subsystem is initialized) to surface missing dependencies and model files
early with clear, human-readable error messages.

Failure policy:
    HARD failures (raise RuntimeError — caller decides whether to abort):
        - openwakeword version mismatch — ONLY when wake_word_enabled=True AND
          the model file is already present (wrong installed version is a real
          deployment bug that must be fixed, not worked around).
        - RAG packages missing when config.rag.enabled is True.

    SOFT failures (log WARNING; returned as list[str] for UI display):
        - Wake word ONNX model file not found
        - No microphone device detected
        - llama-cpp-python not installed
        - STT model directory not found
        - LLM model file not found
        - TTS model or voices file not found

Return value:
    run_startup_checks() returns list[str] — each entry describes a missing
    component with a short installation/download hint.  Empty list means all
    required components are present.

    The orchestrator passes this list into SystemStatusEvent so the frontend
    can show a first-run setup panel.  main.py skips Ears startup when the
    wake-word related items are present (model missing or mic absent).

Constraints:
    - No print() calls — all output via logging.getLogger(__name__)
    - No sys.exit() — raise RuntimeError only for genuine deployment bugs
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

_SETUP_WIZARD_PATH: str = "scripts/setup_wizard.py"


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


def _check_wake_word_model(model_path: str) -> list[str]:
    """Return a non-empty list if the wake-word ONNX file is absent.

    Soft failure: the Brain can start without the model (PTT still works).
    """
    path = Path(model_path)
    if not path.is_file():
        logger.warning("Wake word model not found: '%s'.", model_path)
        return [
            f"Wake-word model not found: {model_path}\n"
            "  → place hey_lumi.onnx at the path above, or set\n"
            "    audio.wake_word_model_path in config.yaml, or disable\n"
            "    wake-word with audio.wake_word_enabled: false and use PTT."
        ]
    logger.info("Wake word model found: %s", model_path)
    return []


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
            'uv run python -c "from faster_whisper import WhisperModel; '
            "WhisperModel('%s', device='cpu', compute_type='int8')\"",
            model_path,
            model_path,
        )
    else:
        logger.info("STT model directory found: %s", model_path)


def _check_llm_model(model_path: str) -> list[str]:
    """Return a non-empty list if the LLM GGUF file is missing, empty otherwise.

    Changed from hard-exit to soft-return so the Brain can start in degraded
    mode and the frontend can display a setup panel with download instructions.
    """
    path = Path(model_path)
    if not path.is_file():
        logger.warning(
            "LLM model file not found: '%s'. "
            "Lumi will start in degraded mode without LLM inference. "
            "Download: huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF "
            "Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir models/llm/",
            model_path,
        )
        return [
            f"LLM model not found: {model_path}\n"
            "  → huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF "
            "Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir models/llm/"
        ]
    logger.info("LLM model file found: %s", model_path)
    return []


def _check_tts_model(model_path: str, voices_path: str) -> list[str]:
    """Return a list of missing TTS file descriptions (empty if all present).

    Changed from hard-exit to soft-return so the Brain can start in degraded
    mode (silent) and the frontend can display a setup panel.
    """
    missing: list[str] = []
    _KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases"

    if not Path(model_path).is_file():
        logger.warning("TTS model file not found: '%s'.", model_path)
        missing.append(
            f"TTS model not found: {model_path}\n"
            f"  → download kokoro-v1_0.onnx from {_KOKORO_URL}"
        )
    else:
        logger.info("TTS model file found: %s", model_path)

    if not Path(voices_path).is_file():
        logger.warning("TTS voices file not found: '%s'.", voices_path)
        missing.append(
            f"TTS voices not found: {voices_path}\n"
            f"  → download voices.bin from {_KOKORO_URL}"
        )
    else:
        logger.info("TTS voices file found: %s", voices_path)

    return missing


def _check_llm_package() -> list[str]:
    """Return a non-empty list if llama-cpp-python is not installed.

    Soft failure: the Brain starts in degraded mode; the setup screen guides
    the user to install the package.
    """
    try:
        import llama_cpp  # noqa: F401
        logger.info("llama-cpp-python package check passed.")
        return []
    except ImportError:
        logger.warning(
            "llama-cpp-python is not installed — LLM inference unavailable."
        )
        return [
            "LLM package not installed: llama-cpp-python\n"
            "  → uv sync --extra llm\n"
            "  Note: requires CMake + C++ compiler\n"
            "    sudo apt install cmake build-essential  # Debian/Ubuntu"
        ]


def _check_tts_package(enabled: bool) -> None:
    """Raise RuntimeError if kokoro-onnx is missing and TTS is enabled.

    When TTS is explicitly enabled in config, a missing kokoro-onnx package
    will cause silent audio failures at runtime.  Hard-failing here ensures
    the user knows exactly what to install before the app appears to work.
    Set config.tts.enabled = false to run in silent/headless mode.
    """
    if not enabled:
        return
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "kokoro-onnx is not installed but config.tts.enabled is True.\n"
            "TTS playback requires the tts extra:\n"
            "  uv sync --extra tts\n"
            "To run without TTS, set config.tts.enabled = false in config.yaml."
        ) from exc

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


def _check_microphone() -> list[str]:
    """Return a non-empty list if no microphone is available.

    Soft failure: the Brain starts without audio capture (PTT + typed text
    still work).
    """
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Failed to query audio devices: %s", exc)
        return [
            f"Microphone check failed: {exc}\n"
            "  → sudo apt install portaudio19-dev  # Debian/Ubuntu\n"
            "  → connect a microphone and grant audio input permission"
        ]

    # query_devices() returns either a dict (single device) or a list.
    if isinstance(devices, dict):
        device_list = [devices]
    else:
        device_list = list(devices)

    input_devices = [d for d in device_list if d.get("max_input_channels", 0) > 0]

    if not input_devices:
        logger.warning("No microphone (input audio device) detected.")
        return [
            "No microphone detected\n"
            "  → connect a microphone and grant audio input permission"
        ]

    logger.info(
        "Microphone check passed: %d input device(s) available.",
        len(input_devices),
    )
    return []


def run_startup_checks(config: LumiConfig) -> list[str]:
    """Run all startup validation checks for Project Lumi.

    Args:
        config: The fully loaded LumiConfig instance produced by
                load_config().

    Returns:
        List of human-readable strings for each missing model file.
        Empty list means everything required is present.

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

    missing: list[str] = []

    # Wake-word pipeline checks — skipped entirely when wake_word_enabled=False.
    # These are now SOFT: missing model/mic means Ears won't start, but the
    # Brain still launches (PTT or typed text remain available).
    if config.audio.wake_word_enabled:
        missing.extend(_check_wake_word_model(config.audio.wake_word_model_path) or [])
        missing.extend(_check_microphone() or [])

        # OWW version check is HARD but only runs when the model IS present
        # (wrong installed version is a real deployment bug, not a first-run
        # issue).  If the model is absent we already collected it above and
        # Ears won't start, so there's no point validating the package version.
        wake_word_ok = not any(
            "wake" in item.lower() or "microphone" in item.lower()
            for item in missing
        )
        if wake_word_ok:
            _check_openwakeword_version()  # raises RuntimeError on version mismatch

    # LLM package — soft: Brain starts without LLM; setup screen guides install.
    missing.extend(_check_llm_package() or [])

    # STT model directory — soft (faster-whisper downloads on first use).
    _check_stt_model(config.scribe.model_path)

    # LLM + TTS model files — soft.
    missing.extend(_check_llm_model(config.llm.model_path) or [])
    _check_tts_package(config.tts.enabled)
    if config.tts.enabled:
        missing.extend(_check_tts_model(config.tts.model_path, config.tts.voices_path) or [])

    # RAG packages — hard when explicitly enabled without its extras.
    _check_rag_packages(config.rag.enabled)

    if missing:
        logger.warning(
            "Setup required: %d component(s) missing. "
            "Lumi will run in degraded mode.",
            len(missing),
        )
    logger.info("--- Startup checks complete ---")
    return missing
