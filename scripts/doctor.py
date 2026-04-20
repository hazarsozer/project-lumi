"""
Lumi Doctor — environment diagnostic tool.

Prints a green/red status for every runtime dependency, model file, and
optional extra so you can confirm the environment is ready before a demo.

Usage:
    uv run python scripts/doctor.py
    uv run python scripts/doctor.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import importlib.metadata as _meta
import sys
from pathlib import Path

# Colour codes — degraded gracefully on terminals that don't support ANSI.
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {_GREEN}PASS{_RESET}  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"\n         {detail}" if detail else ""
    print(f"  {_RED}FAIL{_RESET}  {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f"\n         {detail}" if detail else ""
    print(f"  {_YELLOW}WARN{_RESET}  {label}{suffix}")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{title}{_RESET}")


def _check_package(import_name: str, display_name: str, install_hint: str) -> bool:
    try:
        __import__(import_name)
        try:
            version = _meta.version(display_name.split()[0].lower())
            _ok(display_name, f"v{version}")
        except Exception:
            _ok(display_name)
        return True
    except ImportError:
        _fail(display_name, install_hint)
        return False


def _check_file(path_str: str, label: str, hint: str = "") -> bool:
    p = Path(path_str).expanduser()
    if p.is_file():
        size_mb = p.stat().st_size / 1_048_576
        _ok(label, f"{size_mb:.0f} MB  ({p})")
        return True
    _fail(label, hint or f"Not found: {p}")
    return False


def _check_dir(path_str: str, label: str, hint: str = "") -> bool:
    p = Path(path_str).expanduser()
    if p.is_dir():
        _ok(label, f"({p})")
        return True
    _warn(label, hint or f"Not found: {p}  (will attempt download on first use)")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Lumi environment diagnostic")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "After dependency checks, run scripts/smoke_live.py to validate "
            "real model loading and inference (requires model files on disk)."
        ),
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}Lumi Doctor{_RESET}")
    print("=" * 48)

    # Load config (gracefully fall back to defaults if file missing).
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core.config import load_config
    config = load_config(args.config)

    failures = 0

    # -----------------------------------------------------------------------
    # Core runtime packages (always required)
    # -----------------------------------------------------------------------
    _section("Core packages")
    for imp, disp, hint in [
        ("sounddevice", "sounddevice", "uv sync"),
        ("numpy", "numpy", "uv sync"),
        ("openwakeword", "openwakeword", "uv sync"),
        ("faster_whisper", "faster-whisper", "uv sync"),
        ("yaml", "PyYAML", "uv sync"),
    ]:
        if not _check_package(imp, disp, hint):
            failures += 1

    # -----------------------------------------------------------------------
    # LLM extra
    # -----------------------------------------------------------------------
    _section("LLM (--extra llm)")
    if not _check_package("llama_cpp", "llama-cpp-python", "uv sync --extra llm"):
        failures += 1
    if not _check_file(
        config.llm.model_path,
        f"LLM model  ({config.llm.model_path})",
        "Download Phi-3.5-mini GGUF and place at config.yaml → llm.model_path",
    ):
        failures += 1

    # -----------------------------------------------------------------------
    # TTS extra
    # -----------------------------------------------------------------------
    _section(f"TTS (--extra tts)  [enabled={config.tts.enabled}]")
    if config.tts.enabled:
        if not _check_package("kokoro_onnx", "kokoro-onnx", "uv sync --extra tts"):
            failures += 1
        if not _check_file(config.tts.model_path, f"Kokoro model  ({config.tts.model_path})"):
            failures += 1
        if not _check_file(config.tts.voices_path, f"Kokoro voices  ({config.tts.voices_path})"):
            failures += 1
    else:
        _warn("TTS", "disabled in config — skipping checks")

    # -----------------------------------------------------------------------
    # RAG extra
    # -----------------------------------------------------------------------
    _section(f"RAG (--extra rag)  [enabled={config.rag.enabled}]")
    if config.rag.enabled:
        for imp, disp in [
            ("sqlite_vec", "sqlite-vec"),
            ("sentence_transformers", "sentence-transformers"),
            ("pypdf", "pypdf"),
        ]:
            if not _check_package(imp, disp, "uv sync --extra rag"):
                failures += 1
    else:
        _warn("RAG", "disabled in config — skipping checks")

    # -----------------------------------------------------------------------
    # Wake word model
    # -----------------------------------------------------------------------
    _section("Wake word model")
    if not _check_file(
        config.audio.wake_word_model_path,
        f"hey_lumi.onnx  ({config.audio.wake_word_model_path})",
        "Train or download hey_lumi.onnx and set config.yaml → audio.wake_word_model_path",
    ):
        failures += 1

    # -----------------------------------------------------------------------
    # STT model
    # -----------------------------------------------------------------------
    _section("STT model (faster-whisper)")
    _check_dir(
        config.scribe.model_path,
        f"Whisper model dir  ({config.scribe.model_path})",
        "Will auto-download on first use — or pre-download to avoid startup delay",
    )

    # -----------------------------------------------------------------------
    # Vision extra
    # -----------------------------------------------------------------------
    _section(f"Vision  [enabled={config.vision.enabled}]")
    if config.vision.enabled:
        if not _check_file(config.vision.model_path, f"moondream2  ({config.vision.model_path})"):
            failures += 1
    else:
        _warn("Vision", "disabled in config — skipping checks")

    # -----------------------------------------------------------------------
    # Audio hardware
    # -----------------------------------------------------------------------
    _section("Audio hardware")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        device_list = [devices] if isinstance(devices, dict) else list(devices)
        inputs = [d for d in device_list if d.get("max_input_channels", 0) > 0]
        if inputs:
            _ok(f"Microphone", f"{len(inputs)} input device(s) detected")
        else:
            _fail("Microphone", "No input devices found — connect a microphone")
            failures += 1
    except Exception as exc:
        _fail("Microphone", f"sounddevice error: {exc}")
        failures += 1

    # -----------------------------------------------------------------------
    # Live model smoke (--live flag)
    # -----------------------------------------------------------------------
    if args.live:
        import subprocess

        _section("Live model smoke (scripts/smoke_live.py)")
        smoke_script = Path(__file__).parent / "smoke_live.py"
        result = subprocess.run(
            ["uv", "run", "python", str(smoke_script)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"\n  {_RED}{_BOLD}FAIL{_RESET}  smoke_live.py exited with code {result.returncode}")
            failures += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 48}")
    if failures == 0:
        print(f"{_GREEN}{_BOLD}All checks passed — environment is ready.{_RESET}\n")
        return 0
    else:
        print(f"{_RED}{_BOLD}{failures} check(s) failed — see FAIL lines above.{_RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
