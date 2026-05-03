# brain.spec — PyInstaller spec for the Lumi Brain sidecar binary.
#
# IMPORTANT CONSTRAINTS
# =====================
# openwakeword==0.4.0 MUST be exact. ears.py monkey-patches internal symbols
# that changed in 0.5.x / 0.6.x.  The startup check enforces this at runtime;
# PyInstaller freezes whatever version is installed, so always run `uv sync`
# before building.
#
# Build command:
#   uv run pyinstaller scripts/brain.spec \
#       --distpath scripts/dist \
#       --workpath scripts/build \
#       --noconfirm
#
# Output: scripts/dist/lumi-brain/lumi-brain  (--onedir mode)
#
# --onedir is preferred over --onefile because:
#   - llama-cpp-python ships CUDA .so files that must remain as separate files.
#   - openwakeword ONNX model files co-locate cleanly.
#   - AppImage bundles the whole directory anyway.
#
# After building, run scripts/build_brain.sh which copies the output to
# app/src-tauri/binaries/ in the Tauri-expected triple-suffixed form:
#   lumi-brain-x86_64-unknown-linux-gnu
#
# KNOWN MISSING-IMPORT ERRORS ENCOUNTERED DURING BUILD
# =====================================================
# None yet — update this block as errors surface during CI builds.

from pathlib import Path

REPO_ROOT = Path(SPECPATH).parent
SRC_ENTRY = str(REPO_ROOT / "src" / "main.py")
_model_dir = REPO_ROOT / "models"
_config_src = str(REPO_ROOT / "config.yaml")

import glob as _glob
import sys as _sys

_VENV_SITE = str(REPO_ROOT / ".venv" / "lib" / f"python{_sys.version_info.major}.{_sys.version_info.minor}" / "site-packages")

# llama-cpp-python ships its own libllama.so; ctypes loads it by name at runtime.
_LLAMA_LIBS = _glob.glob(f"{_VENV_SITE}/llama_cpp/lib/libllama*")

a = Analysis(
    [SRC_ENTRY],
    pathex=[str(REPO_ROOT)],
    binaries=[(lib, "llama_cpp/lib") for lib in _LLAMA_LIBS],
    datas=[
        (_config_src, "."),
        # openwakeword ships melspectrogram.onnx and sample wake-word models
        # as package resources; onnxruntime loads them by absolute path at
        # runtime so they must be collected into the bundle.
        (f"{_VENV_SITE}/openwakeword/resources", "openwakeword/resources"),
    ] + (
        [(str(_model_dir), "models")] if _model_dir.is_dir() else []
    ),
    hiddenimports=[
        # Wake word — exact pin required; see IMPORTANT CONSTRAINTS above
        "openwakeword",
        "openwakeword.utils",
        "openwakeword.model",
        # Audio I/O
        "sounddevice",
        "soundfile",
        # LLM inference
        "llama_cpp",
        # TTS
        "kokoro",
        "kokoro_onnx",
        # IPC / WebSocket
        "websockets",
        "websockets.legacy",
        "websockets.server",
        "websockets.exceptions",
        # Source packages
        "src.main",
        "src.core",
        "src.core.config",
        "src.core.events",
        "src.core.event_bridge",
        "src.core.ws_transport",
        "src.core.orchestrator",
        "src.core.startup_check",
        "src.core.logging_config",
        "src.audio",
        "src.audio.ears",
        "src.audio.scribe",
        "src.audio.mouth",
        "src.audio.hotkey",
        "src.llm",
        "src.llm.inference_dispatcher",
        "src.llm.reasoning_router",
        "src.llm.reflex_router",
        "src.llm.model_loader",
        "src.llm.prompt_engine",
        "src.llm.memory",
        "src.tools",
        "src.tools.os_actions",
        "src.tools.web_search",
        "src.tools.datetime_tool",
        "src.rag",
        # Numeric / signal processing
        "numpy",
        "scipy",
        "scipy.signal",
        # ONNX runtime (openwakeword + faster-whisper)
        "onnxruntime",
        "onnxruntime.capi",
        # Speech / VAD
        "webrtcvad",
        "faster_whisper",
        # Config / utils
        "yaml",
        "colorama",
        "pynput",
        "pynput.keyboard",
        # RAG (optional — included so the bundle works with rag enabled)
        "sqlite_vec",
        "sentence_transformers",
        "pypdf",
        # HTTP / HTML parsing (WebSearchTool)
        "requests",
        "bs4",
        "bs4.builder",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Training-only dependencies — not needed at runtime
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "jupyter",
        "speechbrain",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lumi-brain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX breaks CUDA .so files
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="lumi-brain",
)
