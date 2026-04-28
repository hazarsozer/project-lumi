"""
Centralized, typed configuration for Project Lumi.

Usage:
    from src.core.config import load_config, LumiConfig

    config = load_config()          # uses config.yaml in cwd
    config = load_config("my.yaml") # custom path

All config objects are frozen dataclasses — they cannot be mutated after
construction, which prevents accidental side-effects across modules.

Circular-import constraint: this module MUST NOT import from src/audio/,
src/llm/, or src/interface/.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for the audio capture and wake-word pipeline.

    All values correspond to constants previously hardcoded in ears.py.
    """

    # Microphone sample rate in Hz — openwakeword and faster-whisper both
    # require 16 kHz input.
    sample_rate: int = 16000

    # InputStream blocksize in frames — 1280 frames @ 16 kHz = 80 ms per chunk,
    # which is the recommended chunk size for openwakeword inference.
    chunk_size: int = 1280

    # Wake-word detection threshold; scores above this value trigger on_wake.
    sensitivity: float = 0.8

    # Voice Activity Detection threshold; scores above this value count as
    # speech during command recording.
    vad_threshold: float = 0.5

    # Seconds of continuous silence after speech before recording stops.
    silence_timeout_s: float = 1.5

    # Hard upper bound on command recording duration in seconds.
    recording_timeout_s: float = 10.0

    # Path to the custom "hey Lumi" ONNX wake-word model.
    wake_word_model_path: str = "models/hey_lumi.onnx"


@dataclass(frozen=True)
class ScribeConfig:
    """Configuration for the faster-whisper STT engine (scribe.py).

    Defaults match the values hardcoded in scribe.py at Phase 3 entry.
    """

    # Whisper model variant — "tiny.en" is CPU-friendly for Phase 3.
    model_size: str = "tiny.en"

    # Beam search width; higher values improve accuracy at the cost of speed.
    beam_size: int = 5

    # Quantization type passed to WhisperModel — "int8" for CPU inference.
    compute_type: str = "int8"

    # Local directory containing a pre-downloaded faster-whisper model.
    # When this directory exists, the engine loads from disk instead of
    # downloading from Hugging Face.
    model_path: str = "models/faster-whisper-tiny.en"

    # Optional context string injected at the start of each transcription.
    # None disables the initial prompt entirely.
    initial_prompt: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the local LLM (Phase 4 — not active in Phase 3).

    Included here so ipc-engineer and llm-engineer can depend on stable
    field names before their modules are written.
    """

    # Path to the GGUF model file on disk.
    model_path: str = "models/llm/phi-3.5-mini.gguf"

    # Number of transformer layers to offload to GPU.
    # 0 = full CPU inference (required for "light" edition).
    n_gpu_layers: int = 0

    # KV-cache context window in tokens.
    context_length: int = 4096

    # Maximum number of tokens to generate per response.
    max_tokens: int = 512

    # Sampling temperature — higher values produce more varied output.
    temperature: float = 0.7

    # VRAM budget in gigabytes used to decide the offload strategy at runtime.
    vram_budget_gb: float = 4.0

    # Optional KV-cache quantization type forwarded to llama_cpp.Llama as
    # ``cache_type_k`` / ``cache_type_v``.  ``None`` leaves the cache as FP16
    # (llama.cpp default).  Set to e.g. ``"turbo3"`` once upstream
    # llama-cpp-python ships TurboQuant (llama.cpp PR #21089).  If the
    # installed build does not yet support the kwarg, ModelLoader silently
    # falls back to the FP16 cache and logs a warning.
    kv_cache_quant: str | None = None

    # Directory for persistent conversation memory (expanded at use time).
    memory_dir: str = "~/.lumi/memory"

    # Maximum wall-clock seconds allowed for a single LLM inference call.
    # A watchdog timer set to this value will fire if the inference thread
    # does not complete within the budget, setting _llm_cancel_flag and
    # posting an error event so the orchestrator returns to IDLE.
    # 0.0 disables the watchdog entirely.
    inference_timeout_s: float = 30.0

    # Optional path to a .gguf LoRA adapter file.  When set, the adapter is
    # loaded at model init time via the llama-cpp-python constructor.  Requires
    # lora_scale to be set; use_mmap is forced off by llama-cpp-python when any
    # lora_path is provided.
    lora_path: str | None = None

    # LoRA adapter strength (0.0–1.0+). Only used when lora_path is set.
    lora_scale: float = 1.0


@dataclass(frozen=True)
class TTSConfig:
    """Configuration for the Kokoro ONNX text-to-speech engine (mouth.py).

    Defaults match the values expected by KokoroTTS at Phase 4 entry.
    """

    # Whether TTS is active.  Set to false to run in silent mode without
    # loading any model files (useful for headless or CI environments).
    enabled: bool = True

    # Kokoro voice identifier passed to kokoro_onnx.Kokoro.create().
    # "af_heart" is the default English voice bundled with the model.
    voice: str = "af_heart"

    # Path to the Kokoro ONNX model file.
    model_path: str = "models/tts/kokoro-v1_0.onnx"

    # Path to the Kokoro voices binary file.
    voices_path: str = "models/tts/voices.bin"


@dataclass(frozen=True)
class IPCConfig:
    """ZeroMQ IPC endpoint configuration (consumed by ipc-engineer)."""

    # Whether the IPC server is active.  Set to false to disable the
    # TCP server entirely (useful for headless or CI environments).
    enabled: bool = False

    # ZMQ transport + host prefix — port is appended as ":PORT".
    address: str = "tcp://127.0.0.1"

    # Port number for the ZMQ socket.
    port: int = 5555


@dataclass(frozen=True)
class ToolsConfig:
    """Configuration for the OS action tool framework (Phase 6)."""

    # Whether OS tools are enabled at all.
    enabled: bool = True

    # Allowlist of tool names the executor may run.  Any tool name not in
    # this tuple is rejected before execution — provides a secondary defence
    # against prompt-injection attacks asking Lumi to invoke arbitrary tools.
    allowed_tools: tuple[str, ...] = (
        "launch_app",
        "clipboard",
        "file_info",
        "window_list",
        "rag_ingest",
    )

    # Per-tool execution timeout in seconds.  Tool threads that exceed this
    # budget are abandoned and a failure ToolResult is returned.
    execution_timeout_s: float = 10.0


@dataclass(frozen=True)
class VisionConfig:
    """Configuration for the screenshot + moondream2 vision model (Phase 6)."""

    # Set to true only after downloading the moondream2.gguf model.
    enabled: bool = False

    # Path to the moondream2 GGUF model file.
    model_path: str = "models/vision/moondream2.gguf"

    # Screenshot capture method: "auto" | "grim" (Wayland) | "scrot" (X11) | "pillow"
    capture_method: str = "auto"

    # Downscale captured screenshots to this max resolution before inference.
    max_resolution: int = 1280


@dataclass(frozen=True)
class PersonaConfig:
    """Configuration for Lumi's persona and system prompt (Wave F1).

    The ``system_prompt`` field holds the full text sent as the ``<|system|>``
    block.  It defaults to ``None`` so that ``PromptEngine`` can fall back to
    its built-in ``DEFAULT_SYSTEM_PROMPT`` constant when the config file does
    not override it.
    """

    # Full system prompt text.  None means "use the built-in default".
    system_prompt: str | None = None


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the personal knowledge-base retriever (Phase 7)."""

    # Whether the RAG feature is active.  Off by default — must be explicitly
    # enabled by the user (via config.yaml or the Godot UI toggle).
    enabled: bool = False

    # Path to the SQLite database file used for document, chunk, and vector
    # storage.  Tilde is expanded at use time.
    db_path: str = "~/.lumi/rag.db"

    # HuggingFace model ID for the sentence-embedding model.
    # all-MiniLM-L6-v2 is 80 MB, ~20 ms CPU inference, 384-dimensional vectors.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Fixed chunk size in tokens used by the chunker at ingest time.
    chunk_size: int = 512

    # Overlap in tokens between consecutive chunks to preserve context across
    # chunk boundaries.
    chunk_overlap: int = 64

    # Number of top candidates retrieved per retrieval mode (BM25 + vector)
    # before Reciprocal Rank Fusion re-ranks and trims the final result list.
    retrieval_top_k: int = 8

    # Hard budget (in characters, not tokens) for the retrieved context block
    # injected into the prompt.  2400 chars ≈ 600 tokens, fitting comfortably
    # in the 4096-token context window alongside history and generation budget.
    context_char_budget: int = 2400

    # Minimum fused RRF score threshold [0.0–1.0].  Hits below this floor are
    # discarded entirely — prevents low-quality chunks from being injected when
    # no genuinely relevant document exists.
    min_score: float = 0.15

    # Directory the ingest CLI will scan for documents (.md, .txt, .pdf).
    # Tilde is expanded at use time.
    corpus_dir: str = "~/.lumi/docs"

    # Hard ceiling (in seconds) for a single retrieval call.  If the call
    # exceeds this budget the retriever returns an empty result and logs a
    # warning, so the LLM still responds — without retrieved context.
    retrieval_timeout_s: float = 0.4


@dataclass(frozen=True)
class LumiConfig:
    """Top-level configuration object passed to every subsystem at startup."""

    # Performance edition, auto-detected by detect_edition() or set in YAML.
    # Values: "light" | "standard" | "pro"
    edition: str = "standard"

    audio: AudioConfig = field(default_factory=AudioConfig)
    scribe: ScribeConfig = field(default_factory=ScribeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)

    # Root-logger level forwarded to setup_logging().
    log_level: str = "INFO"

    # When True, emit structured JSON logs instead of human-readable format.
    json_logs: bool = False


# ---------------------------------------------------------------------------
# Edition detection
# ---------------------------------------------------------------------------

# VRAM thresholds (in MiB) used to select the performance edition.
_LIGHT_THRESHOLD_MIB: int = 2048  # < 2 GiB  → light
_STANDARD_THRESHOLD_MIB: int = 4096  # < 4 GiB  → standard
# ≥ 4 GiB (and the Pro band starts at 8 GiB) → pro


def detect_edition() -> str:
    """Auto-detect the performance edition based on available VRAM.

    Shells out to ``nvidia-smi`` to query total VRAM.  If the tool is
    unavailable (no NVIDIA GPU, nvidia-smi not on PATH, subprocess error),
    the "light" edition is returned — ensuring CPU-only operation is always
    the safe fallback.

    Returns:
        One of "light", "standard", or "pro".
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.debug("nvidia-smi not available; defaulting to 'light' edition.")
        return "light"

    if result.returncode != 0:
        logger.debug(
            "nvidia-smi exited with code %d; defaulting to 'light' edition.",
            result.returncode,
        )
        return "light"

    # nvidia-smi may report multiple GPUs; take the maximum.
    vram_mib: int = 0
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            vram_mib = max(vram_mib, int(line))

    if vram_mib < _LIGHT_THRESHOLD_MIB:
        edition = "light"
    elif vram_mib < _STANDARD_THRESHOLD_MIB:
        edition = "standard"
    else:
        edition = "pro"

    logger.debug("Detected %d MiB VRAM → edition '%s'.", vram_mib, edition)
    return edition


# ---------------------------------------------------------------------------
# YAML loader and config factory
# ---------------------------------------------------------------------------


def _merge_section(defaults: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a dict suitable for constructing a frozen dataclass.

    Starts from ``defaults.__dataclass_fields__`` keys/values and overlays
    any keys present in ``overrides``.  Unknown keys in ``overrides`` are
    silently ignored so that YAML files with extra comments or future keys
    do not crash older code.

    Args:
        defaults: A frozen dataclass instance whose fields supply defaults.
        overrides: A flat dict of string→value pairs from the YAML section.

    Returns:
        A dict containing only valid field names with merged values.
    """
    merged: dict[str, Any] = {
        f.name: getattr(defaults, f.name) for f in fields(defaults)
    }
    for key, value in overrides.items():
        if key in merged:
            merged[key] = value
        else:
            logger.warning("config.yaml: unknown key '%s' in section — ignored.", key)
    return merged


def load_config(path: str = "config.yaml") -> LumiConfig:
    """Load and merge configuration from a YAML file into typed defaults.

    If the file does not exist or is empty, all defaults are used.  This
    means the application starts successfully even without a ``config.yaml``,
    which is important for CI environments.

    Args:
        path: Path to the YAML configuration file.  Relative paths are
              resolved against the current working directory.

    Returns:
        A fully populated, frozen ``LumiConfig`` instance.
    """
    config_path = Path(path)
    raw: dict[str, Any] = {}

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded
            else:
                logger.warning(
                    "config.yaml did not parse to a dict (got %s); "
                    "using all defaults.",
                    type(loaded).__name__,
                )
        except yaml.YAMLError as exc:
            logger.error("Failed to parse config.yaml: %s — using all defaults.", exc)
    else:
        logger.debug("No config file found at '%s'; using built-in defaults.", path)

    # Build nested section configs from sub-dicts in the YAML.
    audio_cfg = AudioConfig(**_merge_section(AudioConfig(), raw.get("audio", {})))
    scribe_cfg = ScribeConfig(**_merge_section(ScribeConfig(), raw.get("scribe", {})))
    llm_cfg = LLMConfig(**_merge_section(LLMConfig(), raw.get("llm", {})))
    tts_cfg = TTSConfig(**_merge_section(TTSConfig(), raw.get("tts", {})))
    ipc_cfg = IPCConfig(**_merge_section(IPCConfig(), raw.get("ipc", {})))

    # ToolsConfig: YAML lists parse as Python list; convert allowed_tools to tuple.
    tools_raw = _merge_section(ToolsConfig(), raw.get("tools", {}))
    tools_raw["allowed_tools"] = tuple(tools_raw["allowed_tools"])
    tools_cfg = ToolsConfig(**tools_raw)

    vision_cfg = VisionConfig(**_merge_section(VisionConfig(), raw.get("vision", {})))

    rag_cfg = RAGConfig(**_merge_section(RAGConfig(), raw.get("rag", {})))

    persona_cfg = PersonaConfig(
        **_merge_section(PersonaConfig(), raw.get("persona", {}))
    )

    # Top-level scalar overrides.
    top_defaults = LumiConfig()
    edition = raw["edition"] if "edition" in raw else detect_edition()
    log_level = raw.get("log_level", top_defaults.log_level)
    json_logs = raw.get("json_logs", top_defaults.json_logs)

    return LumiConfig(
        edition=edition,
        audio=audio_cfg,
        scribe=scribe_cfg,
        llm=llm_cfg,
        tts=tts_cfg,
        ipc=ipc_cfg,
        tools=tools_cfg,
        vision=vision_cfg,
        rag=rag_cfg,
        persona=persona_cfg,
        log_level=log_level,
        json_logs=json_logs,
    )
