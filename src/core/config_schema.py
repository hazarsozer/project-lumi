"""
Config schema metadata for the Settings UI panel (Wave S0).

``FIELD_META`` maps dotted config path keys (matching the ``changes`` dict
consumed by ``ConfigManager.apply()``) to UI-rendering hints and restart
semantics.

This module has no runtime imports from src.core.config — it is a pure
data declaration so the UI layer can import it without loading the full
config stack.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# FIELD_META: one entry per user-facing config field
# ---------------------------------------------------------------------------
# Key format:  "<section>.<field>"  or  "<field>" for top-level scalars.
#
# Entry structure:
#   label            : str   — Human-readable label for the UI widget
#   help             : str   — Tooltip / description text
#   control          : str   — Widget type: slider | toggle | select | text |
#                              path | number | multiselect
#   restart_required : bool  — True → restart needed to apply change
#   min              : float | int   (slider / number only)
#   max              : float | int   (slider / number only)
#   step             : float | int   (slider only)
#   options          : list[str]     (select / multiselect only)
# ---------------------------------------------------------------------------

FIELD_META: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Top-level scalars
    # -------------------------------------------------------------------------
    "edition": {
        "label": "Performance Edition",
        "help": (
            "Hardware tier used to decide LLM GPU offload strategy. "
            "'light' is CPU-only. 'standard' uses partial GPU offload. "
            "'pro' offloads all layers to GPU."
        ),
        "control": "select",
        "options": ["light", "standard", "pro"],
        "restart_required": True,
    },
    "log_level": {
        "label": "Log Level",
        "help": (
            "Root logger verbosity. 'DEBUG' is the most verbose. "
            "'CRITICAL' is the least verbose."
        ),
        "control": "select",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "restart_required": False,
    },
    "json_logs": {
        "label": "JSON Log Format",
        "help": (
            "When enabled, logs are emitted as one JSON object per line "
            "instead of human-readable text. Useful for log aggregators."
        ),
        "control": "toggle",
        "restart_required": False,
    },
    # -------------------------------------------------------------------------
    # audio section
    # -------------------------------------------------------------------------
    "audio.sample_rate": {
        "label": "Sample Rate (Hz)",
        "help": (
            "Microphone sample rate in Hz. Both openwakeword and "
            "faster-whisper require 16000 Hz."
        ),
        "control": "number",
        "restart_required": True,
    },
    "audio.chunk_size": {
        "label": "Chunk Size (frames)",
        "help": (
            "InputStream blocksize in frames. 1280 frames @ 16 kHz = 80 ms "
            "per chunk, which is the recommended size for openwakeword."
        ),
        "control": "number",
        "restart_required": True,
    },
    "audio.sensitivity": {
        "label": "Wake-Word Sensitivity",
        "help": (
            "Wake-word detection threshold [0.0–1.0]. Lower values increase "
            "recall (more detections, more false positives). Higher values "
            "increase precision."
        ),
        "control": "slider",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "restart_required": False,
    },
    "audio.vad_threshold": {
        "label": "VAD Threshold",
        "help": (
            "Voice Activity Detection threshold [0.0–1.0]. Scores above this "
            "value during command recording count as speech."
        ),
        "control": "slider",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "restart_required": False,
    },
    "audio.silence_timeout_s": {
        "label": "Silence Timeout (s)",
        "help": ("Seconds of continuous silence after speech before recording stops."),
        "control": "slider",
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "restart_required": False,
    },
    "audio.recording_timeout_s": {
        "label": "Recording Timeout (s)",
        "help": "Hard upper bound on command recording duration in seconds.",
        "control": "slider",
        "min": 1.0,
        "max": 60.0,
        "step": 1.0,
        "restart_required": False,
    },
    "audio.wake_word_model_path": {
        "label": "Wake-Word Model Path",
        "help": "Path to the custom 'hey Lumi' ONNX wake-word model file.",
        "control": "path",
        "restart_required": True,
    },
    # -------------------------------------------------------------------------
    # scribe section
    # -------------------------------------------------------------------------
    "scribe.model_size": {
        "label": "Whisper Model Size",
        "help": (
            "Whisper model variant. Smaller models are faster but less "
            "accurate. '.en' variants are English-only and slightly faster."
        ),
        "control": "select",
        "options": [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "large-v3",
        ],
        "restart_required": True,
    },
    "scribe.beam_size": {
        "label": "Beam Size",
        "help": (
            "Beam search width. Higher values improve accuracy at the cost "
            "of inference speed."
        ),
        "control": "slider",
        "min": 1,
        "max": 10,
        "step": 1,
        "restart_required": True,
    },
    "scribe.compute_type": {
        "label": "Compute Type",
        "help": (
            "'int8' is fastest on CPU. 'float16' requires a GPU. "
            "'float32' has the highest quality but is slowest on CPU."
        ),
        "control": "select",
        "options": ["int8", "float16", "float32"],
        "restart_required": True,
    },
    "scribe.model_path": {
        "label": "STT Model Directory",
        "help": (
            "Local directory containing a pre-downloaded faster-whisper model. "
            "When this directory exists, the engine loads from disk instead of "
            "downloading from Hugging Face."
        ),
        "control": "path",
        "restart_required": True,
    },
    "scribe.initial_prompt": {
        "label": "Initial Prompt",
        "help": (
            "Optional context string injected at the start of each "
            "transcription. Helps the model recognise proper nouns. "
            "Leave empty to disable."
        ),
        "control": "text",
        "restart_required": False,
    },
    # -------------------------------------------------------------------------
    # llm section
    # -------------------------------------------------------------------------
    "llm.model_path": {
        "label": "LLM Model Path",
        "help": "Path to the GGUF model file on disk.",
        "control": "path",
        "restart_required": True,
    },
    "llm.n_gpu_layers": {
        "label": "GPU Layers",
        "help": (
            "Number of transformer layers to offload to GPU. "
            "0 = full CPU inference. -1 = offload all layers."
        ),
        "control": "number",
        "min": -1,
        "restart_required": True,
    },
    "llm.context_length": {
        "label": "Context Length (tokens)",
        "help": "KV-cache context window length in tokens.",
        "control": "number",
        "min": 512,
        "max": 131072,
        "restart_required": True,
    },
    "llm.max_tokens": {
        "label": "Max Output Tokens",
        "help": "Maximum number of tokens to generate per assistant response.",
        "control": "slider",
        "min": 64,
        "max": 4096,
        "step": 64,
        "restart_required": False,
    },
    "llm.temperature": {
        "label": "Temperature",
        "help": (
            "Sampling temperature [0.0–2.0]. Lower = more deterministic. "
            "Higher = more creative."
        ),
        "control": "slider",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "restart_required": False,
    },
    "llm.vram_budget_gb": {
        "label": "VRAM Budget (GB)",
        "help": (
            "VRAM budget in gigabytes used at runtime to decide the GPU "
            "offload strategy. The LLM engine will not exceed this limit."
        ),
        "control": "slider",
        "min": 0.0,
        "max": 48.0,
        "step": 0.5,
        "restart_required": True,
    },
    "llm.memory_dir": {
        "label": "Memory Directory",
        "help": (
            "Directory for persistent conversation memory. "
            "Tilde (~) is expanded at use time."
        ),
        "control": "path",
        "restart_required": False,
    },
    # -------------------------------------------------------------------------
    # tts section
    # -------------------------------------------------------------------------
    "tts.enabled": {
        "label": "Enable TTS",
        "help": (
            "When disabled, Lumi runs in silent mode without loading any "
            "TTS model files. Useful for headless or CI environments."
        ),
        "control": "toggle",
        "restart_required": False,
    },
    "tts.voice": {
        "label": "TTS Voice",
        "help": (
            "Kokoro voice identifier. 'af_heart' is the default English voice. "
            "Other voices depend on the bundled voices.bin file."
        ),
        "control": "text",
        "restart_required": False,
    },
    "tts.model_path": {
        "label": "TTS Model Path",
        "help": "Path to the Kokoro ONNX model file.",
        "control": "path",
        "restart_required": True,
    },
    "tts.voices_path": {
        "label": "TTS Voices Path",
        "help": "Path to the Kokoro voices binary file.",
        "control": "path",
        "restart_required": True,
    },
    # -------------------------------------------------------------------------
    # ipc section
    # -------------------------------------------------------------------------
    "ipc.enabled": {
        "label": "Enable IPC Server",
        "help": (
            "When enabled, starts a ZeroMQ TCP server to communicate with "
            "the Godot frontend. Keep disabled for audio-only or CI runs."
        ),
        "control": "toggle",
        "restart_required": True,
    },
    "ipc.address": {
        "label": "IPC Address",
        "help": (
            "ZMQ transport and host prefix. The port is appended as ':PORT'. "
            "For local-only use keep 'tcp://127.0.0.1'."
        ),
        "control": "text",
        "restart_required": True,
    },
    "ipc.port": {
        "label": "IPC Port",
        "help": "Port number for the ZMQ socket [1024–65535].",
        "control": "number",
        "min": 1024,
        "max": 65535,
        "restart_required": True,
    },
    # -------------------------------------------------------------------------
    # tools section
    # -------------------------------------------------------------------------
    "tools.enabled": {
        "label": "Enable OS Tools",
        "help": "When enabled, Lumi can execute OS action tools on your behalf.",
        "control": "toggle",
        "restart_required": False,
    },
    "tools.allowed_tools": {
        "label": "Allowed Tools",
        "help": (
            "Allowlist of OS tool names the executor may run. "
            "Tools not in this list are rejected before execution."
        ),
        "control": "multiselect",
        "options": [
            "launch_app",
            "clipboard",
            "file_info",
            "window_list",
            "rag_ingest",
        ],
        "restart_required": False,
    },
    "tools.execution_timeout_s": {
        "label": "Tool Execution Timeout (s)",
        "help": (
            "Per-tool execution timeout in seconds. Tool threads that exceed "
            "this budget are abandoned and a failure result is returned."
        ),
        "control": "slider",
        "min": 1.0,
        "max": 60.0,
        "step": 1.0,
        "restart_required": False,
    },
    # -------------------------------------------------------------------------
    # vision section
    # -------------------------------------------------------------------------
    "vision.enabled": {
        "label": "Enable Vision",
        "help": (
            "Enable screenshot capture and moondream2 vision model. "
            "Set to true only after downloading the moondream2.gguf model."
        ),
        "control": "toggle",
        "restart_required": True,
    },
    "vision.model_path": {
        "label": "Vision Model Path",
        "help": "Path to the moondream2 GGUF model file.",
        "control": "path",
        "restart_required": True,
    },
    "vision.capture_method": {
        "label": "Screen Capture Method",
        "help": (
            "'auto' selects the best method for your desktop environment. "
            "'grim' for Wayland. 'scrot' for X11. 'pillow' as a fallback."
        ),
        "control": "select",
        "options": ["auto", "grim", "scrot", "pillow"],
        "restart_required": True,
    },
    "vision.max_resolution": {
        "label": "Max Capture Resolution",
        "help": (
            "Downscale captured screenshots to this maximum resolution "
            "before passing to the vision model."
        ),
        "control": "slider",
        "min": 320,
        "max": 4096,
        "step": 64,
        "restart_required": True,
    },
    # -------------------------------------------------------------------------
    # persona section
    # -------------------------------------------------------------------------
    "persona.system_prompt": {
        "label": "System Prompt",
        "help": (
            "Full system prompt sent to the LLM as the <|system|> block. "
            "Leave empty to use the built-in default Lumi persona."
        ),
        "control": "text",
        "restart_required": False,
    },
    # -------------------------------------------------------------------------
    # rag section
    # -------------------------------------------------------------------------
    "rag.enabled": {
        "label": "Enable RAG",
        "help": (
            "Enable the personal knowledge-base retriever. When enabled, "
            "Lumi searches your documents before responding."
        ),
        "control": "toggle",
        "restart_required": False,
    },
    "rag.db_path": {
        "label": "RAG Database Path",
        "help": (
            "Path to the SQLite database file used for document, chunk, and "
            "vector storage. Tilde (~) is expanded at use time."
        ),
        "control": "path",
        "restart_required": True,
    },
    "rag.embedding_model": {
        "label": "Embedding Model",
        "help": (
            "HuggingFace model ID for the sentence-embedding model. "
            "'sentence-transformers/all-MiniLM-L6-v2' is 80 MB, CPU-only."
        ),
        "control": "text",
        "restart_required": True,
    },
    "rag.chunk_size": {
        "label": "Chunk Size (tokens)",
        "help": "Fixed chunk size in tokens used by the chunker at ingest time.",
        "control": "number",
        "min": 64,
        "max": 2048,
        "restart_required": True,
    },
    "rag.chunk_overlap": {
        "label": "Chunk Overlap (tokens)",
        "help": (
            "Overlap in tokens between consecutive chunks to preserve context "
            "across chunk boundaries."
        ),
        "control": "number",
        "min": 0,
        "max": 512,
        "restart_required": True,
    },
    "rag.retrieval_top_k": {
        "label": "Retrieval Top-K",
        "help": (
            "Number of top candidates retrieved per retrieval mode (BM25 + "
            "vector) before Reciprocal Rank Fusion re-ranks the results."
        ),
        "control": "slider",
        "min": 1,
        "max": 20,
        "step": 1,
        "restart_required": False,
    },
    "rag.context_char_budget": {
        "label": "Context Character Budget",
        "help": (
            "Hard budget in characters for the retrieved context block "
            "injected into the prompt. ~2400 chars ≈ 600 tokens."
        ),
        "control": "number",
        "min": 500,
        "restart_required": False,
    },
    "rag.min_score": {
        "label": "Minimum Retrieval Score",
        "help": (
            "Minimum fused RRF score threshold [0.0–1.0]. Hits below this "
            "floor are discarded to prevent low-quality chunks from being "
            "injected when no relevant document exists."
        ),
        "control": "slider",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "restart_required": False,
    },
    "rag.corpus_dir": {
        "label": "Corpus Directory",
        "help": (
            "Directory the ingest CLI will scan for documents "
            "(.md, .txt, .pdf). Tilde (~) is expanded at use time."
        ),
        "control": "path",
        "restart_required": False,
    },
    "rag.retrieval_timeout_s": {
        "label": "Retrieval Timeout (s)",
        "help": (
            "Hard ceiling in seconds for a single retrieval call. "
            "If exceeded, an empty result is returned and the LLM still "
            "responds without retrieved context."
        ),
        "control": "slider",
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
        "restart_required": False,
    },
}
