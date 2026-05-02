# Project Lumi (ルミ)

> **"The Ghost in the Machine."**
>
> Lumi is a local, privacy-first desktop companion designed to be invisible until needed. Built with a "Zero-Cost" philosophy: runs entirely on consumer hardware without interfering with games, renders, or any foreground task.

**For the full technical design, see [ARCHITECTURE.md](ARCHITECTURE.md).**
**For the development plan and known issues, see [TODO.md](TODO.md).**

---

## Current Status

**Phases 1–9.5 + Ring 1 complete. MVP-track: voice assistant pipeline + OS tools + personal knowledge base RAG + runtime settings UI + Tauri/React frontend.**

| Phase | Name | Status |
|---|---|---|
| 1 | The Ears (Audio Input) | COMPLETE |
| 2 | The Scribe (Transcription) | COMPLETE |
| 3 | The Brain (Intelligence) | COMPLETE |
| 4 | The Mouth (TTS) | COMPLETE |
| 5 | The Body (Visuals + IPC) | COMPLETE |
| 6 | The Hands (OS Control) | COMPLETE |
| 7 | RAG Personal Knowledge Base | COMPLETE |
| 8.5 | Settings UI (Runtime Config) | COMPLETE |
| 9 | Avatar Artwork | NOT STARTED |
| 9.5 | Tauri UI Overlay + Ring 1 | COMPLETE |

---

## What Works Right Now

- Say **"Hey Lumi"** → custom ONNX wake word model detects it (>0.8 confidence)
- VAD-based smart recording stops automatically when you stop speaking
- `faster-whisper` transcribes your speech to text on CPU (int8 quantized)
- Context injection reduces mis-transcriptions ("Firefox" vs "Fire folks")
- Event-driven pipeline: `Ears` posts `WakeDetectedEvent` to a central queue; `Orchestrator` dispatches all events
- Thread-safe state machine enforces `IDLE → LISTENING → PROCESSING → SPEAKING` transitions
- Local LLM responses via `llama-cpp-python` (Phi-3.5 Mini / Gemma 2 2B); reflex fast-path for greetings and time queries
- Kokoro-82M ONNX TTS synthesis with non-blocking `SpeakerThread` playback
- Typed configuration loaded from `config.yaml` with auto-detected hardware edition
- Startup validation halts on missing wake word model, wrong openwakeword version, or no microphone
- Structured logging (human-readable or JSON) via `src/core/logging_config.py`
- **Tauri/React UI** (`app/`) connects to the Python Brain via WebSocket on `ws://127.0.0.1:5556`; drives an animated avatar from Brain state events. IPC is enabled by default (`ipc.enabled: true` in `config.yaml`).
- **Runtime settings panel** — open with gear icon or Ctrl+, in the Tauri/React UI; reads live config schema from Brain, applies hot changes instantly, marks restart-required fields with `[↻]`
- **RAG personal knowledge base** — hybrid BM25 + vector kNN retrieval (SQLite FTS5 + sqlite-vec), RRF fusion, `all-MiniLM-L6-v2` embeddings; `scripts/ingest_docs.py` to index personal documents; off by default
- OS automation tools: app launch, clipboard, file info, window list, screenshot analysis (moondream2)
- LLM token streaming to Tauri/React UI; per-viseme-group mouth animations (Kokoro lip-sync)
- **Push-to-talk** global hotkey (default Ctrl+Space) as wake-word fallback (`src/audio/hotkey.py`, `PTTListener`; optional `pynput` dep); toggle via `audio.ptt_enabled` in settings
- **First-run setup screen** — `SetupPanel` appears when required models are absent; lists missing items from `SystemStatusEvent`
- ~900 tests passing (7 skipped); 80% coverage gate enforced in CI; behavioral regression contract suite in `tests/test_regression.py`

---

## Roadmap

### Phase 1: The Ears — Complete

- [x] Custom "Hey Lumi" wake word model (openWakeWord ONNX)
- [x] VAD smart stop (silero-vad)
- [x] Threaded Producer-Consumer audio pipeline
- [x] Double-trigger cooldown fix
- [x] Latency tuning ⚠️ Monitoring for buffer underruns

### Phase 2: The Scribe — Complete

- [x] faster-whisper int8 transcription (CPU)
- [x] Context injection for accuracy
- [x] Command parsing infrastructure (event-driven pipeline wired; `CommandResultEvent` defined)

### Phase 3: The Brain — Complete

- [x] Structured logging (`logging_config.py`, `setup_logging()`)
- [x] Startup validation (`startup_check.py`, hard fail on missing models / wrong versions)
- [x] Test infrastructure (`pytest` + `pytest-cov`, 80% coverage gate)
- [x] Configuration system + hardware edition auto-detection (`config.yaml` + `config.py`)
- [x] Event-driven orchestrator (`orchestrator.py`, replaces synchronous pipeline)
- [x] State machine (`state_machine.py`, `IDLE → LISTENING → PROCESSING → SPEAKING`)
- [x] `openwakeword==0.4.0` exact pin with startup version check
- [x] LLM integration (Phi-3.5 Mini / Gemma 2 2B via llama-cpp-python)
- [x] VRAM hibernate/wake lifecycle (`ModelLoader`)
- [x] Reflex router + Reasoning router
- [x] Memory (JSON conversation history, `ConversationMemory`)

**Fine-tuning roadmap (Phase 3+):** QLoRA-based fine-tuning with LoRA adapter hot-swap architecture (<100ms domain switch). Personality injection, OS tool-call training, and multi-turn context. See [ARCHITECTURE.md § 5](ARCHITECTURE.md#5-fine-tuning-strategy-phase-3--beyond) for full strategy.

### Phase 4: The Mouth — Complete

- [x] Kokoro-82M ONNX TTS (`src/audio/mouth.py`, `KokoroTTS`)
- [x] Non-blocking audio playback (`SpeakerThread`, `src/audio/speaker.py`)
- [x] TTS config keys (`TTSConfig`, `tts:` section in `config.yaml`, startup check)
- [ ] Viseme extraction for lip-sync (deferred to Phase 6)

### Phase 5: The Body — Complete

- [x] IPC transport (`src/core/ipc_transport.py`) and event bridge (`src/core/event_bridge.py`, `EventBridge`)
- [x] Enable with `ipc.enabled: true` in `config.yaml`
- [ ] LightRAG personal knowledge base (deferred to Phase 6). See [ARCHITECTURE.md § 6](ARCHITECTURE.md#6-lightrag-optional-personal-knowledge-base-phase-5) for details.

### Phase 6: The Hands — Complete

- [x] OS automation tools: `AppLaunchTool`, `ClipboardTool`, `FileInfoTool`, `WindowListTool`; cross-platform adapters (macOS, Windows)
- [x] Vision tool: `ScreenshotTool` with moondream2 GGUF description
- [x] LLM token streaming to frontend (`llm_token` wire frame)
- [x] Viseme extraction for lip-sync (`viseme_map.py`, `VisemeEvent` from Kokoro phonemes)
- [ ] Real avatar artwork (placeholder images still in use)

### Phase 7: RAG Personal Knowledge Base — Complete

- [x] `src/rag/` package — `DocumentStore` (SQLite FTS5 + sqlite-vec), `Chunker`, `Embedder`, `Loader`, RRF fusion, `RAGRetriever`
- [x] `scripts/ingest_docs.py` — CLI to chunk, embed, and index personal documents
- [x] `scripts/measure_rag_latency.py` — end-to-end latency benchmark (gate p95 < 2.0 s)
- [x] IPC wiring: `rag_retrieval`, `rag_status` outbound; `rag_set_enabled` inbound
- [x] RAG off by default (`config.rag.enabled: false`); enable + `uv sync --extra rag` to activate
- [ ] Real avatar artwork

### Phase 8.5: Settings UI (Runtime Config) — Complete

- [x] `src/core/config_runtime.py` — `ConfigManager`, `ConfigObserver`, `ConfigUpdateResult`; live apply via `dataclasses.replace()`; thread-safe RLock
- [x] `src/core/config_schema.py` — `FIELD_META` dict; UI metadata (control type, min/max, restart_required) for 47 user-facing fields
- [x] `src/core/config_writer.py` — atomic YAML write (tmp + fsync + rename), `.bak` rollover
- [x] IPC wire events: `config_schema_request` (Body→Brain), `config_schema` (Brain→Body), `config_update` (Body→Brain), `config_update_result` (Brain→Body)
- [x] `app/src/components/SettingsPanel.tsx` — gear icon / Ctrl+, entry; 7 tabs; 7 control types
- [x] `scripts/setup_wizard.py` — guided first-run configuration

### Phase 9.5: Tauri UI Overlay + Ring 1 — Complete

- [x] `src/core/ws_transport.py` (`WSTransport`) — asyncio WebSocket server; Brain now runs WS directly; no bridge process
- [x] `app/` — Tauri 2 + React 18 frontend: transparent overlay, chat panel, settings; `useLumiState` hook; `IBrainClient`
- [x] `app/src/components/SetupPanel.tsx` — first-run guidance driven by `system_status.setup_required`
- [x] `src/audio/hotkey.py` (`PTTListener`) — global push-to-talk hotkey (Ctrl+Space default)
- [x] Cross-platform OS tools: macOS bundle dispatch, Windows adapters (pyperclip/pygetwindow)
- [x] `SystemStatusEvent` — broadcasts capability flags + setup state to frontend on startup

---

## Configuring Lumi

**`config.yaml`** lives in the repo root; all keys are optional (defaults in `src/core/config.py`). Hardware is auto-detected: set `edition: auto` (or omit) to read VRAM via `nvidia-smi`; choose `light` / `standard` / `pro` to override.

**Configuration sections:**
- `audio` — microphone sample rate, wake-word sensitivity, VAD threshold, recording timeouts. See `config.yaml` for full docs.
- `scribe` — faster-whisper model size and quantization. Smaller models run faster on CPU.
- `llm` — model path, GPU layer offload (`n_gpu_layers`). Phi-3.5 Mini or Gemma 2 by default.
- `tts` — TTS voice, speech rate. Kokoro ONNX recommended.
- `ipc` — `enabled: true` by default; activates the WebSocket server on port 5556.
- `tools` — enable/disable OS automation (app launch, clipboard, window list, screenshot).
- `vision` — screenshot tool settings and moondream2 vision model.
- `persona` — custom system prompt for personality injection.
- `rag` — enable/disable personal knowledge base; embedding model path.

**Settings UI (new in Phase 8.5):** Open with the **⚙ gear icon** or **Ctrl+,** in the Tauri/React UI. The panel reads the live schema from the Brain and writes changes back via the IPC channel. Changes marked `[↻]` require a restart:
- **Live (hot) changes** — audio thresholds, `log_level`, LLM `temperature` / `max_tokens`, TTS `voice`, tools enabled, RAG enabled, persona prompt
- **Restart required `[↻]`** — model paths, `sample_rate`, IPC settings, `n_gpu_layers`, KV cache quantization

**First-run setup:** Run `uv run python scripts/setup_wizard.py` for guided configuration.

---

## Architecture Overview

Lumi uses a "Split-Brain" design: a Python backend handles all intelligence and audio processing; a Tauri/React frontend renders the avatar overlay and settings UI. They communicate over WebSocket with 4-byte length-prefix framing. The Brain runs the WebSocket server directly — no bridge process required.

```
Python Backend  ◄──── WebSocket (length-prefix) ────►  Tauri/React Frontend
(Ears, Brain,          JSON wire frames                 (Avatar, Animations,
 Scribe, Mouth)        ws://127.0.0.1:5556               Settings, Overlay)
```

Start the Python Brain first (`uv run python -m src.main`), then start the Tauri app (`cd app && npm run dev`). IPC is enabled by default.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package Manager | `uv` |
| Wake Word | openWakeWord (custom ONNX model) |
| VAD | Silero VAD v5 |
| STT | faster-whisper (int8, CPU) |
| LLM | llama-cpp-python, Phi-3.5 Mini or Gemma 2 2B |
| Fine-tuning (planned) | QLoRA + llama.cpp GGUF export |
| TTS | Kokoro-82M ONNX |
| Knowledge retrieval | Hybrid BM25+vec RAG (Phase 7); SQLite FTS5 + sqlite-vec; all-MiniLM-L6-v2 |
| Frontend | Tauri 2 + React 18 (`app/`) |
| IPC | WebSocket, 4-byte length-prefix framing (`WSTransport` + `EventBridge`; `websockets` library) |
| IPC handshake | Version negotiation via `src/core/handshake.py` (`hello` / `hello_ack`) |
| Testing | pytest + pytest-cov (80% coverage gate) |
| Logging | Python `logging` module (`src/core/logging_config.py`) |

---

## Pre-Alpha Notice

This is a work-in-progress. Phases 1–9.5 and Ring 1 are complete: the full audio-to-speech pipeline, Tauri/React overlay, OS tools, personal knowledge base RAG, runtime settings UI, and first-run setup screen are all functional. The next milestone is Ring 2: Brain sidecar bundling, persona LoRA training, streaming TTS, and an E2E integration smoke test. See [TODO.md](TODO.md) and [MVP_REPORT.md](MVP_REPORT.md) for the active work backlog.
