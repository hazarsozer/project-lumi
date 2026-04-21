# Project Lumi (ルミ)

> **"The Ghost in the Machine."**
>
> Lumi is a local, privacy-first desktop companion designed to be invisible until needed. Built with a "Zero-Cost" philosophy: runs entirely on consumer hardware without interfering with games, renders, or any foreground task.

**For the full technical design, see [ARCHITECTURE.md](ARCHITECTURE.md).**
**For the development plan and known issues, see [TODO.md](TODO.md).**

---

## Current Status

**Phases 1–7 complete. MVP-ready: voice assistant pipeline + OS tools + personal knowledge base RAG.**

| Phase | Name | Status |
|---|---|---|
| 1 | The Ears (Audio Input) | COMPLETE |
| 2 | The Scribe (Transcription) | COMPLETE |
| 3 | The Brain (Intelligence) | COMPLETE |
| 4 | The Mouth (TTS) | COMPLETE |
| 5 | The Body (Visuals + IPC) | COMPLETE |
| 6 | The Hands (OS Control) | COMPLETE |
| 7 | RAG Personal Knowledge Base | COMPLETE |

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
- **Godot 4 transparent overlay** (`ui/`) connects to the Python Brain via raw TCP on port 5555; drives an animated avatar from Brain state events. Set `ipc.enabled: true` in `config.yaml` to activate the IPC server.
- **RAG personal knowledge base** — hybrid BM25 + vector kNN retrieval (SQLite FTS5 + sqlite-vec), RRF fusion, `all-MiniLM-L6-v2` embeddings; `scripts/ingest_docs.py` to index personal documents; off by default
- OS automation tools: app launch, clipboard, file info, window list, screenshot analysis (moondream2)
- LLM token streaming to Godot overlay; per-viseme-group mouth animations (Kokoro lip-sync)
- ~749 tests at 91% coverage (80% gate enforced in CI); behavioral regression contract suite in `tests/test_regression.py`

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

- [x] Raw TCP IPC server (`src/core/ipc_transport.py`, 4-byte length-prefix framing)
- [x] Event bridge (`src/core/zmq_server.py`, `ZMQServer` — translates internal events ↔ JSON wire protocol)
- [x] Godot 4 transparent overlay (`ui/`) — 200×200 borderless window, X11/Wayland
- [x] `StreamPeerTCP` IPC client (`ui/scripts/lumi_client.gd`) with auto-reconnect
- [x] Avatar controller (`ui/scripts/avatar_controller.gd`) drives `AnimatedSprite2D` from Brain state events
- [x] Enable with `ipc.enabled: true` in `config.yaml` (default `false`)
- [ ] LightRAG personal knowledge base (deferred to Phase 6). See [ARCHITECTURE.md § 6](ARCHITECTURE.md#6-lightrag-optional-personal-knowledge-base-phase-5) for details.

### Phase 6: The Hands — Complete

- [x] OS automation tools: `AppLaunchTool`, `ClipboardTool`, `FileInfoTool`, `WindowListTool`
- [x] Vision tool: `ScreenshotTool` with moondream2 GGUF description
- [x] LLM token streaming to Godot overlay (`llm_token` wire frame)
- [x] Viseme extraction for lip-sync (`viseme_map.py`, `VisemeEvent` from Kokoro phonemes)
- [x] Godot: streaming text bubble, 8 per-viseme-group mouth animations
- [ ] Real avatar artwork (placeholder colored-circle sprites still in use)

### Phase 7: RAG Personal Knowledge Base — Complete

- [x] `src/rag/` package — `DocumentStore` (SQLite FTS5 + sqlite-vec), `Chunker`, `Embedder`, `Loader`, RRF fusion, `RAGRetriever`
- [x] `scripts/ingest_docs.py` — CLI to chunk, embed, and index personal documents
- [x] `scripts/measure_rag_latency.py` — end-to-end latency benchmark (gate p95 < 2.0 s)
- [x] ZMQ wiring: `rag_retrieval`, `rag_status` outbound; `rag_set_enabled` inbound
- [x] RAG off by default (`config.rag.enabled: false`); enable + `uv sync --extra rag` to activate
- [ ] Godot citation panel UI (deferred)
- [ ] Real avatar artwork

---

## Architecture Overview

Lumi uses a "Split-Brain" design: a Python backend handles all intelligence and audio processing; a Godot 4 frontend renders the avatar overlay. They communicate over raw TCP with 4-byte length-prefix framing.

```
Python Backend  ◄──── Raw TCP (length-prefix) ────►  Godot 4 Frontend
(Ears, Brain,          JSON wire frames               (Avatar, Animations,
 Scribe, Mouth)        127.0.0.1:5555                  Overlay)
```

To connect the frontend, set `ipc.enabled: true` in `config.yaml`, start the Python Brain first, then open `ui/project.godot` in Godot 4 and press F5.

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
| Frontend | Godot 4 (`ui/`; set `ipc.enabled: true` to activate) |
| IPC | Raw TCP, 4-byte length-prefix framing (`IPCTransport` + `ZMQServer`; no pyzmq) |
| IPC handshake | Version negotiation via `src/core/handshake.py` (`hello` / `hello_ack`) |
| Metrics | Stdlib histogram module `src/core/metrics.py` (p50/p95/p99, no external deps) |
| Testing | pytest + pytest-cov (80% coverage gate) |
| Logging | Python `logging` module (`src/core/logging_config.py`) |

---

## Pre-Alpha Notice

This is a work-in-progress. Phases 1–7 are complete: the full audio-to-speech pipeline, Godot 4 overlay, OS tools, and personal knowledge base RAG are all functional. The next milestone is MVP stabilization: graceful runtime error recovery and end-to-end integration smoke tests. See [TODO.md](TODO.md) for all identified issues and their planned solutions.
