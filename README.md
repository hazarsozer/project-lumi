# Project Lumi (ルミ)

> **"The Ghost in the Machine."**
>
> Lumi is a local, privacy-first desktop companion designed to be invisible until needed. Built with a "Zero-Cost" philosophy: runs entirely on consumer hardware without interfering with games, renders, or any foreground task.

**For the full technical design, see [ARCHITECTURE.md](ARCHITECTURE.md).**
**For the development plan and known issues, see [TODO.md](TODO.md).**

---

## Current Status

**Phase 1 and Phase 2 complete. Phase 3 foundations complete; LLM integration not started.**

| Phase | Name | Status |
|---|---|---|
| 1 | The Ears (Audio Input) | COMPLETE |
| 2 | The Scribe (Transcription) | COMPLETE |
| 3 | The Brain (Intelligence) | IN PROGRESS |
| 4 | The Mouth (TTS) | NOT STARTED |
| 5 | The Body (Visuals) | NOT STARTED |
| 6 | The Hands (OS Control) | NOT STARTED |

---

## What Works Right Now

- Say **"Hey Lumi"** → custom ONNX wake word model detects it (>0.8 confidence)
- VAD-based smart recording stops automatically when you stop speaking
- `faster-whisper` transcribes your speech to text on CPU (int8 quantized)
- Context injection reduces mis-transcriptions ("Firefox" vs "Fire folks")
- Event-driven pipeline: `Ears` posts `WakeDetectedEvent` to a central queue; `Orchestrator` dispatches all events
- Thread-safe state machine enforces `IDLE → LISTENING → PROCESSING → SPEAKING` transitions
- Typed configuration loaded from `config.yaml` with auto-detected hardware edition
- Startup validation halts on missing wake word model, wrong openwakeword version, or no microphone
- Structured logging (human-readable or JSON) via `src/core/logging_config.py`
- 83 tests with 80% coverage gate enforced in CI

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

### Phase 3: The Brain — In Progress

**Foundations complete:**
- [x] Structured logging (`logging_config.py`, `setup_logging()`)
- [x] Startup validation (`startup_check.py`, hard fail on missing models / wrong versions)
- [x] Test infrastructure (83 tests, `pytest` + `pytest-cov`, 80% coverage gate)
- [x] Configuration system + hardware edition auto-detection (`config.yaml` + `config.py`)
- [x] Event-driven orchestrator (`orchestrator.py`, replaces synchronous pipeline)
- [x] State machine (`state_machine.py`, `IDLE → LISTENING → PROCESSING → SPEAKING`)
- [x] `openwakeword==0.4.0` exact pin with startup version check

**Not yet started:**
- [ ] LLM integration (Phi-3.5 Mini / Gemma 2 2B via llama-cpp-python)
- [ ] VRAM hibernate/wake (load on demand, offload when idle)
- [ ] Reflex router + Reasoning router
- [ ] Memory (JSON user profile + conversation history)
- [ ] ZMQ server (`zmq_server.py`)

**Fine-tuning roadmap (Phase 3+):** QLoRA-based fine-tuning with LoRA adapter hot-swap architecture (<100ms domain switch). Personality injection, OS tool-call training, and multi-turn context. See [ARCHITECTURE.md § 5](ARCHITECTURE.md#5-fine-tuning-strategy-phase-3--beyond) for full strategy.

### Phase 4: The Mouth

- [ ] Kokoro-82M ONNX TTS
- [ ] Non-blocking audio playback
- [ ] Viseme extraction for lip-sync

### Phase 5: The Body

- [ ] Godot transparent overlay (X11/Wayland)
- [ ] ZeroMQ client in GDScript
- [ ] Avatar + state machine animations
- [ ] LightRAG personal knowledge base (optional, UI toggle, off by default). Users feed Lumi documents; query via natural language. See [ARCHITECTURE.md § 6](ARCHITECTURE.md#6-lightrag-optional-personal-knowledge-base-phase-5) for details.

### Phase 6: The Hands

- [ ] Vision tool (screenshot analysis)
- [ ] Automation tools (app launch, file management)
- [ ] v1.0 release

---

## Architecture Overview

Lumi uses a "Split-Brain" design: a Python backend handles all intelligence and audio processing; a Godot/Tauri frontend renders the avatar overlay. They communicate via ZeroMQ.

```
Python Backend  ◄──── ZeroMQ ────►  Godot Frontend
(Ears, Brain,                       (Avatar, Animations,
 Scribe, Mouth)                      Overlay)
```

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
| LLM (planned) | llama-cpp-python, Phi-3.5 Mini or Gemma 2 2B |
| Fine-tuning (planned) | QLoRA + llama.cpp GGUF export |
| TTS (planned) | Kokoro-82M ONNX |
| Knowledge retrieval (planned) | LightRAG (optional, Phase 5) with all-MiniLM-L6-v2 embeddings |
| Frontend (planned) | Godot 4 |
| IPC (planned) | ZeroMQ |
| Testing | pytest + pytest-cov (80% coverage gate) |
| Logging | Python `logging` module (`src/core/logging_config.py`) |

---

## Pre-Alpha Notice

This is a work-in-progress. The pipeline is event-driven as of Phase 3 foundations. The next milestone is LLM integration. See [TODO.md](TODO.md) for all identified issues and their planned solutions.
