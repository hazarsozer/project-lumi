# Project Lumi (ルミ)

> **"The Ghost in the Machine."**
>
> Lumi is a local, privacy-first desktop companion designed to be invisible until needed. Built with a "Zero-Cost" philosophy: runs entirely on consumer hardware without interfering with games, renders, or any foreground task.

**For the full technical design, see [ARCHITECTURE.md](ARCHITECTURE.md).**
**For the development plan and known issues, see [TODO.md](TODO.md).**

---

## Current Status

**Phase 1 complete. Phase 2 in progress.**

| Phase | Name | Status |
|---|---|---|
| 1 | The Ears (Audio Input) | Complete |
| 2 | The Scribe (Transcription) | In Progress |
| 3 | The Brain (Intelligence) | Not Started |
| 4 | The Mouth (TTS) | Not Started |
| 5 | The Body (Visuals) | Not Started |
| 6 | The Hands (OS Control) | Not Started |

---

## What Works Right Now

- Say **"Hey Lumi"** → custom ONNX wake word model detects it (>0.8 confidence)
- VAD-based smart recording stops automatically when you stop speaking
- `faster-whisper` transcribes your speech to text on CPU (int8 quantized)
- Context injection reduces mis-transcriptions ("Firefox" vs "Fire folks")

---

## Roadmap

### Phase 1: The Ears — Complete

- [x] Custom "Hey Lumi" wake word model (openWakeWord ONNX)
- [x] VAD smart stop (silero-vad)
- [x] Threaded Producer-Consumer audio pipeline
- [x] Double-trigger cooldown fix
- [x] Latency tuning ⚠️ Monitoring for buffer underruns

### Phase 2: The Scribe — In Progress

- [x] faster-whisper int8 transcription (CPU)
- [x] Context injection for accuracy
- [ ] Command parsing (regex/keyword matching)

### Phase 3: The Brain

- [ ] Structured logging (`logging_config.py`, replace all `print()`)
- [ ] Startup validation (`startup_check.py`, hard fail on missing models / wrong versions)
- [ ] Test infrastructure (`pytest` + `pytest-cov`, 80% coverage gate)
- [ ] Configuration system + hardware edition auto-detection
- [ ] Event-driven orchestrator (replaces current synchronous pipeline)
- [ ] State machine (Idle → Listening → Processing → Speaking)
- [ ] LLM integration (Phi-3.5 Mini / Gemma 2 2B via llama-cpp-python)
- [ ] VRAM hibernate/wake (load on demand, offload when idle)
- [ ] Reflex router + Reasoning router
- [ ] Memory (JSON user profile + conversation history)

### Phase 4: The Mouth

- [ ] Kokoro-82M ONNX TTS
- [ ] Non-blocking audio playback
- [ ] Viseme extraction for lip-sync

### Phase 5: The Body

- [ ] Godot transparent overlay (X11/Wayland)
- [ ] ZeroMQ client in GDScript
- [ ] Avatar + state machine animations

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
| TTS (planned) | Kokoro-82M ONNX |
| Frontend (planned) | Godot 4 |
| IPC (planned) | ZeroMQ |
| Testing (planned) | pytest + pytest-cov (80% coverage gate) |
| Logging (planned) | Python `logging` module |

---

## Pre-Alpha Notice

This is a work-in-progress. The current pipeline is functional but the concurrency model is synchronous — it will be replaced with an event-driven orchestrator in Phase 3. See [TODO.md](TODO.md) for all identified issues and their planned solutions.
