# Project Lumi: Architecture (v2.0)

**Project Goal:** A local, privacy-first Desktop Assistant — "Siri on Steroids"
**Core Philosophy:** Architect First. Zero Cost. Local Only. Privacy by Default.

> This document is the canonical design reference. README.md links here for deep dives.
> Last updated to reflect actual state as of Phase 1 completion (2026-04-02).

---

## 1. System Architecture: "The Split-Brain"

Lumi is decoupled into two independent processes that communicate via ZeroMQ. This ensures the desktop (especially games and renderers) remains fully responsive regardless of what Lumi's brain is doing.

```
┌──────────────────────────────┐        ZeroMQ PAIR              ┌──────────────────────┐
│         THE BRAIN            │ ◄──────────────────────────────► │       THE BODY       │
│      (Python Backend)        │  JSON: {event, payload,         │  (Godot / Tauri UI)  │
│                              │         timestamp, version}     │                      │
│  Ears → Orchestrator → LLM   │                                  │  Avatar + Animations │
│  Scribe → TTS → OS Tools     │                                  │  State Overlay       │
└──────────────────────────────┘                                  └──────────────────────┘
```

### The Brain (Python Backend)
- **Role:** Intelligence, audio pipeline, OS control, IPC server
- **Tech:** Python 3.12+ managed via `uv`
- **Resource Strategy:** Hibernate & Wake
  - **Idle:** LLM offloaded to system RAM. Only wake word detection runs (CPU).
  - **Active:** LLM loaded to VRAM. Full processing pipeline engaged.

### The Body (Frontend)
- **Role:** Visual avatar, animated overlay, user-facing UI
- **Tech:** Godot Engine (transparent X11/Wayland window) — Tauri as alternative
- **Target:** < 200MB RAM, negligible GPU at all times

### The Nerves (IPC)
- **Protocol:** ZeroMQ PAIR socket
- **Format:** `{ "event": string, "payload": object, "timestamp": float, "version": string }`
- **Defined events (planned):**

| Event | Direction | Payload |
|---|---|---|
| `state_change` | Brain → Body | `{ "state": "idle" \| "listening" \| "processing" \| "speaking" }` |
| `transcript` | Brain → Body | `{ "text": string }` |
| `tts_start` | Brain → Body | `{ "text": string, "duration_ms": int }` |
| `tts_viseme` | Brain → Body | `{ "viseme": string, "duration_ms": int }` |
| `tts_stop` | Brain → Body | `{}` |
| `error` | Brain → Body | `{ "code": string, "message": string }` |
| `interrupt` | Body → Brain | `{}` |
| `user_text` | Body → Brain | `{ "text": string }` |

---

## 2. Internal Event Architecture (Planned)

The current synchronous pipeline must evolve into an event-driven model before Phase 3.

### Internal Event Types (`src/core/events.py`)

| Event | Posted by | Consumed by |
|---|---|---|
| `WakeDetectedEvent` | Ears thread | Orchestrator |
| `RecordingCompleteEvent(audio)` | Orchestrator (after VAD) | Orchestrator |
| `TranscriptReadyEvent(text)` | Scribe | Orchestrator |
| `LLMResponseReadyEvent(text)` | LLM engine | Orchestrator |
| `TTSChunkReadyEvent(audio, viseme)` | TTS engine | Speaker thread |
| `InterruptEvent` | Any source (Body via ZMQ, new wake word) | Orchestrator |
| `ShutdownEvent` | main.py / signal handler | Orchestrator |

### Pipeline Flow

```
Microphone
    │
    ▼
[Audio Queue]  ──► Ears Thread (wake word + VAD)
                         │
                         │ posts WakeDetectedEvent
                         ▼
                   [Event Queue]
                         │
                         ▼
                   Orchestrator Thread
                    ├── Reflex Router (regex commands → instant OS actions)
                    └── Reasoning Router (LLM → response generation)
                              │
                        ┌─────┴──────┐
                        ▼            ▼
                    Scribe        LLM Engine
                  (Whisper STT)  (llama-cpp-python)
                        │            │
                        └─────┬──────┘
                              ▼ posts LLMResponseReadyEvent
                         TTS Engine (Kokoro ONNX)
                              │ posts TTSChunkReadyEvent
                              ▼
                       [Audio Output Queue]
                              │
                              ▼
                         Speaker Thread
```

### Interrupt Handling

When the Orchestrator receives `InterruptEvent` while in `PROCESSING` or `SPEAKING` state:
1. Sets a cancel flag on the in-progress stage (LLM generation or TTS synthesis)
2. Drains all pending `TTSChunkReadyEvent`s from the speaker queue
3. Transitions the state machine back to `IDLE`
4. Re-enables wake word detection in the Ears thread

### State Machine (Planned)

```
          wake word
  IDLE ──────────────► LISTENING
   ▲                       │ silence detected
   │                       ▼
   │               PROCESSING (STT + LLM)
   │   InterruptEvent       │ response ready
   │◄──────────────────     ▼
   └─────────────── SPEAKING (TTS playback)
```

State transitions are published to the Body via `state_change` IPC events.

---

## 3. Performance Editions

Hardware is auto-detected at startup to select the appropriate edition.

| Feature | Lumi Light | Lumi Standard (current target) | Lumi Pro |
|---|---|---|---|
| **VRAM Budget** | < 2 GB | < 4 GB (dynamic offloading) | 8 GB+ always loaded |
| **LLM** | Qwen-1.5 1.8B / Phi-3 Mini int4 | Phi-3.5 Mini / Gemma 2 2B | Llama-3 8B / Gemma 2 9B |
| **TTS** | System TTS / Piper | Kokoro ONNX | StyleTTS2 |
| **Vision** | Disabled | On-demand (screenshots) | Real-time (camera + screen) |
| **Avatar** | Static / 2-frame | Live2D standard | 3D VRM full motion |

---

## 4. Technology Stack (Standard Edition)

### Audio Input — The Ears
| Component | Technology | Notes |
|---|---|---|
| Wake Word | openWakeWord + custom Hey Lumi ONNX model | CPU-only, always running |
| VAD | Silero VAD v5 (via openWakeWord) | Smart Stop for end-of-speech |
| STT | faster-whisper tiny.en (int8 quantized) | CPU-only, ~200ms on modern hardware |

### Audio Output — The Mouth (Phase 4)
| Component | Technology |
|---|---|
| TTS | Kokoro-82M (ONNX) |
| Playback | sounddevice (non-blocking, queued) |
| Lip-sync | Viseme extraction from TTS phoneme output |

### Intelligence — The Brain (Phase 3)
| Component | Technology |
|---|---|
| Engine | llama-cpp-python (GGUF, GPU offloading) |
| Context | Rolling window, last 10 turns |
| Routing | Reflex (regex) + Reasoning (LLM) |
| Memory | JSON-based user profile + conversation history |

### Frontend — The Body (Phase 5)
| Component | Technology |
|---|---|
| Renderer | Godot 4 (transparent overlay) |
| IPC Client | GDScript ZeroMQ client |
| Avatar | Live2D (Standard) / 3D VRM (Pro) |

### Infrastructure
| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Package Manager | `uv` |
| IPC | ZeroMQ (pyzmq) |
| Config | `config.yaml` + `src/core/config.py` (planned) |
| Logging | Python `logging` module via `src/core/logging_config.py` (planned) |
| Startup Validation | `src/core/startup_check.py` (planned) |
| Testing | `pytest` + `pytest-cov`, 80% coverage gate (planned) |

---

## 5. Actual Directory Structure

Current state as of 2026-04-02:

```
Lumi/
├── .venv/                      # Managed by uv (not committed)
├── models/                     # ONNX/GGUF model binaries (not committed)
│   └── hey_lumi.onnx           # Custom wake word model
├── src/
│   ├── main.py                 # Entry point — wires Ears + Scribe (temporary orchestrator)
│   ├── utils.py                # Shared utilities (play_ready_sound)
│   ├── audio/
│   │   ├── ears.py             # Microphone listener, wake word, VAD recording
│   │   └── scribe.py           # faster-whisper transcription
│   ├── core/                   # Placeholder — orchestrator.py planned here
│   └── llm/                    # Placeholder — model_loader.py planned here
├── ARCHITECTURE.md             # This file
├── README.md
├── SUGGESTIONS.md              # Known issues and improvement plans
├── TODO.md
└── pyproject.toml
```

Planned additions (not yet created):

```
Lumi/
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic audio, mocks)
│   ├── test_scribe.py          # Scribe.transcribe() unit tests
│   ├── test_ears.py            # VAD timeout / silence paths
│   ├── test_state_machine.py   # All valid/invalid transition branches
│   ├── test_model_loader.py    # VRAM wake/hibernate lifecycle
│   └── test_orchestrator.py    # Event routing and interrupt handling
├── src/
│   ├── core/
│   │   ├── events.py           # All typed internal events + ZMQMessage dataclass
│   │   ├── orchestrator.py     # Central event loop and component lifecycle
│   │   ├── config.py           # AudioConfig, ScribeConfig, LumiConfig + load_config()
│   │   ├── state_machine.py    # LumiState enum + transition enforcement
│   │   ├── logging_config.py   # setup_logging() — configures Python logging module
│   │   └── startup_check.py    # Pre-flight validation (model paths, mic, version pins)
│   ├── llm/
│   │   ├── model_loader.py     # VRAM hibernate/wake lifecycle (wraps llama_cpp.Llama)
│   │   └── prompt_engine.py    # Prompt templates + context window management
│   ├── audio/
│   │   └── mouth.py            # TTS engine + non-blocking speaker thread
│   ├── tools/
│   │   ├── os_actions.py       # App launch, file management
│   │   └── vision.py           # Screenshot analysis
│   └── interface/
│       └── zmq_server.py       # ZeroMQ IPC pub/sub server
├── ui/                         # Godot or Tauri frontend project
└── config.yaml                 # User settings (edition selection, model paths)
```

---

## 6. Development Roadmap

### Phase 1: The Ears (Audio Input) — COMPLETE
*Goal: Low-latency, non-blocking listening pipeline on CPU.*

- [x] Audio driver integration (`sounddevice` + `libportaudio2`)
- [x] Threaded listener (Producer-Consumer pattern)
- [x] Custom "Hey Lumi" wake word model (openWakeWord ONNX, >0.8 confidence)
- [x] VAD smart stop (`silero-vad`, silence threshold calibrated)
- [x] Double-trigger fix (cooldown + flush logic)
- [x] Latency tuning (`latency='high'` to reduce buffer underruns) ⚠️ Needs monitoring

### Phase 2: The Scribe (Transcription) — IN PROGRESS
*Goal: Accurate speech-to-text without GPU.*

- [x] Model integration (`faster-whisper` tiny.en, int8, CPU)
- [x] Context injection (initial prompt for proper noun accuracy)
- [ ] Command parsing (regex/keyword matching for instant actions)

### Phase 3: The Brain (Intelligence) — NOT STARTED
*Goal: Smart decision-making using local LLMs.*

- [ ] Structured logging (`src/core/logging_config.py`, replace all `print()`)
- [ ] Startup validation (`src/core/startup_check.py`, hard `RuntimeError` on bad state)
- [ ] Test infrastructure (`pytest` + `pytest-cov`, `--cov-fail-under=80`)
- [ ] Configuration system (`config.yaml` + `config.py` + `detect_edition()`)
- [ ] Typed internal events (`src/core/events.py`, all 7 event types + `ZMQMessage`)
- [ ] Event-driven orchestrator (`src/core/orchestrator.py`, replaces synchronous chain)
- [ ] State machine (`src/core/state_machine.py`, enforced transitions + IPC publication)
- [ ] LLM integration (Phi-3.5 Mini or Gemma 2 2B via llama-cpp-python)
- [ ] VRAM hibernate/wake logic (`src/llm/model_loader.py`)
- [ ] Reflex router (regex commands → OS actions)
- [ ] Reasoning router (LLM-based response)
- [ ] Memory system (JSON user profile + rolling conversation history)

### Phase 4: The Mouth (TTS) — NOT STARTED
*Goal: High-quality voice response without GPU.*

- [ ] TTS engine (Kokoro-82M ONNX)
- [ ] Non-blocking audio playback (queued speaker thread)
- [ ] Viseme extraction for avatar lip-sync

### Phase 5: The Body (Visuals) — NOT STARTED
*Goal: Transparent, interactive desktop overlay.*

- [ ] Godot project setup (transparent window, X11/Wayland)
- [ ] ZeroMQ client (GDScript, receives state events)
- [ ] Avatar rendering (Live2D or 2D sprite)
- [ ] State machine animation (Idle / Listening / Thinking / Speaking)

### Phase 6: The Hands (OS Control) — NOT STARTED
*Goal: Lumi can act on the desktop.*

- [ ] Vision tool (screenshot capture + analysis)
- [ ] Automation tools (app launch, file management, clipboard)
- [ ] v1.0 release
