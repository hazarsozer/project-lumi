# Project Lumi: Architecture (v2.0)

**Project Goal:** A local, privacy-first Desktop Assistant — "Siri on Steroids"
**Core Philosophy:** Architect First. Zero Cost. Local Only. Privacy by Default.

> This document is the canonical design reference. README.md links here for deep dives.
> Last updated to reflect actual state as of Phase 6 complete (2026-04-17).

---

## 1. System Architecture: "The Split-Brain"

Lumi is decoupled into two independent processes that communicate via ZeroMQ. This ensures the desktop (especially games and renderers) remains fully responsive regardless of what Lumi's brain is doing.

```
┌──────────────────────────────┐        Raw TCP (length-prefix)  ┌──────────────────────┐
│         THE BRAIN            │ ◄──────────────────────────────► │       THE BODY       │
│      (Python Backend)        │  JSON: {event, payload,         │  (Godot 4 UI)        │
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
- **Protocol:** Raw TCP, 4-byte big-endian uint32 length prefix + UTF-8 JSON body
- **Transport class:** `IPCTransport` (`src/core/ipc_transport.py`) — single-client, two daemon threads (`ipc-accept`, `ipc-recv`), stdlib `socket` only (no pyzmq)
- **Event bridge:** `ZMQServer` (`src/core/zmq_server.py`) — sits on top of `IPCTransport`, translates outbound internal events to JSON wire frames, translates inbound JSON frames to internal events posted to the orchestrator queue
- **Enabled by:** `config.ipc.enabled: true` in `config.yaml` (default `false`; keep `false` for headless / CI runs)
- **Default endpoint:** `tcp://127.0.0.1:5555`
- **Format:** `{ "event": string, "payload": object, "timestamp": float, "version": string }`
- **Wire-format schema:** `ZMQMessage` dataclass in `src/core/events.py`. **IPC event types:**

| Event | Direction | Payload |
|---|---|---|
| `state_change` | Brain → Body | `{ "state": "idle" \| "listening" \| "processing" \| "speaking" }` |
| `transcript` | Brain → Body | `{ "text": string }` |
| `tts_start` | Brain → Body | `{ "text": string, "duration_ms": int }` |
| `tts_viseme` | Brain → Body | `{ "viseme": string, "duration_ms": int }` |
| `tts_stop` | Brain → Body | `{}` |
| `llm_token` | Brain → Body | `{ "token": string, "utterance_id": string }` |
| `error` | Brain → Body | `{ "code": string, "message": string }` |
| `interrupt` | Body → Brain | `{}` |
| `user_text` | Body → Brain | `{ "text": string }` |

---

## 2. Internal Event Architecture

The pipeline is event-driven. All components post typed, frozen dataclass events to a central `queue.Queue` owned by the `Orchestrator`.

### Internal Event Types (`src/core/events.py`)

| Event | Posted by | Consumed by |
|---|---|---|
| `WakeDetectedEvent` | Ears thread | Orchestrator |
| `RecordingCompleteEvent` | Ears thread (after VAD) | Orchestrator |
| `TranscriptReadyEvent` | Scribe | Orchestrator |
| `CommandResultEvent` | Orchestrator (command parser) | Orchestrator |
| `LLMResponseReadyEvent` | LLM engine | Orchestrator |
| `TTSChunkReadyEvent` | TTS engine | Speaker thread |
| `VisemeEvent` | TTS engine (mouth.py) | Orchestrator / ZMQ server (lip-sync) |
| `SpeechCompletedEvent` | SpeakerThread / KokoroTTS | Orchestrator |
| `LLMTokenEvent` | LLM reasoning router | Orchestrator (streaming tokens) |
| `InterruptEvent` | Any source (Body via ZMQ, new wake word) | Orchestrator |
| `ShutdownEvent` | main.py / signal handler | Orchestrator |
| `UserTextEvent` | ZMQ server (Body → Brain) | Orchestrator |

`ZMQMessage` is also defined in `src/core/events.py` as the wire-format dataclass for ZMQ IPC communication (`event`, `payload`, `timestamp`, `version`).

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

### State Machine (`src/core/state_machine.py`)

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

The `LumiState` enum defines exactly four states: `IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`. The `StateMachine` class enforces all valid transitions and notifies registered observers. `InvalidTransitionError` is raised for any illegal transition attempt.

State transitions are published to the Body via `state_change` IPC events. `ZMQServer` registers itself as a `StateMachine` observer so every transition is forwarded automatically when the IPC server is enabled.

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
| Renderer | Godot 4 (transparent 200×200 overlay, borderless, X11/Wayland) |
| IPC Client | `LumiClient` (GDScript `StreamPeerTCP`, auto-reconnect every 2 s) |
| Frame protocol | `ipc_protocol.gd` — 4-byte length-prefix encode/decode |
| Avatar | `AvatarController` drives `AnimatedSprite2D` from Brain state events; placeholder colored-circle sprites in `ui/assets/sprites/` |
| Avatar (Phase 6) | Real artwork replacing placeholder sprites; Live2D (Standard) / 3D VRM (Pro) |

### Infrastructure
| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Package Manager | `uv` |
| IPC | Raw TCP with 4-byte length-prefix framing (`IPCTransport` + `ZMQServer`; stdlib `socket`, no pyzmq) |
| Config | `config.yaml` + `src/core/config.py` (`LumiConfig`, `AudioConfig`, `ScribeConfig`, `LLMConfig`, `TTSConfig`, `IPCConfig`, `load_config()`, `detect_edition()`) |
| Logging | Python `logging` module via `src/core/logging_config.py` (`setup_logging()`) |
| Startup Validation | `src/core/startup_check.py` (`run_startup_checks()`) |
| Testing | `pytest` + `pytest-cov`, 80% coverage gate (`tests/` directory, 363 tests) |
| CI | `.github/workflows/ci.yml` |

### OS Tools — The Hands (Phase 6)
| Component | Technology | Notes |
|---|---|---|
| Tool Protocol | `src/tools/base.py` — `Tool` `@runtime_checkable` Protocol + frozen `ToolResult` dataclass | Standard interface for all tools |
| Tool Registry | `src/tools/registry.py` — `register()`/`get()`/`list_tools()` | Warns on name collision |
| Tool Executor | `src/tools/executor.py` — allowlist gate + `threading.Event` timeout + cancel flag | Single entry point for all tool invocations |
| AppLaunchTool | `src/tools/os_actions.py` | `shutil.which` validation + internal allowlist + `subprocess.Popen` |
| ClipboardTool | `src/tools/os_actions.py` | `xclip` read/write; graceful fail if absent |
| FileInfoTool | `src/tools/os_actions.py` | `Path.parts` traversal guard; stat metadata only |
| WindowListTool | `src/tools/os_actions.py` | `wmctrl -l` parse; graceful fail if absent |
| ScreenshotTool | `src/tools/vision.py` | grim → scrot → Pillow fallback; moondream2 GGUF description; 30s idle unload; VRAM mutex with LLM |
| Viseme extraction | `src/audio/viseme_map.py` + `src/audio/mouth.py` | 8 viseme groups; `map_phoneme()` strips stress digits; `VisemeEvent` posted per phoneme |
| Token streaming | `src/llm/reasoning_router.py` + `src/core/zmq_server.py` | `LLMTokenEvent` per token; `utterance_id` UUID threads through; `llm_token` wire frame to Godot |
| Config | `ToolsConfig` + `VisionConfig` in `src/core/config.py`; `tools:` + `vision:` keys in `config.yaml` | |

**Tool-call flow (two-pass):** `_run_inference` → LLM generates `<tool_call>` block → `ToolExecutor.execute()` → result injected into conversation → second LLM pass → `LLMResponseReadyEvent`.

---

## 5. Fine-Tuning Strategy (Phase 3 & Beyond)

### Overview

Out of the box, Phi-3.5 Mini claims to be "a large language model by Microsoft" and will refuse benign OS operations it doesn't recognize. Fine-tuning shapes behavior, tone, and structured output format without retraining from scratch.

### Canonical Personality Definition

All training data must conform to this character spec. Consistency matters more than cleverness.

```
Name:        Lumi
Pronouns:    they/them (neutral, non-gendered)
Voice style: Calm, concise, slightly warm. Never condescending. No filler phrases
             ("Certainly!", "Of course!", "Great question!"). Gets to the point.
Awareness:   Knows it runs locally on the user's machine. Never claims internet access
             unless the internet tool is active. Knows its own VRAM budget and model size.
Limits:      Honest about what it can't do. Does not hallucinate capabilities.
             "I don't have access to that right now" > making something up.
Expertise:   Power-user assistant. Comfortable with code, system tasks, terminal output.
             Treats the user as a capable adult.
```

**Anti-patterns to train away (explicitly in every training batch):**
- Never start a response with "Certainly!", "Of course!", "Sure!", "Great question!"
- Never use markdown in voice responses (`**bold**`, `# headers`)
- Never claim internet access if the internet tool is disabled
- Never say "As an AI language model, I..."

**Recommended approach: QLoRA** (Quantized LoRA) reduces full fine-tuning's 24GB VRAM requirement to ~6–8GB, viable on a single RTX 3060/3080.

### LoRA Adapter Hot-Swap Architecture

Rather than loading multiple full GGUF models, keep one base model in VRAM permanently and domain-specific LoRA adapters (~10-50 MB each) are swapped in <100ms.

**VRAM Budget (Q4_K_M):**

| Component | VRAM |
|---|---|
| Base model (Phi-3.5 Mini) | ~2.2 GB |
| Audio pipeline (openwakeword + faster-whisper) | ~0.3 GB |
| KV cache | ~0.3 GB |
| LoRA adapters (per active) | ~0.05 GB |
| **Total** | ~2.85 GB (comfortable within 3.8 GB budget) |

**Prerequisites:** Before building hot-swap infrastructure, verify that `llama-cpp-python>=0.2.90` exposes `llama_lora_adapter_set` / `llama_lora_adapter_remove` via a quick validation test. If unavailable, fall back to pre-merged GGUFs with `ModelRegistry`.

### Domain Router

Classifies transcripts to select the appropriate LoRA adapter:

- **Option A (start here): Regex classifier** — <1ms latency, ~70-80% accuracy. Wrong classifications degrade quality, not correctness.
- **Option B (upgrade path): Embedding similarity** — `all-MiniLM-L6-v2`, ~10-30ms latency, 85-90% accuracy. Only build if regex miss rate >20% in production.

### Dataset Strategy: 5 Categories

| Category | Priority | Count | Content |
|---|---|---|---|
| 1: Identity & Personality | HIGH | ~200 | Who is Lumi, voice style, limitations |
| 2: Brevity & Voice-Friendliness | HIGH | ~150 | Short, spoken answers (no markdown) |
| 3: OS Control | HIGH | 400–500 | Tool calls, JSON schema, safety |
| 4: Code Generation | MEDIUM | ~150 | Snippets, no preamble |
| 5: Internet Tools | LOW | ~200 | Web search, fetch (Phase 5+) |
| 6: Multi-Turn Context | MEDIUM | ~100 | Conversational flow, context preservation |

Total: ~1000–1200 examples for full personality + tool-call training.

### Training Workflow

1. Generate dataset (synthetic via Claude/GPT-4 ~80%, manual curation ~15%, live sessions ~5%)
2. Format to model's native chat template
3. QLoRA fine-tune (`r=16` for personality, `r=32` for tool-use) — 90/10 train/val split
4. Evaluate held-out set at FP16 (check overfitting)
5. Merge LoRA → base model
6. Evaluate merged at FP16 (pre-quantization baseline)
7. Convert → GGUF (llama.cpp `convert_hf_to_gguf.py`)
8. Quantize to Q4_K_M
9. **Critical step:** Evaluate Q4_K_M vs FP16 baseline — if quality delta >1pt, use Q5_K_M or Q6_K instead
10. Drop into `models/llm/` and run full evaluation checklist

### Tool Call Format & Parser

Format: `<tool_call>{...}</tool_call>` (XML-style delimiters, JSON payload)

All result-returning tools must use inline `[TOOL_RESULT: ...]` injection (no closing tag):

```
User: Run git status.
Lumi: <tool_call>{"tool": "terminal.run", "args": {"command": "git status"}}</tool_call>
[TOOL_RESULT: On branch main\nnothing to commit]
Lumi: Clean — nothing to commit.
```

**`ToolCallParser` class** (`src/llm/tool_call_parser.py`) handles parsing, validation, and recovery:
- Extracts all `<tool_call>` blocks (supports multi-call)
- Validates tool names against `VALID_TOOLS` registry
- Best-effort fix for common JSON errors (unescaped quotes, trailing commas)
- `extract_spoken_text()` strips all tool tags for TTS

### Versioning Scheme

```
lumi-{base}-v{version}-{quant}.gguf

Examples:
  lumi-phi35-v1-Q4_K_M.gguf              # personality + brevity
  lumi-phi35-v2-Q4_K_M.gguf              # + OS tools
  lumi-phi35-chat-v1-Q4_K_M.gguf         # specialist: chat domain
  lumi-phi35-os-v1-Q4_K_M.gguf           # specialist: OS control domain
```

### Evaluation Checklist

**Automated (80%+ coverage via tests):**
- Identity questions confirm "Lumi", no base model name
- OS command prompts emit valid `<tool_call>` JSON
- Response word counts <80 for single-turn factuals
- Negative assertions: no "Certainly!", "Of course!", markdown, "language model"
- Base-capability regression (general knowledge + code unchanged >1pt)

**Manual:**
- 10 identity questions — all confirm "Lumi"
- 5 dangerous commands — all ask for confirmation
- 5 multi-turn conversations — context maintained
- Voice output check — read 10 responses aloud

### Phased Rollout

| Phase | Model | Dataset | New Capabilities |
|---|---|---|---|
| v0 (now) | Stock Phi-3.5 Mini Q4_K_M | None | Baseline |
| v1 | lumi-phi35-v1 | Cat. 1+2 (~350) | Lumi identity, voice brevity |
| v2 | lumi-phi35-v2 | + Cat. 3 (~750–850) | OS tool calls |
| v3 | lumi-phi35-v3 | + Cat. 4+6 (~1000) | Code style, multi-turn |
| v4 | lumi-phi35-v4 | + Cat. 5 (~1200) | Internet tools (Phase 5+) |

### Open Questions (Fine-Tuning)

1. **LoRA API availability:** Does `llama-cpp-python>=0.2.90` expose `llama_lora_adapter_remove`? Run `hasattr(model, "set_lora")` before building hot-swap infrastructure. Highest-priority investigation.
2. **System prompt vs fine-tuning tradeoff:** Some behaviors (brevity, no filler) can be enforced via system prompt without fine-tuning. Measure the system-prompt baseline first before investing in fine-tuning for those behaviors.
3. **Voice-specific symbol avoidance:** TTS reads code symbols aloud badly. A post-processing step in `PromptEngine.extract_response()` may be sufficient; a fine-tune that naturally avoids symbols in voice contexts would be cleaner.
4. **Catastrophic forgetting:** Run base-capability regression tests (general knowledge + code) before and after each fine-tuning version. If the fine-tuned average drops >1pt, reduce LoRA rank.
5. **Multi-call tool ordering:** When the user requests two actions in sequence, should Lumi execute sequentially (safer, slower) or emit both calls in one response (faster, requires orchestrator concurrent tool execution)? Training data must be consistent.

---

## 6. LightRAG: Optional Personal Knowledge Base (Phase 6)

### What It Does

- **Standard RAG:** chunk documents → embed → vector similarity search → stuff context
- **LightRAG:** extracts entities and relationships, builds a knowledge graph, performs dual-level retrieval (local facts + global relationships)

**Complementary to LoRA** (personality) **but competes for context window and VRAM budget** — if personality LoRA is trained, retrain with 50–100 `[CONTEXT]` block examples before deploying LightRAG.

### Architectural Fit

| Mechanism | Role |
|---|---|
| Base LLM (Phi-3.5 Mini) | Reasoning |
| LoRA adapters | Personality, behavior, OS tool-call schema |
| LightRAG | External factual knowledge retrieval |

### Token Budget (Hard Cap)

| Item | Tokens |
|---|---|
| System prompt | ~120 |
| Retrieved context (LightRAG) | **600 max** |
| Conversation history (3–4 turns) | ~800 |
| Current user query | ~50 |
| Generation headroom | ~512 |
| Safety margin | ~200 |
| **Total** | ~2,280 of 4,096 |

### Technical Stack

- **Embedding model:** `all-MiniLM-L6-v2` (~80MB, 10–30ms CPU inference, 384-dim vectors)
- **Graph storage:** SQLite (zero-config, <50ms cold-start, crash-safe)
- **Prompt construction:**
  ```
  [system prompt]
  [conversation history (last N turns)]
  [CONTEXT]
  <retrieved chunks — max 600 tokens>
  [/CONTEXT]
  [current user query]
  ```

### Integration Point: No New Events

RAG retrieval is a pre-processing step inside `ReasoningRouter.route()`, gated by a flag from `_on_transcript_ready`:

- `_on_transcript_ready`: Check transcript against RAG trigger regex
- `ReasoningRouter.route(text, cancel_flag, event_queue, rag_enabled=False)`:
  1. If `rag_enabled`: query LightRAG → `retrieved_context` (max 600 tokens)
  2. `PromptEngine.format_prompt(text, history, model_family, retrieved_context=...)`
  3. `ModelLoader.generate(prompt, cancel_flag)`

**Files modified (all in `src/llm/`):**
- `reasoning_router.py`: optional `rag_enabled` param to `route()`
- `prompt_engine.py`: optional `retrieved_context` param to `format_prompt()`
- `orchestrator.py`: RAG trigger check in `_on_transcript_ready`
- `rag_retriever.py` (new): encapsulates LightRAG query, token budget, result formatting

### Trigger Models

**Option A: Explicit Skill (Phase 5 — implement first)**

Add to `domain_router.py` patterns: `\b(search my docs|look up in my notes|check my knowledge base)\b` → sets `rag_enabled=True` on reasoning router call. Clear user expectation. Failure mode: no relevant nodes → tell the user.

**Option B: Automatic Routing (Phase 6 — after classifier proven)**

Embedding-based classifier (Option A validated + >90% precision). Silent fallback if no match.

**Option C: Hybrid (Target state)**

Explicit for document search, automatic for knowledge queries. Post-Option B.

### Phase Placement & Prerequisites

| Phase | Content |
|---|---|
| Phase 3 | LLM pipeline |
| Phase 4 | TTS + VRAM/latency benchmarking |
| Phase 5 | Godot frontend + IPC transport (complete; LightRAG deferred) |
| Phase 6 | OS tools + LightRAG Option A (explicit skill) + Option B/C (automatic, if classifier proven) |

**Must complete before LightRAG work:**
1. Phase 3 LLM pipeline stable
2. Phase 4 TTS integrated + end-to-end latency measured
3. `all-MiniLM-L6-v2` CPU latency benchmarked on target hardware
4. If personality LoRA in use: retrain with `[CONTEXT]` examples

**Go/No-Go gate:** If end-to-end latency after Phase 4 exceeds 2 seconds, defer LightRAG until base pipeline optimized (150–600ms RAG retrieval would push past 3-second voice UI threshold).

### UI/UX

- **Surfaced as UI toggle** (off by default, similar to Lumi Pro camera detection)
- "Searching documents…" animation masks retrieval latency
- User commands: "search my docs for X", dedicated UI panel for document uploads
- Explicit "remove document" and "re-index" commands for graph maintenance

---

## 7. Actual Directory Structure

Current state as of 2026-04-13 (Phase 5 complete):

```
Lumi/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI pipeline (lint, type check, pytest --cov-fail-under=80)
├── .venv/                      # Managed by uv (not committed)
├── models/                     # ONNX/GGUF model binaries (not committed)
│   └── hey_lumi.onnx           # Custom wake word model
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures: synthetic audio arrays, sounddevice mock,
│   │                           #   faster-whisper mock, openwakeword mock, mock_llama_cpp
│   ├── test_ears.py            # Ears: wake word detection, VAD recording paths
│   ├── test_events.py          # All event types + ZMQMessage construction
│   ├── test_ipc_protocol_conformance.py  # Integration: wire protocol round-trips (6 tests, @pytest.mark.integration)
│   ├── test_ipc_transport.py   # IPCTransport: bind, send, recv, stop (7 tests)
│   ├── test_memory.py          # ConversationMemory: add_turn, prune, JSON persistence
│   ├── test_model_loader.py    # ModelLoader: load/unload lifecycle, path validation
│   ├── test_mouth.py           # KokoroTTS: synthesize, cancel, is_busy
│   ├── test_orchestrator.py    # Event routing, interrupt handling, shutdown
│   ├── test_prompt_engine.py   # PromptEngine: ChatML format, token budget truncation
│   ├── test_reasoning_router.py # ReasoningRouter: token-by-token generation, cancel
│   ├── test_reflex_router.py   # ReflexRouter: greeting + time patterns
│   ├── test_scribe.py          # Scribe.transcribe() unit tests
│   ├── test_speaker.py         # SpeakerThread: playback, resampling, SpeechCompletedEvent
│   ├── test_state_machine.py   # All valid/invalid transition branches
│   ├── test_tool_call_parser.py # parse_tool_calls: extraction, validation, recovery
│   ├── test_utils.py           # play_ready_sound() unit tests
│   └── test_zmq_server.py      # ZMQServer: outbound events, inbound parsing, lifecycle (16 tests)
├── src/
│   ├── __init__.py
│   ├── main.py                 # Thin bootstrap: logging → config → checks → orchestrator
│   ├── utils.py                # Shared utilities (play_ready_sound)
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── ears.py             # Microphone capture, wake word detection (openWakeWord),
│   │   │                       #   VAD recording; posts WakeDetectedEvent + RecordingCompleteEvent
│   │   ├── mouth.py            # KokoroTTS: sentence-level streaming, prepare()/synthesize()/cancel()/is_busy;
│   │   │                       #   posts TTSChunkReadyEvent, VisemeEvent
│   │   ├── scribe.py           # faster-whisper STT transcription; posts TranscriptReadyEvent
│   │   └── speaker.py          # SpeakerThread: daemon audio playback with resampling;
│   │                           #   posts SpeechCompletedEvent on final chunk
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # LumiConfig, AudioConfig, ScribeConfig, LLMConfig, TTSConfig, IPCConfig,
│   │   │                       #   load_config(), detect_edition()
│   │   ├── events.py           # Frozen event types + ZMQMessage dataclass
│   │   ├── ipc_transport.py    # IPCTransport: single-client TCP server, 4-byte length-prefix framing,
│   │   │                       #   two daemon threads (ipc-accept, ipc-recv), stdlib socket only
│   │   ├── logging_config.py   # setup_logging() — human-readable or JSON structured output
│   │   ├── orchestrator.py     # Orchestrator: event queue, handler dispatch, interrupt handling,
│   │   │                       #   TranscriptReadyEvent → ReflexRouter / ReasoningRouter wiring,
│   │   │                       #   ZMQServer injection, _handle_user_text handler
│   │   ├── startup_check.py    # run_startup_checks(): hard/soft pre-flight validation
│   │   ├── state_machine.py    # LumiState enum (IDLE/LISTENING/PROCESSING/SPEAKING),
│   │   │                       #   StateMachine, InvalidTransitionError, unregister_observer()
│   │   └── zmq_server.py       # ZMQServer: event translation bridge over IPCTransport;
│   │                           #   Brain → Body (state_change, transcript, tts_start, tts_viseme,
│   │                           #   tts_stop, error); Body → Brain (interrupt, user_text)
│   └── llm/
│       ├── __init__.py         # Public exports: ReflexRouter, ReasoningRouter, parse_tool_calls,
│       │                       #   ConversationMemory, ModelLoader, PromptEngine
│       ├── memory.py           # JSON-persisted conversation history (ConversationMemory)
│       ├── model_loader.py     # VRAM hibernate/wake lifecycle (wraps llama_cpp.Llama)
│       ├── prompt_engine.py    # ChatML prompt assembly + token-budget truncation
│       ├── reasoning_router.py # Token-by-token LLM inference with cancel flag support
│       ├── reflex_router.py    # Regex fast-path: greetings, time queries
│       └── tool_call_parser.py # <tool_call> extractor + JSON recovery (parse_tool_calls)
├── ui/                         # Godot 4 frontend project
│   ├── project.godot           # Godot project descriptor
│   ├── assets/
│   │   └── sprites/            # Placeholder colored-circle sprites (Phase 5); real art in Phase 6
│   ├── scenes/
│   │   ├── avatar.tscn         # AnimatedSprite2D scene
│   │   └── main.tscn           # Root scene (wires client signals)
│   ├── scripts/
│   │   ├── avatar_controller.gd # Drives AnimatedSprite2D from Brain state events
│   │   ├── ipc_protocol.gd     # 4-byte length-prefix frame encode/decode
│   │   ├── lumi_client.gd      # StreamPeerTCP client with auto-reconnect (2 s retry)
│   │   └── main.gd             # Root scene logic: wires signals, Escape → interrupt
│   ├── README.md               # Godot setup and running instructions
│   └── TESTING.md              # Manual test checklist for Godot frontend
├── config.yaml                 # Runtime configuration (all keys optional, defaults in config.py)
├── ARCHITECTURE.md             # This file
├── README.md
├── SUGGESTIONS.md              # Known issues and improvement plans
├── TODO.md
└── pyproject.toml              # runtime / llm / training / tts / dev optional dep groups
```

Planned additions (not yet created):

```
src/
├── llm/
│   ├── domain_router.py        # Regex/embedding-based domain classification (Phase 3+)
│   ├── model_registry.py       # Fallback: full GGUF model swapping (Phase 3+, if LoRA API unavailable)
│   └── rag_retriever.py        # LightRAG query wrapper, token budget enforcement (Phase 6 optional)
├── tools/
│   ├── os_actions.py           # App launch, file management (Phase 6)
│   └── vision.py               # Screenshot analysis (Phase 6)
scripts/
├── train_lumi.py               # QLoRA training entrypoint (Phase 3+)
└── merge_lora.py               # Adapter merge + GGUF export (Phase 3+)
```

---

## 8. Development Roadmap

### Phase 1: The Ears (Audio Input) — COMPLETE
*Goal: Low-latency, non-blocking listening pipeline on CPU.*

- [x] Audio driver integration (`sounddevice` + `libportaudio2`)
- [x] Threaded listener (Producer-Consumer pattern)
- [x] Custom "Hey Lumi" wake word model (openWakeWord ONNX, >0.8 confidence)
- [x] VAD smart stop (`silero-vad`, silence threshold calibrated)
- [x] Double-trigger fix (cooldown + flush logic)
- [x] Latency tuning (`latency='high'` to reduce buffer underruns) ⚠️ Needs monitoring

### Phase 2: The Scribe (Transcription) — COMPLETE
*Goal: Accurate speech-to-text without GPU.*

- [x] Model integration (`faster-whisper` tiny.en, int8, CPU)
- [x] Context injection (initial prompt for proper noun accuracy)
- [x] Command parsing infrastructure (event-driven pipeline wired; `CommandResultEvent` defined)

### Phase 3: The Brain (Intelligence) — IN PROGRESS
*Goal: Smart decision-making using local LLMs.*

**Foundations — COMPLETE:**
- [x] Structured logging (`src/core/logging_config.py`, `setup_logging()`)
- [x] Startup validation (`src/core/startup_check.py`, `run_startup_checks()`)
- [x] Test infrastructure (`tests/` directory, `--cov-fail-under=80`)
- [x] Configuration system (`config.yaml` + `src/core/config.py` + `detect_edition()`)
- [x] Typed internal events (`src/core/events.py`, 9 event types + `ZMQMessage`)
- [x] Event-driven orchestrator (`src/core/orchestrator.py`, replaces synchronous chain)
- [x] State machine (`src/core/state_machine.py`, enforced transitions)
- [x] `openwakeword==0.4.0` exact pin with startup version check

**LLM Modules — COMPLETE (Waves 0–3):**
- [x] `src/llm/model_loader.py` — VRAM hibernate/wake via llama-cpp-python (8 tests)
- [x] `src/llm/prompt_engine.py` — ChatML prompt assembly + token-budget truncation (7 tests)
- [x] `src/llm/memory.py` — JSON-persisted conversation history (9 tests)
- [x] `src/llm/reflex_router.py` — regex fast-path: greetings, time queries (8 tests)
- [x] `src/llm/reasoning_router.py` — token-by-token inference with cancel support (6 tests)
- [x] `src/llm/tool_call_parser.py` — `<tool_call>` extractor + JSON recovery (10 tests)
- [x] `src/llm/__init__.py` — public exports for all 6 modules
- [x] `[project.optional-dependencies] llm` group in `pyproject.toml`
- [x] Replace `print()` → `logger.info()` in `src/audio/scribe.py`
- [x] `Orchestrator._handle_transcript()` — reflex fast-path + reasoning daemon thread wired

**Remaining:**
- [ ] Coverage gate ≥80% on all `src/llm/` and `src/core/` modules (Wave 4)
- [ ] Full code review (Wave 4)

### Phase 4: The Mouth (TTS) — COMPLETE
*Goal: High-quality voice response without GPU.*

- [x] TTS engine (Kokoro-82M ONNX) — `src/audio/mouth.py`, KokoroTTS with sentence streaming
- [x] Non-blocking audio playback (SpeakerThread, `src/audio/speaker.py`)
- [x] TTS config keys (`TTSConfig` in `config.py`, `tts:` section in `config.yaml`, startup check)
- [x] Viseme extraction for avatar lip-sync (`viseme_map.py` + `mouth.py`; `VisemeEvent` fully wired — Phase 6)

### Phase 5: The Body (Visuals) — COMPLETE
*Goal: Transparent, interactive desktop overlay + IPC transport to Python Brain.*

- [x] `src/core/ipc_transport.py` — raw TCP server, 4-byte big-endian length prefix, single-client, two daemon threads
- [x] `src/core/zmq_server.py` — event translation bridge: internal events ↔ JSON wire protocol
- [x] `src/core/state_machine.py` — `unregister_observer()` added
- [x] `src/core/config.py` — `IPCConfig.enabled` field added (default `false`)
- [x] `src/core/orchestrator.py` — ZMQServer injection, `_handle_user_text` handler, shutdown cleanup
- [x] `src/main.py` — ZMQServer auto-created when `config.ipc.enabled = true`
- [x] Godot 4 frontend scaffold (`ui/`) — transparent 200×200 borderless overlay, `StreamPeerTCP` client, avatar controller, IPC protocol GDScript
- [x] `tests/test_ipc_transport.py` — 7 tests
- [x] `tests/test_zmq_server.py` — 16 tests
- [x] `tests/test_ipc_protocol_conformance.py` — 6 integration tests
- [ ] LightRAG Option A (explicit skill trigger, UI toggle, off by default — deferred to Phase 6)
- [ ] Real avatar artwork (placeholder colored-circle sprites used — deferred to Phase 6)

### Phase 6: The Hands (OS Control) — COMPLETE
*Goal: Lumi can act on the desktop. Token streaming. Viseme lip-sync. Advanced RAG routing.*

- [x] `src/tools/` package — `Tool` Protocol, `ToolRegistry`, `ToolExecutor` (allowlist + timeout)
- [x] OS tools: `AppLaunchTool`, `ClipboardTool`, `FileInfoTool`, `WindowListTool` (`src/tools/os_actions.py`)
- [x] Vision tool — `ScreenshotTool` with grim→scrot→Pillow fallback + moondream2 GGUF description (`src/tools/vision.py`)
- [x] LLM token streaming — `LLMTokenEvent` per token; `llm_token` wire frame to Godot
- [x] Viseme extraction — `src/audio/viseme_map.py` (8 groups); `VisemeEvent` posted from `mouth.py`
- [x] Orchestrator two-pass tool-call loop + `utterance_id` threading
- [x] Godot: `text_bubble.gd`/`.tscn` for streaming display; per-viseme-group mouth animations
- [x] 363 tests passing, 4 skipped; `vision.py` at 86% coverage
- [ ] Real avatar artwork (placeholder colored-circle sprites still in use)
- [ ] LightRAG Option A (deferred to Phase 7)
- [ ] v1.0 release

### Phase 7: LightRAG Personal Knowledge Base — NOT STARTED
*Goal: Users can query personal documents via natural language. Latency benchmark gate.*

- [ ] End-to-end latency benchmark (gate: < 2s round-trip required before LightRAG)
- [ ] `src/llm/rag_retriever.py` — LightRAG integration (explicit skill trigger, UI toggle, off by default)
- [ ] `all-MiniLM-L6-v2` CPU embedding latency benchmark on target hardware
- [ ] LightRAG Option B/C (automatic routing via classifier, gated on >90% precision proof)
- [ ] v1.0 release
