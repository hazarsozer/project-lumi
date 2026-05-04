# Project Lumi: Architecture (v2.0)

**Project Goal:** A local, privacy-first Desktop Assistant — "Siri on Steroids"
**Core Philosophy:** Architect First. Zero Cost. Local Only. Privacy by Default.

> This document is the canonical design reference. README.md links here for deep dives.
> Last updated to reflect actual state as of Ring 2 complete (2026-05-04). Active work: Ring 3.

---

## 1. System Architecture: "The Split-Brain"

Lumi is decoupled into two independent processes that communicate via WebSocket with 4-byte length-prefix framing. This ensures the desktop (especially games and renderers) remains fully responsive regardless of what Lumi's brain is doing.

```
┌──────────────────────────────┐     WebSocket (length-prefix)   ┌──────────────────────┐
│         THE BRAIN            │ ◄──────────────────────────────► │       THE BODY       │
│      (Python Backend)        │  JSON: {event, payload,         │  (Tauri/React UI)    │
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
- **Tech:** Tauri 2 + React 18
- **Target:** < 200MB RAM, negligible GPU at all times

### The Nerves (IPC)
- **Protocol:** WebSocket, 4-byte big-endian uint32 length prefix + UTF-8 JSON body
- **Transport class:** `WSTransport` (`src/core/ws_transport.py`) — asyncio WebSocket server; the Brain runs the WS server directly (no separate bridge process required)
- **Event bridge:** `EventBridge` (`src/core/event_bridge.py`) — sits on top of `WSTransport`, translates outbound internal events to JSON wire frames, translates inbound JSON frames to internal events posted to the orchestrator queue.
- **Enabled by:** `config.ipc.enabled: true` in `config.yaml` (default `true`; set `false` for headless / CI runs)
- **Default endpoint:** `ws://127.0.0.1:5556`
- **Format:** `{ "event": string, "payload": object, "timestamp": float, "version": string }`
- **Handshake:** Brain sends `hello` on connect; client responds with `hello_ack`. Version negotiation in `src/core/handshake.py`.
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
| `config_schema_request` | Body → Brain | `{}` |
| `config_schema` | Brain → Body | `{ "fields": [...], "current": {...} }` |
| `config_update` | Body → Brain | `{ "changes": { "dotted.key": value, ... }, "persist": bool }` |
| `config_update_result` | Brain → Body | `{ "success": bool, "errors": [...], "restart_required": bool }` |
| `system_status` | Brain → Body | `{ "tts_available": bool, "rag_available": bool, "mic_available": bool, "llm_available": bool, "setup_required": bool, "missing_items": [] }` |

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
| `VisemeEvent` | TTS engine (mouth.py) | Orchestrator / EventBridge (lip-sync) |
| `SpeechCompletedEvent` | SpeakerThread / KokoroTTS | Orchestrator |
| `LLMTokenEvent` | LLM reasoning router | Orchestrator (streaming tokens) |
| `InterruptEvent` | Any source (Body via WebSocket, new wake word) | Orchestrator |
| `ShutdownEvent` | main.py / signal handler | Orchestrator |
| `UserTextEvent` | EventBridge (Body → Brain via WebSocket) | Orchestrator |
| `EarsErrorEvent` | Ears thread (after all retries exhausted) | Orchestrator |
| `ToolResultEvent` | Async tool callback (e.g. rag_ingest) | Orchestrator |
| `ConfigSchemaRequestEvent` | EventBridge (Body → Brain via WebSocket) | Orchestrator |
| `ConfigUpdateEvent` | EventBridge (Body → Brain via WebSocket) | Orchestrator / ConfigManager |
| `SystemStatusEvent` | Orchestrator (on startup and subsystem degradation) | EventBridge → Body |

`ZMQMessage` is also defined in `src/core/events.py` as the wire-format dataclass for IPC communication (`event`, `payload`, `timestamp`, `version`).

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

State transitions are published to the Body via `state_change` IPC events. `EventBridge` registers itself as a `StateMachine` observer so every transition is forwarded automatically when the WebSocket server is enabled.

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

### Frontend — The Body (Phase 9.5)
| Component | Technology |
|---|---|
| Renderer | Tauri 2 + React 18 |
| IPC Client | `IBrainClient` interface + `useLumiState` hook (`app/src/state/useLumiState.ts`) — connects directly to Brain WebSocket on `ws://127.0.0.1:5556` |
| Frame protocol | 4-byte length-prefix encode/decode over WebSocket (no bridge process required) |
| Windows | Three Tauri windows: `OverlayRoot` (transparent avatar overlay), `ChatRoot` (chat panel), `SettingsRoot` (settings panel) |
| Avatar | `LumiAvatar` React component driving animated avatar from Brain state events; placeholder images in `app/src/assets/` |
| Settings Panel | `app/src/components/SettingsPanel.tsx` — gear icon / Ctrl+, entry; 7-tab configuration UI; component-based controls (toggle, slider, select, text, number, path, multiselect); requests schema from Brain via `config_schema_request`, applies changes live or marks restart-required |
| First-run setup | `app/src/components/SetupPanel.tsx` — displayed when `system_status.setup_required` is true; lists missing models and guides initial configuration |
| Push-to-talk | `src/audio/hotkey.py` (`PTTListener`) — global hotkey listener (default Ctrl+Space); optional `pynput` dependency; toggle via `audio.ptt_enabled` in `config.yaml` |

### Infrastructure
| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Package Manager | `uv` |
| IPC | WebSocket with 4-byte length-prefix framing (`WSTransport` + `EventBridge`; `websockets` library). Version negotiation via `src/core/handshake.py` (`hello` → `hello_ack`). **Settings wiring (Phase 8.5):** `config_schema_request` / `config_schema`, `config_update` / `config_update_result` wire events for runtime settings panel. |
| Config | `config.yaml` + `src/core/config.py` (`LumiConfig`, `AudioConfig`, `ScribeConfig`, `LLMConfig`, `TTSConfig`, `IPCConfig`, `load_config()`, `detect_edition()`). **Runtime config (Phase 8.5):** `src/core/config_runtime.py` — `ConfigManager` + `ConfigObserver` + `ConfigUpdateResult`; live apply via `dataclasses.replace()`; thread-safe RLock. `src/core/config_schema.py` — `FIELD_META` dict for 47 user-facing fields. `src/core/config_writer.py` — atomic YAML write (tmp + fsync + rename), `.bak` rollover. |
| Logging | Python `logging` module via `src/core/logging_config.py` (`setup_logging()`) |
| Startup Validation | `src/core/startup_check.py` (`run_startup_checks()`) — all checks return soft `list[str]` missing-item lists; `main.py` gates `Ears` on wake-word absence; includes `_check_llm_package()`, `_check_tts_package()`, `_check_rag_packages()` |
| Testing | `pytest` + `pytest-cov`, 80% coverage gate (`tests/` directory, ~1005 passed, 4 skipped at last run) |
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
| Token streaming | `src/llm/reasoning_router.py` + `src/core/event_bridge.py` | `LLMTokenEvent` per token; `utterance_id` UUID threads through; `llm_token` wire frame to Body |
| Push-to-talk | `src/audio/hotkey.py` (`PTTListener`, optional `pynput` dep) | Global hotkey wake fallback; `audio.ptt_enabled` / `audio.ptt_hotkey` config keys |
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
| KV cache (FP16) | ~0.3 GB |
| KV cache with TurboQuant 3–4 bit (conditional, Wave I1) | ~0.075–0.1 GB |
| LoRA adapters (per active) | ~0.05 GB |
| **Total (baseline)** | ~2.85 GB (comfortable within 3.8 GB budget) |
| **Total with TurboQuant** | ~2.6 GB; savings scale with context length (0.4–0.75 GB saved at 8k–16k ctx) |

**TurboQuant KV quantization (conditional, Wave I1):** Google Research ICLR 2026 technique — rotates KV vectors via Fast Walsh-Hadamard Transform, then scalar-quantizes to 3–4 bits. Stacks on top of GGUF weight quantization with no model conversion. Integrates via `cache_type_k="turbo3"` / `cache_type_v="turbo3"` kwargs on `llama_cpp.Llama()` in `src/llm/model_loader.py`. **Blocked on upstream:** awaits llama.cpp PR #21089 propagating into a `llama-cpp-python` release. Config plumbing (`llm.kv_cache_quant` key) ships in Wave I1; flip-switch activation follows once upstream ships.

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
| Phase 5 | IPC transport + frontend (complete; LightRAG deferred) |
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

Current state as of 2026-05-04 (Ring 2 complete):

```
Lumi/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI pipeline (lint, type check, pytest --cov-fail-under=80)
├── .venv/                      # Managed by uv (not committed)
├── models/                     # ONNX/GGUF model binaries (not committed)
│   ├── hey_lumi.onnx           # Custom wake word model
│   ├── llm/                    # GGUF model files (not committed; download separately)
│   │   └── lumi-phi35-v1-Q4_K_M.gguf  # Persona LoRA v1 (2.4 GB; produced by merge_and_quantize.py)
│   └── lumi-lora-v1/           # Raw LoRA adapter weights (not committed)
├── app/                        # Tauri 2 + React 18 frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── LumiAvatar.tsx      # Animated avatar component (placeholder images)
│   │   │   ├── SettingsPanel.tsx   # 7-tab runtime settings; gear icon / Ctrl+, entry
│   │   │   └── SetupPanel.tsx      # First-run guidance screen (shown when setup_required)
│   │   ├── ipc/
│   │   │   ├── client.ts           # IBrainClient interface + WS implementation
│   │   │   ├── events.ts           # Wire event type definitions
│   │   │   └── mockClient.ts       # In-process mock client for tests
│   │   ├── roots/
│   │   │   ├── OverlayRoot.tsx     # Transparent avatar overlay window
│   │   │   ├── ChatRoot.tsx        # Chat panel window
│   │   │   └── SettingsRoot.tsx    # Settings panel window
│   │   ├── state/
│   │   │   └── useLumiState.ts     # React hook: manages Brain state from WS events
│   │   └── styles/
│   │       └── tokens.ts           # Design tokens (colors, typography, spacing)
│   └── src-tauri/                  # Tauri Rust shell + config
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures: synthetic audio arrays, sounddevice mock,
│   │                           #   faster-whisper mock, openwakeword mock, mock_llama_cpp
│   ├── test_ears.py            # Ears: wake word detection, VAD recording paths
│   ├── test_events.py          # All event types + ZMQMessage construction
│   ├── test_ipc_protocol_conformance.py  # Integration: wire protocol round-trips (@pytest.mark.integration)
│   ├── test_ipc_transport.py   # IPCTransport: bind, send, recv, stop
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
│   ├── test_zmq_server.py      # EventBridge outbound events, inbound parsing, lifecycle
│   ├── test_zmq_server_rag.py  # EventBridge RAG event forwarding
│   ├── test_zmq_server_token.py # EventBridge on_llm_token() wire frame
│   ├── test_rag_config.py      # RAGConfig loading and validation
│   ├── test_rag_store.py       # DocumentStore: upsert, FTS5, kNN, WAL
│   ├── test_rag_chunker.py     # chunk_text: overlap, edge cases
│   ├── test_rag_embedder.py    # Embedder: dim, batch, slow model test
│   ├── test_rag_loader.py      # load(): txt, md, unsupported formats
│   ├── test_rag_ingest_script.py # ingest_docs.py CLI integration
│   ├── test_rag_fusion.py      # reciprocal_rank_fusion
│   ├── test_rag_retriever.py   # RAGRetriever: timeout, cancel, char-budget
│   ├── test_rag_intent.py      # route_rag_intent: patterns, edge cases
│   ├── test_prompt_engine_rag.py  # rag_context injection
│   ├── test_eval_persona.py        # scripts/eval_persona.py harness
│   ├── test_ipc_handshake.py       # Version handshake protocol; 100% coverage
│   ├── test_orchestrator_audio_wiring.py # Wake-while-speaking interrupt
│   ├── test_orchestrator_recovery.py # Orchestrator: memory.save() crash → IDLE recovery
│   ├── test_reasoning_router_rag.py # use_rag flag, _maybe_retrieve
│   ├── test_reasoning_router_streaming.py # Token streaming path
│   ├── test_regression.py          # Behavioral contract regressions (persona + state invariants)
│   ├── test_orchestrator_rag.py    # RAGSetEnabledEvent, use_rag wiring
│   ├── test_orchestrator_tools.py  # Orchestrator tool registry wiring
│   ├── test_tool_rag_ingest.py     # rag_ingest tool
│   ├── test_tool_executor.py       # ToolExecutor: allowlist gate, timeout, cancel
│   ├── test_tool_registry.py       # ToolRegistry: register, get, list_tools
│   ├── test_os_actions.py          # AppLaunchTool, ClipboardTool, FileInfoTool, WindowListTool
│   ├── test_viseme_map.py          # map_phoneme(): 8 viseme groups, stress digit stripping
│   ├── test_mouth_visemes.py       # KokoroTTS _post_visemes() integration
│   ├── test_vision.py              # ScreenshotTool: fallback chain, moondream2 stub
│   ├── test_vram_mutex_concurrent.py # _VRAM_LOCK mutual exclusion under concurrency
│   ├── test_logging_config.py      # setup_logging(): human-readable and JSON modes
│   ├── test_startup_check.py       # run_startup_checks(): hard/soft checks
│   ├── test_model_quality.py       # Automated model quality assertions (identity, tool calls, brevity)
│   ├── test_domain_router.py       # DomainRouter.classify(): all 6 domains + general fallback
│   ├── test_model_registry.py      # ModelRegistry: register, load, unload, properties
│   ├── test_ears_recovery.py       # EarsErrorEvent, retry loop, orchestrator handler
│   ├── test_e2e_smoke.py           # End-to-end smoke tests
│   ├── test_kokoro_phoneme_discovery.py # Kokoro phoneme tuple format discovery
│   ├── test_datetime_tool.py       # DatetimeTool (Ring 2)
│   ├── test_timer_tool.py          # TimerTool (Ring 2)
│   ├── test_web_search_tool.py     # WebSearchTool (Ring 2)
│   ├── test_hotkey.py              # PTTListener unit tests
│   ├── test_inference_timeout.py   # LLM inference timeout path
│   ├── test_setup_wizard.py        # scripts/setup_wizard.py smoke tests
│   ├── test_smoke_live.py          # scripts/smoke_live.py smoke tests
│   ├── test_synth_dataset.py       # scripts/synth_dataset.py unit tests
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_config_runtime.py  # ConfigManager, ConfigObserver, ConfigUpdateResult
│   │   ├── test_config_writer.py   # Atomic YAML write, .bak rollover
│   │   ├── test_event_bridge_config.py # config_schema_request/config_update wire events
│   │   ├── test_event_bridge_outbound_coverage.py # Full outbound coverage sweep
│   │   ├── test_hotreload_observers.py # Hot-reload observer wiring
│   │   ├── test_orchestrator_reconfigure.py # Live config apply via Orchestrator
│   │   └── test_rag_status_wiring.py # RAG status event routing
│   ├── ipc/
│   │   ├── __init__.py
│   │   └── test_ws_bridge.py       # ws_bridge deprecated stub smoke test
│   └── integration/
│       ├── __init__.py
│       ├── test_brain_e2e.py       # Full Brain end-to-end integration smoke test (Ring 2)
│       └── test_ipc_full_turn.py   # Full-turn IPC integration tests over real WebSocket
├── src/
│   ├── __init__.py
│   ├── main.py                 # Thin bootstrap: logging → config → checks → orchestrator
│   ├── utils.py                # Shared utilities (play_ready_sound)
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── ears.py             # Microphone capture, wake word detection (openWakeWord),
│   │   │                       #   VAD recording; posts WakeDetectedEvent + RecordingCompleteEvent
│   │   ├── hotkey.py           # PTTListener: global hotkey daemon (pynput, optional);
│   │   │                       #   toggle via audio.ptt_enabled / audio.ptt_hotkey in config.yaml
│   │   ├── mouth.py            # KokoroTTS: sentence-level streaming, prepare()/synthesize()/cancel()/is_busy;
│   │   │                       #   posts TTSChunkReadyEvent, VisemeEvent
│   │   ├── scribe.py           # faster-whisper STT transcription; posts TranscriptReadyEvent
│   │   └── speaker.py          # SpeakerThread: daemon audio playback with resampling;
│   │                           #   posts SpeechCompletedEvent on final chunk
│   ├── ipc/
│   │   ├── __init__.py
│   │   └── ws_bridge.py        # DEPRECATED stub (kept to avoid stale import errors; will be deleted in Ring 3 I7)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # LumiConfig, AudioConfig, ScribeConfig, LLMConfig, TTSConfig, IPCConfig,
│   │   │                       #   ToolsConfig, VisionConfig, PersonaConfig, RAGConfig,
│   │   │                       #   load_config(), detect_edition()
│   │   ├── config_runtime.py   # ConfigManager, ConfigObserver, ConfigUpdateResult;
│   │   │                       #   live apply via dataclasses.replace(); thread-safe RLock
│   │   ├── config_schema.py    # FIELD_META dict; UI metadata for 47+ user-facing config fields
│   │   ├── config_writer.py    # Atomic YAML write (tmp + fsync + rename), .bak rollover
│   │   ├── event_bridge.py     # EventBridge: event translation bridge over WSTransport;
│   │   │                       #   Brain → Body (state_change, transcript, tts_start, tts_viseme,
│   │   │                       #   tts_stop, llm_token, rag_retrieval, rag_status, system_status,
│   │   │                       #   error, config_schema, config_update_result);
│   │   │                       #   Body → Brain (interrupt, user_text, rag_set_enabled,
│   │   │                       #   config_schema_request, config_update)
│   │   ├── events.py           # Frozen event dataclasses + ZMQMessage wire-format type +
│   │   │                       #   SystemStatusEvent, SystemStatusSource
│   │   ├── handshake.py        # IPC version handshake (hello / hello_ack); 100% coverage
│   │   ├── ipc_transport.py    # IPCTransport: legacy raw TCP server (stdlib socket, no pyzmq);
│   │   │                       #   retained but not used in the main path; WSTransport is canonical
│   │   ├── logging_config.py   # setup_logging() — human-readable or JSON structured output
│   │   ├── orchestrator.py     # Orchestrator: event queue, handler dispatch, interrupt handling,
│   │   │                       #   TranscriptReadyEvent → ReflexRouter / ReasoningRouter wiring,
│   │   │                       #   EventBridge injection, _handle_user_text handler,
│   │   │                       #   _handle_config_schema_request, _handle_config_update,
│   │   │                       #   _handle_system_status
│   │   ├── startup_check.py    # run_startup_checks(): soft returns list[str] for missing items;
│   │   │                       #   _check_llm_package(), _check_tts_package(), _check_rag_packages()
│   │   ├── state_machine.py    # LumiState enum (IDLE/LISTENING/PROCESSING/SPEAKING),
│   │   │                       #   StateMachine, InvalidTransitionError, unregister_observer()
│   │   └── ws_transport.py     # WSTransport: asyncio WebSocket server on ws://127.0.0.1:5556;
│   │                           #   single-client, length-prefix framing, replaces IPCTransport
│   └── llm/
│       ├── __init__.py         # Public exports: ReflexRouter, ReasoningRouter, parse_tool_calls,
│       │                       #   ConversationMemory, ModelLoader, PromptEngine
│       ├── domain_router.py    # DomainRouter: regex domain classifier (<1ms); classify(text) → str;
│       │                       #   6 domains (refusal_no_apology, tool_call, out_of_scope,
│       │                       #   knowledge_limit, concise_factual, plain_prose) + "general" fallback;
│       │                       #   safety-first priority order (refusal checked before tool_call)
│       ├── inference_dispatcher.py # LLMInferenceDispatcher: extracted inference subsystem
│       ├── memory.py           # JSON-persisted conversation history (ConversationMemory)
│       ├── model_loader.py     # VRAM hibernate/wake lifecycle (wraps llama_cpp.Llama); module-level _VRAM_LOCK shared with ScreenshotTool
│       ├── model_registry.py   # ModelRegistry: named GGUF hot-swap registry; register(name, config),
│       │                       #   load(name), unload(), current_name, is_loaded, model,
│       │                       #   list_registered(); composition over ModelLoader; ~2.5–7s swap latency
│       ├── prompt_engine.py    # ChatML prompt assembly + token-budget truncation
│       ├── reasoning_router.py # Token-by-token LLM inference with cancel flag; use_rag flag;
│       │                       #   posts RAGRetrievalEvent after retrieval
│       ├── reflex_router.py    # Regex fast-path: greetings, time queries, RAG intent
│       └── tool_call_parser.py # <tool_call> extractor + JSON recovery (parse_tool_calls)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py             # Tool Protocol + ToolResult frozen dataclass
│   │   ├── datetime_tool.py    # DatetimeTool: current date/time queries (Ring 2)
│   │   ├── executor.py         # ToolExecutor: allowlist gate + threading.Event timeout + cancel flag
│   │   ├── os_actions.py       # AppLaunchTool, ClipboardTool, FileInfoTool, WindowListTool
│   │   ├── rag_ingest.py       # RagIngestTool: trigger document ingestion via tool call
│   │   ├── registry.py         # ToolRegistry: register()/get()/list_tools()
│   │   ├── timer_tool.py       # TimerTool: set and cancel countdown timers (Ring 2)
│   │   ├── vision.py           # ScreenshotTool: grim→scrot→Pillow fallback; moondream2 GGUF description
│   │   └── web_search.py       # WebSearchTool: DuckDuckGo scrape via requests + BeautifulSoup (Ring 2)
│   └── rag/
│       ├── __init__.py         # Public exports: DocumentStore, RAGRetriever, Embedder, chunk_text
│       ├── chunker.py          # chunk_text() — sliding-window text splitting
│       ├── embedder.py         # Embedder wrapping all-MiniLM-L6-v2 (384-dim CPU); get_embedder()
│       ├── errors.py           # RAGUnavailableError, IngestError
│       ├── fusion.py           # reciprocal_rank_fusion() (RRF k=60)
│       ├── loader.py           # load() — .txt/.md/.pdf/.html document reader
│       ├── retriever.py        # RAGRetriever: BM25+kNN hybrid; Citation, RAGResult; cancel-safe
│       ├── schema.sql          # SQLite schema: documents, chunks, chunks_fts, vec_chunks
│       └── store.py            # DocumentStore: FTS5 BM25 + sqlite-vec kNN; WAL; thread-local conn
├── scripts/
│   ├── brain.spec              # PyInstaller spec for standalone Brain sidecar binary (Ring 2)
│   ├── build_brain.sh          # Shell script: PyInstaller build + Tauri externalBin integration (Ring 2)
│   ├── check_config_schema.py  # CLI: print config schema fields and current values
│   ├── doctor.py               # Pre-flight diagnostics: check deps, model files, hardware
│   ├── eval_persona.py         # 20 prompts × 8 criteria persona eval; offline (--dry-run) + live (--live) modes
│   ├── ingest_docs.py          # CLI: chunk + embed + store personal documents into RAG store
│   ├── measure_base_latency.py # Benchmark: LLM-only p95 gate (< 1.7 s; Phase 7 entry gate)
│   ├── measure_rag_latency.py  # Benchmark: retrieval+LLM p95 gate (< 2.0 s)
│   ├── measure_streaming_latency.py # Benchmark: streaming TTS sentence-boundary latency (Ring 2)
│   ├── merge_and_quantize.py   # LoRA merge → GGUF convert → Q4_K_M quantize pipeline (Ring 2);
│   │                           #   requires llama.cpp binaries (pass --llama-cpp-dir or put on PATH)
│   ├── run_lumi.sh             # Shell launcher: sets up env and starts Python Brain
│   ├── setup_wizard.py         # Guided first-run configuration wizard
│   ├── smoke_live.py           # Manual smoke test: real microphone + live model (requires hardware)
│   ├── smoke_test_voice.py     # Voice pipeline smoke test
│   ├── synth_dataset.py        # Synthetic training data generation for fine-tuning
│   └── train_lumi.py           # QLoRA fine-tune entrypoint (SFTTrainer; requires ≥8 GB VRAM)
├── config.yaml                 # Runtime configuration (all keys optional, defaults in config.py)
├── ARCHITECTURE.md             # This file
├── README.md
├── SUGGESTIONS.md              # Known issues and improvement plans
├── TODO.md
└── pyproject.toml              # runtime / llm / training / tts / qlora / rag / ptt / dev optional dep groups
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

### Phase 3: The Brain (Intelligence) — COMPLETE
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

**Wave 4 — COMPLETE:**
- [x] Coverage gate ≥80% on all `src/llm/` and `src/core/` modules (88% overall; see TODO item 18)
- [x] Full code review (Wave B1, 2026-04-19)

### Phase 4: The Mouth (TTS) — COMPLETE
*Goal: High-quality voice response without GPU.*

- [x] TTS engine (Kokoro-82M ONNX) — `src/audio/mouth.py`, KokoroTTS with sentence streaming
- [x] Non-blocking audio playback (SpeakerThread, `src/audio/speaker.py`)
- [x] TTS config keys (`TTSConfig` in `config.py`, `tts:` section in `config.yaml`, startup check)
- [x] Viseme extraction for avatar lip-sync (`viseme_map.py` + `mouth.py`; `VisemeEvent` fully wired — Phase 6)

### Phase 5: The Body (Visuals) — COMPLETE
*Goal: Transparent, interactive desktop overlay + IPC transport to Python Brain.*

- [x] `src/core/ipc_transport.py` — raw TCP server, 4-byte big-endian length prefix, single-client, two daemon threads (retained; `WSTransport` is now canonical)
- [x] `src/core/event_bridge.py` (`EventBridge`) — event translation bridge: internal events ↔ JSON wire protocol
- [x] `src/core/state_machine.py` — `unregister_observer()` added
- [x] `src/core/config.py` — `IPCConfig.enabled` field added
- [x] `src/core/orchestrator.py` — EventBridge injection, `_handle_user_text` handler, shutdown cleanup
- [x] `src/main.py` — EventBridge auto-created when `config.ipc.enabled = true`
- [x] `tests/test_ipc_transport.py`, `tests/test_zmq_server.py`, `tests/test_ipc_protocol_conformance.py`
- [ ] LightRAG Option A (explicit skill trigger, UI toggle, off by default — deferred to Phase 6)

### Phase 6: The Hands (OS Control) — COMPLETE
*Goal: Lumi can act on the desktop. Token streaming. Viseme lip-sync. Advanced RAG routing.*

- [x] `src/tools/` package — `Tool` Protocol, `ToolRegistry`, `ToolExecutor` (allowlist + timeout)
- [x] OS tools: `AppLaunchTool`, `ClipboardTool`, `FileInfoTool`, `WindowListTool` (`src/tools/os_actions.py`); cross-platform adapters (macOS bundle dispatch, Windows adapters)
- [x] Vision tool — `ScreenshotTool` with grim→scrot→Pillow fallback + moondream2 GGUF description (`src/tools/vision.py`)
- [x] LLM token streaming — `LLMTokenEvent` per token; `llm_token` wire frame to Body
- [x] Viseme extraction — `src/audio/viseme_map.py` (8 groups); `VisemeEvent` posted from `mouth.py`
- [x] Orchestrator two-pass tool-call loop + `utterance_id` threading
- [ ] LightRAG Option A (deferred to Phase 7)
- [ ] v1.0 release

### Phase 7: RAG Personal Knowledge Base — COMPLETE
*Goal: Users can query personal documents via natural language. Hybrid BM25 + vector retrieval.*

- [x] `src/rag/schema.sql` — SQLite schema (documents, chunks, FTS5, sqlite-vec vec0)
- [x] `src/rag/errors.py` — `RAGUnavailableError`, `IngestError`
- [x] `src/rag/store.py` — `DocumentStore` (FTS5 BM25 + sqlite-vec kNN; WAL mode; thread-local connections)
- [x] `src/rag/chunker.py` — `chunk_text()` with sliding-window overlap
- [x] `src/rag/embedder.py` — `Embedder` wrapping `all-MiniLM-L6-v2` (384-dim, CPU-only); `get_embedder()`
- [x] `src/rag/loader.py` — `load()` for .txt/.md/.pdf/.html files
- [x] `src/rag/fusion.py` — `reciprocal_rank_fusion()` (RRF k=60)
- [x] `src/rag/retriever.py` — `RAGRetriever`, `Citation`, `RAGResult`; timeout + cancel-safe
- [x] `src/llm/prompt_engine.py` — `build_prompt()` gained `rag_context` injection
- [x] `src/llm/reasoning_router.py` — `use_rag` flag, `_maybe_retrieve()`, posts `RAGRetrievalEvent`
- [x] `src/llm/reflex_router.py` — `route_rag_intent()` for intent detection
- [x] `src/core/events.py` — `RAGRetrievalEvent`, `RAGStatusEvent`, `RAGSetEnabledEvent`
- [x] `src/core/orchestrator.py` — RAGRetriever at startup; intent check; `_handle_rag_set_enabled()`
- [x] `src/core/event_bridge.py` (`EventBridge`) — `on_rag_retrieval()`, `on_rag_status()` outbound; `rag_set_enabled` inbound
- [x] `src/core/config.py` — `RAGConfig` added to `LumiConfig`
- [x] `scripts/ingest_docs.py` — CLI to chunk, embed, and store documents
- [x] `scripts/measure_rag_latency.py` — end-to-end latency benchmark (gate: p95 < 2.0 s)
- [x] Base latency gate: p95 = 0.431 s (threshold 1.7 s) — PASS
- [x] 534 tests passing, 4 skipped
- [x] RAG disabled by default (`config.rag.enabled: false`)
- [ ] Real avatar artwork (placeholder colored-circle sprites still in use)

### Phase 8.5: Settings UI (Runtime Config) — COMPLETE
*Goal: Users can configure Lumi at runtime without restarting, via the Tauri/React UI.*

- [x] `src/core/config_runtime.py` — `ConfigManager`, `ConfigObserver`, `ConfigUpdateResult`; live apply via `dataclasses.replace()`; thread-safe RLock
- [x] `src/core/config_schema.py` — `FIELD_META` dict; 47 user-facing fields with control type, min/max, restart_required metadata
- [x] `src/core/config_writer.py` — atomic YAML write (tmp + fsync + rename), `.bak` rollover
- [x] `src/core/event_bridge.py` — `config_schema_request` / `config_schema` / `config_update` / `config_update_result` wire events wired
- [x] `src/core/events.py` — `ConfigSchemaRequestEvent`, `ConfigUpdateEvent` added
- [x] `src/core/orchestrator.py` — `_handle_config_schema_request()`, `_handle_config_update()` handlers
- [x] `app/src/components/SettingsPanel.tsx` — gear icon / Ctrl+, entry; 7 tabs; 7 control types
- [x] `scripts/setup_wizard.py` — guided first-run configuration wizard
- [x] `tests/core/` — `test_config_runtime.py`, `test_config_writer.py`, `test_event_bridge_config.py`, `test_orchestrator_reconfigure.py`

### Phase 9.5: Tauri UI Overlay + Ring 1 — COMPLETE
*Goal: Tauri 2 + React 18 frontend replaces Godot. Direct WebSocket IPC. First-run UX. Cross-platform tools.*

- [x] `src/core/ws_transport.py` (`WSTransport`) — asyncio WebSocket server replaces `IPCTransport`; Brain exposes WS directly on `ws://127.0.0.1:5556`; no bridge subprocess required
- [x] `app/` — Tauri 2 + React 18 frontend: `OverlayRoot`, `ChatRoot`, `SettingsRoot` windows; `useLumiState` hook; `IBrainClient` WebSocket client
- [x] `app/src/components/SetupPanel.tsx` — first-run guidance screen driven by `system_status.setup_required`
- [x] `src/core/events.py` — `SystemStatusEvent`, `SystemStatusSource` added
- [x] `src/core/event_bridge.py` — `on_system_status()` outbound; `system_status` wire frame to Body
- [x] `src/core/startup_check.py` — all checks converted to soft returns (`list[str]`); `main.py` gates Ears on missing wake-word items
- [x] `src/audio/hotkey.py` (`PTTListener`) — global push-to-talk hotkey daemon; optional `pynput` dependency
- [x] `src/tools/os_actions.py` — macOS bundle dispatch (`_launch_macos_bundle()`), `pyperclip`/`pygetwindow` Windows adapters
- [x] `config.yaml` — `ipc.enabled: true` default; `audio.ptt_enabled`, `audio.ptt_hotkey` keys added
- [x] `src/core/config_schema.py` — `audio.wake_word_enabled`, `audio.ptt_enabled`, `audio.ptt_hotkey` added to `FIELD_META`
- [x] ~1005 tests passing, 4 skipped; >80% coverage

### Ring 2 — COMPLETE (2026-05-04)
*Goal: Shipable binary, persona identity, streaming TTS, web/datetime/timer tools, E2E test.*

- [x] Brain sidecar bundling — `scripts/brain.spec` (PyInstaller spec) + `scripts/build_brain.sh`; Tauri `externalBin` integration
- [x] Persona LoRA v1 — QLoRA training pipeline fully debugged (`scripts/train_lumi.py` → `scripts/merge_and_quantize.py` → `scripts/eval_persona.py`); merged GGUF at `models/llm/lumi-phi35-v1-Q4_K_M.gguf` (2.4 GB, gitignored); `[qlora]` extra added to `pyproject.toml`. Known quality regressions (identity, refusal, filler-opener) deferred to persona v2.
- [x] Streaming TTS on sentence boundaries; `scripts/measure_streaming_latency.py` benchmark
- [x] Web search tool — `src/tools/web_search.py` (`WebSearchTool`, DuckDuckGo scrape via `requests` + `beautifulsoup4`)
- [x] Datetime tool — `src/tools/datetime_tool.py`
- [x] Timer tool — `src/tools/timer_tool.py`
- [x] End-to-end integration smoke test — `tests/integration/test_brain_e2e.py`

### Ring 3 — IN PROGRESS
*Goal: Privacy-first credibility, memory persistence, repo hygiene.*

- [ ] Privacy and threat-model docs (I2/I3) — backing the "privacy-first" claim
- [ ] Conversation memory rotation + LLM summarisation (I5)
- [ ] Avatar artwork or animated SVG fallback (C2)
- [ ] Delete `src/ipc/ws_bridge.py` deprecated stub; verify `ui/` (Godot legacy) is fully removed (I6)
- [ ] `.gitignore` cleanup + scrub committed binaries (I7)
- [ ] Orchestrator decomposition (I1)
- [ ] `openwakeword` upstream PR or vendor fork to remove monkey-patch (I4)
- [ ] Persona v2 — final pre-MVP task after Ring 3; resolves known v1 regressions
