# Project Lumi: Development TODOs

## ~~1. `time` Module Shadow in `_mic_callback`~~ — DONE
* **Context:** The `time` argument in `sounddevice`'s `_mic_callback` shadows the global `import time`, meaning any use of `time.monotonic()` in that function will crash.
* **Resolution:** `ears.py` now uses `import time as _time` and all calls use `_time.monotonic()`.

## ~~2. Mixed Runtime and Training Dependencies~~ — DONE
* **Context:** `pyproject.toml` mixes core runtime packages (Whisper, ONNX) with heavy ML training tools (PyTorch, Torchaudio), bloating the user installation by 2-5 GB.
* **Resolution:** `pyproject.toml` restructured with `[project.optional-dependencies]` groups: `training`, `tts`, and `dev`. Runtime `[project.dependencies]` contains only inference-time packages.

## ~~3. No Structured Logging~~ — DONE
* **Context:** Debugging currently relies on raw `print()` statements scattered across the project.
* **Resolution:** `src/core/logging_config.py` created with `setup_logging(level, json_format)`. Supports human-readable and JSON structured output. Called once at startup from `main.py`. Note: `src/audio/scribe.py` and `src/utils.py` still contain `print()` calls — replace with `logging` in a follow-up pass.

## ~~4. Zero Test Coverage~~ — DONE
* **Context:** The system has no unit tests. Validating complex audio streaming timeouts and state transitions requires speaking into a microphone manually.
* **Resolution:** `tests/` directory created with `conftest.py` (synthetic audio fixtures, mocks for sounddevice, faster-whisper, openwakeword) and 6 test modules: `test_ears.py`, `test_scribe.py`, `test_events.py`, `test_state_machine.py`, `test_orchestrator.py`, `test_utils.py`. 83 tests passing. `--cov-fail-under=80` enforced in CI. `ModelLoader` tests are deferred until `model_loader.py` is implemented (Phase 3 remaining).

## ~~5. No Configuration System~~ — DONE
* **Context:** Variables like VAD threshold, chunk sizes, beam size, and recording timeouts are scattered and hardcoded across `main.py`, `ears.py`, and `scribe.py`.
* **Resolution:** `src/core/config.py` created with frozen dataclasses `AudioConfig`, `ScribeConfig`, `LLMConfig`, `IPCConfig`, and `LumiConfig`. `load_config()` merges `config.yaml` into typed defaults. `detect_edition()` queries `nvidia-smi` to auto-select `light`/`standard`/`pro`. `config.yaml` exists at project root with all keys documented and optional.

## 6. Monkey-Patching openwakeword Internals — PARTIALLY DONE
* **Context:** `ears.py` currently relies on an unsafe monkey-patch of `openwakeword.utils.AudioFeatures` to bypass an unsupported kwarg.
* **What was done:** `pyproject.toml` pins `openwakeword==0.4.0` (not 0.6.0 — that version has no Python 3.12 wheels). `startup_check.py` enforces this exact version with a hard `RuntimeError` on mismatch. The monkey-patch in `ears.py` remains.
* **Remaining:** Long-term fix — push a PR upstream to add `inference_framework` kwarg and remove the monkey-patch entirely once merged.

## 7. No Graceful Error Recovery — PARTIALLY DONE
* **Context:** Failing to load the Wake Word ONNX model or lacking a microphone will result in an immediate fatal crash.
* **What was done:** `src/core/startup_check.py` runs `run_startup_checks()` before the event loop starts. Hard failures (missing model, wrong openwakeword version, no microphone) raise `RuntimeError` with human-readable messages. Soft failures (missing STT/LLM model directories) log a warning and continue.
* **Remaining:** Safe try-except fallback to `IDLE` for localized runtime errors (e.g., transient VAD dropouts, audio device disconnection) is not yet implemented.

## ~~8. No IPC Contract~~ — DONE
* **Context:** The planned ZeroMQ integration with the Godot frontend lacks a formal schema.
* **What was done:**
  - `ZMQMessage` frozen dataclass added to `src/core/events.py` with fields `event`, `payload`, `timestamp`, `version`. IPC event table documented in `ARCHITECTURE.md`.
  - `src/core/ipc_transport.py` — raw TCP server (stdlib `socket`, 4-byte big-endian length prefix, single-client, two daemon threads). No pyzmq dependency.
  - `src/core/zmq_server.py` — event translation bridge: translates outbound internal events → JSON wire frames; translates inbound frames → `InterruptEvent` / `UserTextEvent` posted to orchestrator queue.
  - `src/core/state_machine.py` — `unregister_observer()` added.
  - `src/core/config.py` — `IPCConfig.enabled: bool = False` added; set to `true` in `config.yaml` to activate.
  - `src/core/orchestrator.py` — ZMQServer injection, `_handle_user_text` handler wired, shutdown cleanup.
  - `src/main.py` — ZMQServer auto-created inside Orchestrator when `config.ipc.enabled`.
  - `tests/test_ipc_transport.py` (7 tests), `tests/test_zmq_server.py` (16 tests), `tests/test_ipc_protocol_conformance.py` (6 integration tests).

## ~~9. No Explicit State Machine~~ — DONE
* **Context:** Ad-hoc boolean flags tracking whether Lumi is listening or processing are scattered everywhere.
* **Resolution:** `src/core/state_machine.py` created. `LumiState` enum defines `IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`. `StateMachine` class enforces a frozenset of valid transitions, raises `InvalidTransitionError` on illegal transitions, and notifies registered observers after each transition. Wired into `Orchestrator`.

## ~~10. No Orchestrator or Event Bus~~ — DONE
* **Context:** `main.py` currently acts as a 42-line god script manually wiring the `Ears` and `Scribe` components.
* **Resolution:** `src/core/orchestrator.py` created. `Orchestrator` owns the event `queue.Queue`, the `StateMachine`, and a handler dispatch table. `orchestrator.run()` blocks until `ShutdownEvent`. `main.py` reduced to an 18-line thin bootstrap.

## ~~11. Synchronous Blocking Audio Pipeline~~ — DONE
* **Context:** The audio pipeline currently calls the wake callback synchronously, blocking the main stream for up to 13 seconds. Phase 3 (LLM) and 4 (TTS) will increase this to 38 seconds of "deafness".
* **Resolution:** `ears.py` refactored — `start(event_queue)` replaces the synchronous callback. `WakeDetectedEvent` is posted to the queue. `src/core/events.py` defines all 9 event types. `Orchestrator._handle_interrupt()` sets an LLM cancel flag, drains pending events by type name, and transitions back to `IDLE`.

## 12. `play_ready_sound()` Blocks the Audio Thread — OPEN
* **Context:** Generating the startup sound utilizes a synchronous `sd.wait()` call, blocking the pipeline for 200ms.
* **Why it matters:** It introduces noticeable latency when trying to execute rapid commands.
* **Current state:** `src/utils.py` still uses blocking `sd.play()` + `sd.wait()`. The non-blocking queue refactor was not implemented in Phase 3 foundations.
* **Step-by-step Actions:**
  1. **Create Output Queue:** Create a dedicated audio output thread with its own playback queue.
  2. **Post sound events:** Refactor `play_ready_sound()` to accept an `output_queue: queue.Queue` and post a playback event instead of blocking.
  3. **Scale for Phase 4:** Expand this thread to act as the eventual handler for TTS streaming.

## 13. No VRAM Resource Manager
* **Context:** The core idea of offloading LLMs to system RAM to keep VRAM free for gaming is completely unhandled right now.
* **Why it matters:** Without VRAM management, Lumi will randomly spike VRAM usage, causing lag during heavy GPU tasks and violating the "Zero Cost" premise.
* **Step-by-step Actions:**
  1. **Create `ModelLoader`:** Build `src/llm/model_loader.py` wrapping `llama_cpp.Llama`.
  2. **Implement Wake/Hibernate:** Add `wake()` to load to VRAM during the `PROCESSING` state, and `hibernate()` to garbage-collect and unload upon returning to `IDLE`.
  3. **Add limits:** Incorporate config logic limiting the `n_gpu_layers` based on the autodetected VRAM budget.

## ~~14. Naming Divergence Between Design and Code~~ — DONE
* **Context:** Several components described in `ARCHITECTURE.md` (like `audio/listener.py`) don't map to the physical file tree (like `audio/ears.py`).
* **Resolution:** `ARCHITECTURE.md` directory structure updated to reflect actual file paths. All references to `listener.py` removed from documentation. New modules (`orchestrator.py`, `state_machine.py`, `config.py`, `events.py`, `logging_config.py`, `startup_check.py`) created at the paths documented in the architecture.

## ~~15. No LLM Integration~~ — DONE (Waves 0–3)

* **Context:** Phase 3 LLM pipeline implemented across 4 implementation waves.
* **What was done:**
  - `src/llm/model_loader.py` — VRAM hibernate/wake lifecycle via `llama-cpp-python` (8 tests)
  - `src/llm/prompt_engine.py` — ChatML prompt assembly + token-budget truncation (7 tests)
  - `src/llm/memory.py` — JSON-persisted conversation history (9 tests)
  - `src/llm/reflex_router.py` — Regex fast-path: greetings, time queries (8 tests)
  - `src/llm/reasoning_router.py` — Token-by-token inference with cancel flag (6 tests)
  - `src/llm/tool_call_parser.py` — `<tool_call>` extractor + JSON recovery (10 tests)
  - `src/llm/__init__.py` — Public exports for all 6 modules
  - `pyproject.toml` — `[project.optional-dependencies] llm` group added
  - `src/audio/scribe.py` — `print()` → `logger.info()`, `__main__` block removed
  - `src/core/orchestrator.py` — `_handle_transcript()` wired: reflex fast-path + reasoning daemon thread
* **Remaining (Wave 4):** Coverage gate ≥80% on all `src/llm/` + `src/core/` modules; full code review.

## ~~19. Phase 4 TTS Integration~~ — DONE

* **Wave 1 (speaker.py) — DONE:** `src/audio/speaker.py` SpeakerThread with resampling, daemon pattern, SpeechCompletedEvent on final chunk. `tests/test_speaker.py` created.
* **Wave 2 (mouth.py) — DONE:** `src/audio/mouth.py` KokoroTTS with sentence-level streaming, prepare()/synthesize()/cancel()/is_busy. Pre-cancel race fixed. `tests/test_mouth.py` created. Orchestrator wired (tts= param, _handle_llm_response, interrupt SPEAKING branch).
* **Wave 3 (config + docs) — DONE:** `TTSConfig` added to `config.py` and `LumiConfig`. `tts:` section added to `config.yaml`. `_check_tts_model()` soft check added to `startup_check.py`.
* **Note:** Viseme extraction for lip-sync is deferred to Phase 6 (VisemeEvent is posted, but phoneme data not yet extracted from Kokoro output).

## ~~20. Phase 5 IPC Transport + Godot Frontend~~ — DONE

* **Context:** Transparent Godot 4 overlay connected to the Python Brain via raw TCP.
* **What was done:**
  - `src/core/ipc_transport.py` — `IPCTransport`: raw TCP server, 4-byte big-endian uint32 length prefix, single-client model, two daemon threads (`ipc-accept`, `ipc-recv`), two-lock design (`_send_lock` + `_client_lock`), stdlib `socket` only (no pyzmq).
  - `src/core/zmq_server.py` — `ZMQServer`: event translation bridge; outbound `on_state_change`, `on_tts_start`, `on_tts_viseme`, `on_tts_stop`, `on_transcript`, `on_error`; inbound `interrupt` → `InterruptEvent`, `user_text` → `UserTextEvent`.
  - `src/core/state_machine.py` — `unregister_observer()` added.
  - `src/core/config.py` — `IPCConfig.enabled: bool = False`; set `ipc.enabled: true` in `config.yaml` to activate the IPC server.
  - `src/core/orchestrator.py` — ZMQServer injection, `_handle_user_text` handler, shutdown cleanup.
  - `src/main.py` — ZMQServer auto-created inside Orchestrator when `config.ipc.enabled`.
  - `ui/` — Godot 4 project: `project.godot`, `scenes/main.tscn`, `scenes/avatar.tscn`, `scripts/ipc_protocol.gd`, `scripts/lumi_client.gd`, `scripts/avatar_controller.gd`, `scripts/main.gd`.
  - `tests/test_ipc_transport.py` (7 tests), `tests/test_zmq_server.py` (16 tests), `tests/test_ipc_protocol_conformance.py` (6 integration tests, `@pytest.mark.integration`).
  - Total test count: **284 passing**.
* **Deferred to Phase 6:** Real avatar artwork (placeholder colored-circle sprites used), LightRAG Option A, LLM token streaming to Godot, viseme extraction.

## 21. Phase 6: The Hands (OS Control) — NOT STARTED

* **Goal:** Lumi can act on the desktop, stream tokens to the Godot overlay, and optionally query personal documents.
* **Items:**
  - Vision tool (`src/tools/vision.py`) — screenshot capture + analysis
  - Automation tools (`src/tools/os_actions.py`) — app launch, file management, clipboard
  - Real avatar artwork replacing Phase 5 placeholder sprites in `ui/assets/sprites/`
  - LLM token streaming to Godot frontend (live text rendering as tokens arrive via `LLMTokenEvent`)
  - Viseme extraction from Kokoro phoneme output (`VisemeEvent` fully wired for lip-sync)
  - LightRAG Option A (`src/llm/rag_retriever.py`, explicit skill trigger, UI toggle, off by default)
  - LightRAG Option B/C (automatic embedding classifier, gated on >90% precision proof)
  - v1.0 release

## 18. Phase 3 Wave 4: Coverage Gate + Code Review — OPEN

* **Context:** All LLM modules (Waves 0–3) are implemented and tested in isolation. Wave 4 closes Phase 3 with a full-suite coverage run and a code review of all changed files.
* **Items:**
  - Run `uv run pytest tests/ --cov=src --cov-report=term-missing` and verify ≥80% coverage across `src/llm/` and `src/core/`
  - Identify any coverage gaps and add targeted tests (especially for `orchestrator._handle_transcript` new paths)
  - `code-reviewer` agent review of: `orchestrator.py`, `reflex_router.py`, `tool_call_parser.py`, `reasoning_router.py`, `memory.py`, `model_loader.py`, `prompt_engine.py`
  - Address any CRITICAL or HIGH findings before committing Wave 4
* **Blocker:** None — all Wave 3 deps are complete.

## 16. No Fine-Tuning Pipeline

* **Context:** Lumi currently uses a stock base model with no personality, no Lumi identity, and no OS tool-call schema. Out of the box it will claim to be "a large language model by Microsoft" and refuse benign OS operations.
* **Why it matters:** Without fine-tuning, the user experience is degraded. With fine-tuning, Lumi becomes a coherent character with predictable behavior.
* **Items:**
  - QLoRA training script (`scripts/train_lumi.py`) — SFTTrainer with 90/10 train/eval split, r=16 for personality, r=32 for tool-use
  - Dataset generation (synthetic + manual + live mining) — ~1000–1200 examples across 6 categories
  - GGUF export pipeline — merge LoRA → convert → quantize → evaluate (Q4_K_M vs FP16 baseline)
  - Evaluation suite (`tests/test_model_quality.py`) — automated assertions (identity, tool calls, brevity) + manual checklist
  - Domain router (`src/llm/domain_router.py`) — Option A (regex, <1ms), Option B (embedding, ~20ms), decision gate at 20% miss rate
  - LoRA hot-swap architecture — verify `llama_lora_adapter_set` API in `llama-cpp-python>=0.2.90`; fallback to ModelRegistry if unavailable
  - ModelRegistry (`src/llm/model_registry.py`) — Full GGUF swapping (2.5–7s) if LoRA API missing
  - Versioning scheme: `lumi-phi35-v{N}-Q4_K_M.gguf` + specialist variants (`lumi-phi35-chat-v1`, `lumi-phi35-os-v1`)
* **Phased rollout:** v1 (identity + brevity), v2 (+ OS tools), v3 (+ code + multi-turn), v4 (+ internet tools)
* **Reference:** See `ARCHITECTURE.md` Section 5 for full strategy: LoRA config table, dataset category specs, training workflow, tool palette, proof-of-concept experiment gate, and open questions.

## 17. LightRAG Personal Knowledge Base (Phase 6 Optional)

* **Context:** Optional user-facing feature deferred from Phase 5 to Phase 6. Users can feed Lumi personal documents (notes, manuals, wikis) and query them via natural language. Not a core mechanic — UI toggle, off by default. Orthogonal to LoRA but competes for context window and 150–600ms latency.
* **Prerequisites:** Phase 3 and 4 complete, end-to-end latency benchmarked, `all-MiniLM-L6-v2` CPU latency benchmarked on target hardware. **Critical:** If personality LoRA in use, retrain with 50–100 `[CONTEXT]` block examples before deploying LightRAG.
* **Items:**
  - `src/llm/rag_retriever.py` (new) — Encapsulates LightRAG query, enforces 600-token hard cap, formats results
  - `src/llm/reasoning_router.py` — Add optional `rag_enabled` flag to `route()`
  - `src/llm/prompt_engine.py` — Add optional `retrieved_context` parameter to `format_prompt()`
  - `src/core/orchestrator.py` — RAG trigger check in `_on_transcript_ready` (regex pattern: "search my docs", "look up in notes", etc.)
  - SQLite graph storage — zero-config, single-file, <50ms cold-start (no Neo4j, no in-memory)
  - Embedding model — `all-MiniLM-L6-v2` (~80MB, ~10-30ms CPU inference, 384-dim vectors), load once and keep in RAM while enabled
  - Trigger model — **Option A (Phase 6):** Explicit skill via regex (clear expectation, no hallucination). **Option B/C (Phase 6+):** Automatic via embedding classifier (gated on >90% precision proof).
  - UI toggle (Godot frontend) — off by default, "searching documents…" animation during retrieval masks latency
  - Document commands — explicit "remove document" and "re-index" exposed to user for graph maintenance
* **Token budget (hard cap):**
  - System prompt: ~120
  - Retrieved context: **600 max**
  - History (3–4 turns): ~800
  - Current query: ~50
  - Generation headroom: ~512
  - Safety margin: ~200
  - **Total: ~2,280 of 4,096**
* **Go/No-Go gate:** If end-to-end latency after Phase 4 exceeds 2 seconds, defer LightRAG until base pipeline optimized (adding 150–600ms retrieval would push past 3-second voice UI threshold).
* **Reference:** See `ARCHITECTURE.md` Section 6 for full analysis: mitigations (latency masking, embedding lifecycle, VRAM budget, context window pressure, prompt injection risk, thread safety), integration point (no new events, inside ReasoningRouter), architectural fit (orthogonal to LoRA with retraining caveat).
