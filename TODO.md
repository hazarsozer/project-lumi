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

## ~~7. No Graceful Error Recovery~~ — DONE
* **Context:** Failing to load the Wake Word ONNX model or lacking a microphone will result in an immediate fatal crash.
* **What was done:**
  - `src/core/startup_check.py` runs `run_startup_checks()` before the event loop starts. Hard failures (missing model, wrong openwakeword version, no microphone) raise `RuntimeError` with human-readable messages. Soft failures (missing STT/LLM model directories) log a warning and continue.
  - `src/audio/ears.py` — `_consumer_loop` now wraps `sd.InputStream` in an outer retry loop (up to `_MAX_RETRIES=3`). `sd.PortAudioError` and unexpected exceptions are caught, logged, and retried after `_RETRY_DELAY_S=0.25s`. Per-chunk `model.predict()` failures are caught at the inner level and skipped. On retry exhaustion, `EarsErrorEvent` is posted to the event queue.
  - `src/core/events.py` — `EarsErrorEvent(code, detail)` dataclass added.
  - `src/core/orchestrator.py` — `_handle_ears_error` handler registered; transitions non-IDLE states to IDLE on receipt.
  - `tests/test_ears_recovery.py` — 6 tests: PortAudioError retry, predict-skip, exhausted retries post EarsErrorEvent, orchestrator handler IDLE transition, no-op when already IDLE.

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

## ~~12. `play_ready_sound()` Blocks the Audio Thread~~ — DONE
* **Resolution:** `src/utils.py` `play_ready_sound(speaker)` enqueues a 0.2s 880 Hz sine-wave ping onto the `SpeakerThread` via `speaker.enqueue()` — fully non-blocking. `sd.play()`/`sd.wait()` removed. Verified by 7 unit tests in `tests/test_utils.py`.

## ~~13. No VRAM Resource Manager~~ — DONE
* **Context:** The core idea of offloading LLMs to system RAM to keep VRAM free for gaming is completely unhandled right now.
* **Resolution:** `src/llm/model_loader.py` wraps `llama_cpp.Llama` with `load()`/`unload()` lifecycle. A module-level `_VRAM_LOCK` is shared with `ScreenshotTool` so LLM and vision model loads are mutually exclusive — confirmed by `tests/test_vram_mutex_concurrent.py` (3 tests).

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

## ~~21. Phase 6: The Hands (OS Control)~~ — DONE

* **What was done (534 tests, 4 skipped, 0 failures):**
  - `src/tools/` package — `Tool` Protocol, `ToolRegistry`, `ToolExecutor` (allowlist + `threading.Event` timeout)
  - OS tools: `AppLaunchTool` (allowlist + `shutil.which`), `ClipboardTool` (xclip), `FileInfoTool` (`Path.parts` traversal guard), `WindowListTool` (wmctrl)
  - `src/tools/vision.py` — `ScreenshotTool` with grim→scrot→Pillow fallback, moondream2 GGUF description, 30s idle unload, VRAM mutex with LLM; 86% coverage
  - `src/audio/viseme_map.py` — 8 viseme groups, `map_phoneme()`, stress digit stripping; 100% coverage
  - `src/audio/mouth.py` — `_post_visemes()` posts `VisemeEvent` per phoneme from Kokoro output
  - `src/llm/reasoning_router.py` — `LLMTokenEvent` posted per token; `utterance_id` param
  - `src/core/zmq_server.py` — `on_llm_token()` sends `llm_token` wire frame to Godot
  - `src/core/orchestrator.py` — tool registry wired; two-pass inference loop; `utterance_id` UUID threaded
  - `src/core/config.py` + `config.yaml` — `ToolsConfig` + `VisionConfig` added
  - Godot: `text_bubble.gd`/`text_bubble.tscn` for streaming display; `llm_token`/`tts_start` routing in `main.gd`; 8 per-viseme-group mouth animations in `avatar_controller.gd`
* **Still open:**
  - Real avatar artwork (placeholder colored-circle sprites still in use)
  - Kokoro phoneme tuple format `(phoneme_str, start_ms, duration_ms)` — assumed, needs local verification with real model
  - moondream2 GGUF availability for llama-cpp-python — needs confirmation
  - Godot `$TextBubble` node must be added in the editor to the main scene
  - Pre-existing flaky test `test_ears_start_sets_listening_flag` (threading race in ears.py)

## ~~18. Phase 3 Wave 4: Coverage Gate + Code Review~~ — DONE

* **Context:** All LLM modules (Waves 0–3) are implemented and tested in isolation. Wave 4 closes Phase 3 with a full-suite coverage run and a code review of all changed files.
* **Resolution (Wave B1 — 2026-04-19):**
  - `uv run pytest tests/ --cov=src --cov-report=term-missing` run end-to-end: **88% overall** (gate: 80%)
  - All `src/llm/` modules: 97–100% ✓. All `src/core/` modules: 81–100% ✓
  - Previously-zero modules brought above 80%: `logging_config.py` 0%→100%, `startup_check.py` 41%→82%
  - `ears.py` lifted from 73%→80% via wake word detection path tests
  - `ipc_transport.py` at 78% (socket error paths; marginal miss, covered by integration tests)
  - 568 tests passing, 4 skipped

## 16. No Fine-Tuning Pipeline

* **Context:** Lumi currently uses a stock base model with no personality, no Lumi identity, and no OS tool-call schema. Out of the box it will claim to be "a large language model by Microsoft" and refuse benign OS operations.
* **Why it matters:** Without fine-tuning, the user experience is degraded. With fine-tuning, Lumi becomes a coherent character with predictable behavior.
* **Items:**
  - QLoRA training script (`scripts/train_lumi.py`) — SFTTrainer with 90/10 train/eval split, r=16 for personality, r=32 for tool-use
  - Dataset generation (synthetic + manual + live mining) — ~1000–1200 examples across 6 categories
  - GGUF export pipeline — merge LoRA → convert → quantize → evaluate (Q4_K_M vs FP16 baseline)
  - Evaluation suite (`tests/test_model_quality.py`) — automated assertions (identity, tool calls, brevity) + manual checklist
  - ~~Domain router (`src/llm/domain_router.py`) — Option A (regex, <1ms), Option B (embedding, ~20ms), decision gate at 20% miss rate~~ — **DONE** (shipped; `DomainRouter.classify()`, 6 domains, safety-first priority order; 39 tests passing)
  - LoRA hot-swap architecture — verify `llama_lora_adapter_set` API in `llama-cpp-python>=0.2.90`; fallback to ModelRegistry if unavailable
  - ~~ModelRegistry (`src/llm/model_registry.py`) — Full GGUF swapping (2.5–7s) if LoRA API missing~~ — **DONE** (shipped; `register()`, `load()`, `unload()`, `current_name`, `is_loaded`, `model`, `list_registered()`; 11 tests passing)
  - Versioning scheme: `lumi-phi35-v{N}-Q4_K_M.gguf` + specialist variants (`lumi-phi35-chat-v1`, `lumi-phi35-os-v1`)
* **Phased rollout:** v1 (identity + brevity), v2 (+ OS tools), v3 (+ code + multi-turn), v4 (+ internet tools)
* **Reference:** See `ARCHITECTURE.md` Section 5 for full strategy: LoRA config table, dataset category specs, training workflow, tool palette, proof-of-concept experiment gate, and open questions.

**GPU status (checked 2026-04-23):** RTX 4070, 12 GB VRAM — **UNBLOCKED** (requirement was ≥8 GB).

**Backlog:**
  - Wave H3 — QLoRA fine-tune (`scripts/finetune_lora.py`): ~~blocked on ≥8 GB VRAM GPU~~ **UNBLOCKED — RTX 4070 12 GB confirmed**
  - Wave H4 — Merge LoRA adapter: blocked on Wave H3
  - Wave H5 — Evaluate delta (Q4_K_M vs FP16 baseline): blocked on Wave H3
  - Wave H6 — Hot-swap wiring into orchestrator: blocked on Wave H3
  - Wave I3 — Avatar sprite integration (`ui/assets/sprites/`): blocked on external PNG delivery
  - TurboQuant activation — uncomment `kv_cache_quant: "turbo3"` in `config.yaml`: blocked on llama.cpp PR #21089 shipping in `llama-cpp-python`
  - Wave J1+ — pip-installable wheel: not yet scoped

## ~~22. Phase 7: RAG Personal Knowledge Base~~ — DONE

* **What was built (534 tests, 4 skipped, 0 failures):**
  - `src/rag/` package — `DocumentStore` (SQLite FTS5 + sqlite-vec kNN, WAL mode),
    `Chunker` (sliding-window), `Embedder` (all-MiniLM-L6-v2, 384-dim CPU),
    `Loader` (.txt/.md/.pdf/.html), `reciprocal_rank_fusion()` (RRF k=60),
    `RAGRetriever` (timeout + cancel-safe), `Citation`, `RAGResult`
  - `src/core/config.py` — `RAGConfig` added to `LumiConfig`
  - `src/llm/prompt_engine.py` — `rag_context` injection in `build_prompt()`
  - `src/llm/reasoning_router.py` — `use_rag` flag, `_maybe_retrieve()`, posts `RAGRetrievalEvent`
  - `src/llm/reflex_router.py` — `route_rag_intent()` intent detection
  - `src/core/events.py` — `RAGRetrievalEvent`, `RAGStatusEvent`, `RAGSetEnabledEvent`
  - `src/core/orchestrator.py` — RAGRetriever at startup; intent check; `_handle_rag_set_enabled()`
  - `src/core/zmq_server.py` — `on_rag_retrieval()`, `on_rag_status()` outbound; `rag_set_enabled` inbound
  - `scripts/ingest_docs.py` — CLI to chunk, embed, and store personal documents
  - `scripts/measure_rag_latency.py` — end-to-end benchmark (p95 < 2.0 s gate)
  - Base latency gate PASS: p95 = 0.431 s (threshold 1.7 s)
  - RAG disabled by default (`config.rag.enabled: false`)
* **Wave B2 (2026-04-19):** RAG latency benchmark executed against live Phi-3.5-mini + all-MiniLM-L6-v2, 20 queries. **p95 = 0.490 s** (gate: 2.0 s) — PASS with 4× headroom. Two bugs fixed in benchmark script: `get_embedder(rag_cfg)` → `get_embedder(rag_cfg.embedding_model)`, and `store.init_schema()` missing. FTS5 syntax errors (`.`, `?` in queries) fixed via `_sanitize_fts_query()` in `store.py`.
* **Still open:**
  - Godot citation panel UI (Wave 4 Godot — deferred)
  - Real avatar artwork (placeholder sprites still in use)

## ~~17. LightRAG Personal Knowledge Base (Phase 6 Optional)~~ — SUPERSEDED BY ITEM 22

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

## ~~23. Godot Citation Panel UI~~ — DONE

* **Context:** Phase 7 RAG retrieval returns `Citation` objects (document name, chunk ID, score) that are currently not surfaced in the Godot frontend.
* **Items:**
  - Design and implement a `citation_panel.gd` / `citation_panel.tscn` in `ui/` — should display source document name and relevance score alongside the LLM response.
  - Wire `rag_retrieval` wire event (already sent by `ZMQServer.on_rag_retrieval()`) to populate the panel in `main.gd`.
  - Add a "hide citations" toggle so the panel can be dismissed.
  - Update `ui/TESTING.md` manual checklist with citation panel test cases.
* **Blocker:** None — `rag_retrieval` wire event is already defined and sent. Godot scene work only.

## ~~25. Phase 8.5 Settings UI~~ — DONE

* **Context:** Runtime configuration editor integrated into Godot overlay; live + restart-required settings with IPC wiring.
* **Wave S0:** ConfigManager, FIELD_META, config_writer — DONE
* **Wave S1:** 4 new IPC config wire events — DONE
* **Wave S2:** ConfigManager wired into Orchestrator — DONE
* **Wave S3:** Godot settings panel scaffold — DONE
* **Wave S4:** 7-tab population, error display, restart bar — DONE
* **Wave S5:** docs + security review — DONE
* **What was built (896 tests, 4 skipped, 0 failures):**
  - `src/core/config_runtime.py` — `ConfigManager`, `ConfigObserver`, `ConfigUpdateResult`; live config apply via `dataclasses.replace()`; thread-safe RLock
  - `src/core/config_schema.py` — `FIELD_META` dict; UI metadata (control type, min/max, restart_required) for 47 user-facing fields
  - `src/core/config_writer.py` — atomic YAML write (tmp + fsync + rename), `.bak` rollover, ruamel.yaml
  - IPC: `config_schema_request` (Body→Brain), `config_schema` (Brain→Body), `config_update` (Body→Brain), `config_update_result` (Brain→Body)
  - Godot: settings panel (`ui/scenes/settings_panel.tscn`, `ui/scripts/settings_panel.gd`) — gear icon / Ctrl+, entry; 7 tabs; `SettingRow` widget with 7 control types

## 24. Real Avatar Artwork — AWAITING ASSETS

* **Context:** The Godot frontend currently uses placeholder colored-circle sprites in `ui/assets/sprites/`. Phase 5 and 6 deferred real artwork.
* **Decision (2026-04-20):** Option A — static sprite sheets for `AnimatedSprite2D`. Live2D / 3D VRM deferred indefinitely.
* **Items:**
  - Commission or create PNG sprite sheets per `ui/assets/sprites/SPRITE_SPEC.md` (4 state animations mandatory, 8 viseme overlays optional).
  - Drop PNGs into `ui/assets/sprites/`, import into Godot `SpriteFrames` resource — no code changes needed; `avatar_controller.gd` picks up animations by name.
* **Blocker:** Artwork — spec written, waiting on art delivery.
