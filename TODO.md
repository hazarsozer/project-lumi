# Project Lumi: Development TODOs

## Ring 1 — Complete (2026-05-02)

The following items were delivered as part of Ring 1 (make it installable):

- **B3 DONE** — `ws_bridge` subprocess dropped; `WSTransport` (`src/core/ws_transport.py`) runs the WebSocket server directly inside the Brain. Architecture is now 2-process (Brain + Tauri app). Default port is `ws://127.0.0.1:5556`.
- **B1 DONE (partial)** — Cross-platform OS tools: macOS bundle dispatch (`_launch_macos_bundle()` in `os_actions.py`); `pyperclip`/`pygetwindow` Windows adapters added to mypy overrides.
- **B5 DONE** — `PTTListener` (`src/audio/hotkey.py`): global push-to-talk hotkey (Ctrl+Space default); `audio.ptt_enabled` / `audio.ptt_hotkey` config keys; `FIELD_META` entries added.
- **C1 DONE** — `SetupPanel.tsx`: first-run guidance screen; `startup_check.py` all-soft-return pattern; `main.py` gates `Ears` on missing wake-word items; `ipc.enabled: true` default.
- **B2 DONE (partial)** — Tauri AppImage + deb bundle targets configured. Full Brain sidecar bundling (PyInstaller + `externalBin`) promoted to Ring 2 item 6.

## Ring 2 — Complete (2026-05-04)

The following items were delivered as part of Ring 2:

- **DONE** — Brain sidecar bundling: `scripts/brain.spec` (PyInstaller spec) + `scripts/build_brain.sh` build script; Tauri `externalBin` integration.
- **DONE** — Persona LoRA v1: QLoRA training pipeline fully debugged (`scripts/train_lumi.py` → `scripts/merge_and_quantize.py` → `scripts/eval_persona.py`). Merged GGUF at `models/llm/lumi-phi35-v1-Q4_K_M.gguf` (2.4 GB, gitignored). `[qlora]` extra added to `pyproject.toml` (`gguf>=0.10.0`, `protobuf>=4.21,<5`, `sentencepiece>=0.2.0`). Note: v1 has known quality regressions in identity consistency, refusal discipline, and filler-opener patterns; these are deferred to persona v2 (final pre-MVP task after Ring 3).
- **DONE** — Streaming TTS on sentence boundaries; `scripts/measure_streaming_latency.py` latency benchmark.
- **DONE** — Web search + datetime/timer tools: `src/tools/web_search.py`, `src/tools/datetime_tool.py`, `src/tools/timer_tool.py`.
- **DONE** — End-to-end integration smoke test: `tests/integration/test_brain_e2e.py`.

## Ring 3 + Ring 3.5 — Complete (2026-05-04)

First pass landed `bb7a4cc`; architect audit found 3 critical + 4 high-severity gaps; Ring 3.5 corrective pass landed `bf0e2fe`. Verified close 2026-05-04. 976 tests passing, zero regressions.

- **DONE I2** — IPC bearer-token handshake **with enforced auth**. Brain generates `secrets.token_hex(32)` → `~/.lumi/ipc_token` (chmod 0600). `HandshakeHandler` drops pre-handshake non-`hello_ack` frames when token required (no bypass); fails closed on timeout (no fail-open). Tauri `read_ipc_token` Rust command + `client.ts` hello_ack with token. `client.ts` skips reconnect on close code 1008. 22 handshake tests including 7 token-auth integration tests.
- **DONE I3** — `PRIVACY.md` (audio buffers, transcripts, memory, telemetry=none, data wipe) + `docs/THREAT_MODEL.md` (T1/T2 actors, M1-M3 mitigations, G1-G3 known gaps).
- **DONE I5** — Conversation memory rotation + LLM summarisation. `src/llm/memory.py` rotates at 40 turns, keeps newest 20 verbatim, summarises oldest into a system message via injected callable. `Orchestrator._make_summariser()` wires the LLM directly (`model.create_chat_completion()`); falls back to truncation if model not yet loaded.
- **DONE I6** — `src/ipc/ws_bridge.py` deprecated stub deleted; `ui/` (Godot legacy) confirmed gone; `ARCHITECTURE.md` Ring 3 checklist closed.
- **DONE I7** — `.gitignore` covers `test_canwrite.txt`, `.codex`, `*.tmp`, `*.tsbuildinfo`, `app/vite.config.{d.ts,js}`, `design_handoff_lumi/`. (Correction: `validation_set_features.npy` was never in git history — no `git filter-repo` needed.)
- **DONE C2 wire-cost** — Viseme events gated on `config.audio.send_visemes` (default `false`); avatar artwork itself deferred to paid commission post-MVP.
- **DONE Schema sync** — `ipc.token_path` added to `FIELD_META`; stale ZeroMQ/Godot help text refreshed to WebSocket/Tauri.
- **DEFERRED I1** — Orchestrator decomposition: post-MVP debt cleanup, no user-visible benefit.
- **DEFERRED I4** — `openwakeword` upstream PR: external work, low MVP value; constraint already documented in `CONTRIBUTING.md`.

## Persona v2 (2026-05-04) — TRAINED, ROLLED BACK, THEN UN-ROLLED-BACK AT Q5_K_M

- **DONE** v2 dataset generated (`scripts/synth_dataset_v2.py` → `data/finetune/synthetic_v2.jsonl`, 5500 examples, conditional tool-aware design).
- **DONE** Runtime tools-in-system-prompt injection wired (`prompt_engine` + `reasoning_router` + `Orchestrator` + `eval_persona`), gated by `llm.tools_in_system_prompt` config flag.
- **DONE** v2 trained, merged, quantized: 1032 steps, 89 min, peak token accuracy 79.8%, final 75.2%.
- **MORNING ROLLBACK (since reverted)** — live eval at default sampling on Q4_K_M revealed mid-token corruption (`I don'concrete`, `Certain Щара`) in ~30% of responses. Diagnosis "BPE-boundary ambiguity" was confidently written up; **diagnosis was wrong**.
- **EVENING UN-ROLLBACK** — direct tokenization showed Phi-3.5's BPE is stable. Phase A diagnostic plan (greedy decode + Q5_K_M / Q8_0 re-quant) localized the cause to **Q4_K_M quantization noise + loose sampling tail**. Q5_K_M restores enough subword-prediction precision that argmax always picks the right contraction suffix. Tightened sampling defaults (T=0.5, top_p=0.9, min_p=0.05, repeat_penalty=1.05) close the tail.
- **CURRENT STATE** — `config.yaml` `model_path` → `models/llm/lumi-phi35-v2-Q5_K_M.gguf` (2.7 GB). **0/23 corruption** at production sampling. 78.3% headline pass rate vs v1's 75%. Both v1 and v2 have Q5_K_M GGUFs on disk; v2-Q5K is shipping.
- New criterion `criterion_no_token_corruption` in `eval_persona.py` is MUST-PASS. New helpers `scripts/check_corruption.py`, `scripts/compare_evals.py`. `merge_and_quantize.py` got `--keep-fp16` / `--skip-merge` flags for cheap re-quantization.
- See [`docs/wiki/postmortems/2026-05-04-persona-v2-bpe-contraction-corruption.md`](docs/wiki/postmortems/2026-05-04-persona-v2-bpe-contraction-corruption.md) for the corrected root-cause analysis (file slug retained for inbound wikilink stability).

## Final Pre-MVP Gate

- [x] **Persona v2.1 (Phi-prior suppression on indirect prompts)** — DONE 2026-05-05. Trained: rank 32, LR 5e-5, 4 epochs, 6102 examples (added `indirect_lumi_voice` + `shipped_tool_emphasis` categories). 0 token corruption, 78.3% headline pass rate. `criterion_lumi_voice_on_indirect_prompts` MUST-PASS added. **However, identity probe revealed the structural ceiling:** capability-denial prompts still 8% Lumi / 67% Phi-prior. v2.1 ships clean as a baseline; the indirect identity fidelity is the open question. See [`docs/wiki/personas/v2.1-findings.md`](docs/wiki/personas/v2.1-findings.md).
- [ ] **Persona v2.2 — Tier 1 (persona vectors + scaled DPO with IPO)** — pivots away from "more LoRA / more DPO" to two parallel tracks per [`docs/wiki/decisions/0008-pivot-to-persona-vectors-and-scaled-dpo.md`](docs/wiki/decisions/0008-pivot-to-persona-vectors-and-scaled-dpo.md):
  - [x] **Track 1.A — Persona-vector inference steering** — COMPLETE 2026-05-06. R1 spike confirmed GGUF has no usable Python hooks → HF fp16 backend. Contrast set (50+50), vectors extracted, sweep all 3 combos PASS (alpha=2: 92.7%/83.3% cap-denial; alpha=8: 95.1%/100%). Wired into runtime: `HFSteeredModel`, `LLMConfig.persona_steering_enabled`, `model_loader.py`. Feature flag off by default.
  - [x] **Track 1.B — Scaled DPO with IPO** — DONE 2026-05-06. Trained 2 epochs, 196 pairs, IPO loss, LR 5e-6. Adapter: `models/lumi-dpo-v2.2/`. Merged + GGUF: `lumi-phi35-v2.2-Q5_K_M.gguf` (2.6 GB). **Eval: cap-denial 8% → 8% (FAILED). Overall 34% → 41% Lumi (below 60% target). Memory/privacy 0% → 40% (unexpected win).** Root cause: LoRA 0.46% of params cannot override Phi-prior encoded across 3.84B params. Results: `results/eval_identity_v2.2.json`.
  - [x] **Eval gates (Tier 1):** FAILED. Cap-denial: 8% (target ≥60%). Overall: 41%/56% L+N (target ≥60%). No Cat-A regression (83% direct-id, ≥target). See `docs/wiki/personas/v2.2-plan.md`.
  - See [`docs/wiki/personas/v2.2-plan.md`](docs/wiki/personas/v2.2-plan.md) for the full plan, risks (R1–R6), and stop conditions.
- [x] **Tier 2 CLOSED — BOTH FAILED (2026-05-06):** DoRA (drop-in, `use_dora=True`, 18M params, IPO loss) — cap-denial 8%/33%L+N, no improvement over v2.1. `results/eval_identity_v2.2_dora.json`. **Weight-delta family (LoRA + DoRA) confirmed dead for this problem.** LoReFT skipped — Track 1.A already achieves 100% cap-denial via representation-space; LoReFT redundant for HF fp16 and cannot help GGUF.
- [x] **Tier 3 — Teacher-distilled SFT — COMPLETE, FAILED (2026-05-07):**
  - Teacher dataset: `data/finetune/synthetic_v3_teacher.jsonl` (312 records, Phi-prior filtered). Script: `scripts/synth_dataset_v3.py`. NOTE: use layers=[12,16,20,24,28] alpha=8, NOT all 32 layers.
  - Combined: `data/finetune/synthetic_v2.3.jsonl` (6414 records = 6102 v2.1 + 312 teacher).
  - SFT: `models/lumi-lora-v2.3/` (35M adapter). 4 epochs, 2h17m, train_loss=2.983, mean_token_accuracy=72.1%.
  - GGUF: `models/llm/lumi-phi35-v2.3-Q5_K_M.gguf` (2626 MiB, 5.77 BPW). Merge complete (56.6s), quant complete (40.5s).
  - **Eval (41-prompt identity probe):** Cap-denial 8% → 8%. Overall 34% → 34% Lumi (zero change). Direct-id 75% → 75%. Memory/privacy 20% → 20%. Full results: `results/eval_identity_v2.3.json`.
  - **Audit (2026-05-07) falsified the "structural ceiling" diagnosis.** Two non-structural causes for the flat 8%: (1) llama.cpp ships native cvec API never applied; (2) eval classifier is target-blind to the cap-denial training corpus. See [[decisions/0009-native-cvec-then-target-aware-teacher]].

### Persona v2.3 — Active plan (per [[personas/v2.3-plan]] + ADR 0009)

- [x] **Track 0 — DONE 2026-05-14.** `eval_identity.py` classifier v2: `LUMI_VOICE` label (warm-refusal verb + Lumi idiom patterns), `is_success()` category-conditional (B/C/D only), `PHI_PRIOR`-first precedence. `scripts/rescore_eval.py` written; all 4 baselines rescored. LUMI_VOICE = 0 across baselines (cap-denial was Phi-prior/neutral, not warm-refusals — correct). 29 classifier unit tests. Suite: 1015 passed / 8 skipped.
- [x] **Track A — DONE 2026-05-14. Partial success — Phi-prior eliminated; cap-denial entered composition mode.**
  - `scripts/export_cvec_to_llama.py` written: `phi_prior_v1.pt` (directions[12:29], alpha=8, sign-negated) → `models/persona_vectors/phi_prior_v1_alpha8_l12-28.cvec.bin` (checksum `44e915b5f5ad1cf4`).
  - `src/llm/model_loader.py` patched: `_apply_gguf_cvec()` calls `llama_cpp.llama_set_adapter_cvec(ctx, data_ptr, buf.size, 3072, 12, 28)` with rc check; logs alpha + layer range + checksum at INFO. `LLMConfig.persona_steering_layer_range` added (default `(12, 28)`).
  - `tests/test_gguf_cvec.py`: 7 CI mock tests + 1 live logit-diff test (slow, skippable). Regression gate in CI.
  - Eval: `results/eval_identity_v2.3_cvec_alpha8.json`. Phi-prior 46%→0% ✅, direct-identity 75%→83% ✅, memory/privacy 0%→40% ✅, cap-denial 8%→0% ❌ (composition mode), edge/meta 67%→17% ❌, overall 34%→32%.
  - Stop condition NOT met (cap-denial < 60%). Root cause: v2.1's cap-denial was Phi-prior-mediated — suppressing the prior reveals base RLHF helpfulness, which composes content. This is a missing-behavior problem, not Phi-prior leak. Sweep would not help.
  - cvec retained: useful as Phi-prior suppressor on top of Track B v2.4 model.
  - See `docs/wiki/personas/v2.3-cvec-findings.md` for full per-category breakdown.
- [ ] **Track B — Claude-distilled target-aware teacher SFT (parallel with A):**
  - `scripts/synth_dataset_v4_claude.py`: Claude API generation. Constraints (validator-enforced, not just prompt wording): first two sentences contain `\bLumi\b`; response matches a warm-refusal pattern AND an alternative-offer pattern; response does NOT match `_PHI_PATTERNS`; length 80–400 chars.
  - Coverage: ~100 unique prompts × 10 paraphrases × 5 responses = ~5,000 records minimum, capped at ~10K. System-prompt variation across the 4 production variants (no-name / Jordan / Taylor / Sam).
  - `data/finetune/synthetic_v4_claude.jsonl` → merge with v2.1 identity (515) + format (688) ONLY (drop v2.1's `conditional_no_tool` 1289) → `data/finetune/synthetic_v2.4.jsonl` (~6–11K records, 70–85% Plan-B).
  - Train SFT from BASE `models/llm/checkpoints/phi-3.5-mini` (NOT from `lumi-merged-v2.1`). Rank 32, LR 5e-5, 4 epochs, save_steps 100. Output: `models/lumi-lora-v2.4/`.
  - Merge → quantize → `models/llm/lumi-phi35-v2.4-Q5_K_M.gguf`.
  - Eval under amended classifier on both 41-prompt identity probe and 23-prompt persona eval. Report Track-A (cvec on v2.4) and Track-B-only (cvec off) separately.
  - **Stop:** ≥ 80% cap-denial (no cvec) → ship v2.4 as default, leave cvec as canary flag. 60–80% (no cvec), ≥ 80% (with cvec) → ship v2.4 + cvec default-on. < 60% even with cvec → B dead, postmortem before D.
- [ ] **Track C — First-token logit-bias / GBNF (reserve, 2–3h):** Negative `logit_bias` against tokenized `["Phi", " Phi", "Microsoft", " Microsoft", " sorry", " language model", " text-based"]` for the first ~6 tokens of each completion (apply, then drop). Or GBNF grammar restricting first ~6 tokens to a Lumi-voice pattern. Triggered if A or B has a long-tail prompt that survives the structural fix.
- [ ] **Track D — Refusal-direction abliteration (LAST RESORT):** Per ADR 0009 §Plan D. Only if A + B both fail to reach 60% cap-denial after tuning. Orthogonalize `o_proj` and `down_proj` against the Phi-prior direction at the targeted layers; quality-recover with one DPO epoch; re-train Lumi LoRA on the abliterated base. Permanent surgery; high variance in side effects on unrelated capabilities.
- [ ] **Live exploit verification** (post Ring 3): run hostile-client script against running Brain to confirm token gate end-to-end; capture proof in `docs/wiki/postmortems/`.
- [ ] **MVP packaging dry-run**: clean fresh-machine install via `.deb` to confirm token-file/sidecar/PTT round-trip works in the wild.

---

## v1.0 Hardening — IN PROGRESS on `hardening/crucible-v1.0` (2026-05-27 → present)

A Crucible review (2026-05-27) graded the project BLOCKED 4.0/10 and found three live integration regressions hiding between unit-green components. The scope-reset decision and persona freeze are recorded in **ADR 0010** (`docs/wiki/decisions/0010-v1-hardening-release-and-persona-freeze.md`). The end-to-end live-test gate and ship criteria are in **`DEFINITION_OF_DONE.md`** (repo root). The 42 Crucible findings are tracked as GitHub issues `#3–#44` (label `crucible-audit`).

Full postmortem at [[docs/wiki/postmortems/2026-05-27-three-integration-regressions-found-on-first-live-test]].

### Integration Regressions (found 2026-05-27 — FIXED on hardening branch)

- [x] **R1 — Audio pipeline orphan** (`src/audio/ears.py`): `_consumer_loop` posted `WakeDetectedEvent` and immediately returned to wake-listening; never called `record_command_with_vad` and never posted `RecordingCompleteEvent`. State machine sat in LISTENING forever after the first wake. **Fixed on `hardening/crucible-v1.0`.**
- [x] **R2 — Reasoning router empty-token streaming** (`src/llm/reasoning_router.py:152-195`): `create_completion(max_tokens=1)` loop's first call decoded to `""` (EOS-class special token), causing `if not token: break` to exit with no collected output. Every WS response carried `tts_start.text = ""`. **Fixed on `hardening/crucible-v1.0`.**
- [x] **R3 — IPC handshake race** (`src/core/handshake.py:50`, `app/src/ipc/client.ts:89-104`): Brain's `HANDSHAKE_TIMEOUT_S = 3.0` was shorter than Tauri's cold-start `invoke("read_ipc_token")` path. Brain disconnected with WS 1008. **Fixed on `hardening/crucible-v1.0`.**

**Status:** R1/R2/R3 fixed; most P0/P1 Crucible findings closed; P2/P3 in progress. Branch NOT yet merged to main — product NOT yet shipped.

**Workaround scripts (remain useful for headless testing):**
- `scripts/chat_ws.py` — Python WS REPL/batch client bypassing Tauri.
- `scripts/probe_persona.py` — direct model call bypassing orchestrator + router.

**Lesson (now codified in `DEFINITION_OF_DONE.md`):** every ship and refactor claim must run `uv run python -m src.main` and exchange at least one message end-to-end. Eval against the direct LLM is necessary but not sufficient.

---

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

## ~~8. No IPC Contract~~ — DONE (superseded by WSTransport in Phase 9.5)
* **Context:** The planned ZeroMQ integration with the Godot frontend lacks a formal schema.
* **What was done (Phase 5):**
  - `ZMQMessage` frozen dataclass added to `src/core/events.py` with fields `event`, `payload`, `timestamp`, `version`. IPC event table documented in `ARCHITECTURE.md`.
  - `src/core/ipc_transport.py` — raw TCP server (stdlib `socket`, 4-byte big-endian length prefix, single-client, two daemon threads). No pyzmq dependency.
  - `src/core/event_bridge.py` (`EventBridge`) — event translation bridge: translates outbound internal events → JSON wire frames; translates inbound frames → `InterruptEvent` / `UserTextEvent` posted to orchestrator queue.
  - `src/core/state_machine.py` — `unregister_observer()` added.
  - `src/core/config.py` — `IPCConfig.enabled: bool = False` added; set to `true` in `config.yaml` to activate.
  - `src/core/orchestrator.py` — EventBridge injection, `_handle_user_text` handler wired, shutdown cleanup.
  - `src/main.py` — EventBridge auto-created when `config.ipc.enabled`.
  - `tests/test_ipc_transport.py` (7 tests), `tests/test_zmq_server.py` (16 tests), `tests/test_ipc_protocol_conformance.py` (6 integration tests).
* **Phase 9.5 update:** `WSTransport` (`src/core/ws_transport.py`) replaced `IPCTransport` as the canonical transport. `src/core/zmq_server.py` was deleted. Default endpoint is now `ws://127.0.0.1:5556`. `ipc.enabled` defaults to `true`.

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

## ~~20. Phase 5 IPC Transport + Godot Frontend~~ — DONE (Godot superseded by Tauri/React in Phase 9.5)

* **Context:** Transparent overlay connected to the Python Brain via IPC.
* **What was done (Phase 5):**
  - `src/core/ipc_transport.py` — `IPCTransport`: raw TCP server, 4-byte big-endian uint32 length prefix, single-client model, two daemon threads (`ipc-accept`, `ipc-recv`), two-lock design (`_send_lock` + `_client_lock`), stdlib `socket` only (no pyzmq).
  - `src/core/event_bridge.py` (`EventBridge`): event translation bridge; outbound `on_state_change`, `on_tts_start`, `on_tts_viseme`, `on_tts_stop`, `on_transcript`, `on_error`; inbound `interrupt` → `InterruptEvent`, `user_text` → `UserTextEvent`.
  - `src/core/state_machine.py` — `unregister_observer()` added.
  - `src/core/config.py` — `IPCConfig.enabled: bool = False`; set `ipc.enabled: true` in `config.yaml` to activate the IPC server.
  - `src/core/orchestrator.py` — EventBridge injection, `_handle_user_text` handler, shutdown cleanup.
  - `src/main.py` — EventBridge auto-created when `config.ipc.enabled`.
  - `tests/test_ipc_transport.py` (7 tests), `tests/test_zmq_server.py` (16 tests), `tests/test_ipc_protocol_conformance.py` (6 integration tests, `@pytest.mark.integration`).
  - Total test count at Phase 5 close: **284 passing**.
* **Phase 9.5 update:** Godot `ui/` directory deleted. Tauri/React frontend (`app/`) is now canonical. `WSTransport` replaced `IPCTransport`. `zmq_server.py` deleted.
* **Deferred to Phase 6:** LightRAG Option A, LLM token streaming, viseme extraction.

## ~~21. Phase 6: The Hands (OS Control)~~ — DONE

* **What was done (534 tests, 4 skipped, 0 failures):**
  - `src/tools/` package — `Tool` Protocol, `ToolRegistry`, `ToolExecutor` (allowlist + `threading.Event` timeout)
  - OS tools: `AppLaunchTool` (allowlist + `shutil.which`), `ClipboardTool` (xclip), `FileInfoTool` (`Path.parts` traversal guard), `WindowListTool` (wmctrl)
  - `src/tools/vision.py` — `ScreenshotTool` with grim→scrot→Pillow fallback, moondream2 GGUF description, 30s idle unload, VRAM mutex with LLM; 86% coverage
  - `src/audio/viseme_map.py` — 8 viseme groups, `map_phoneme()`, stress digit stripping; 100% coverage
  - `src/audio/mouth.py` — `_post_visemes()` posts `VisemeEvent` per phoneme from Kokoro output
  - `src/llm/reasoning_router.py` — `LLMTokenEvent` posted per token; `utterance_id` param
  - `src/core/event_bridge.py` — `on_llm_token()` sends `llm_token` wire frame to Body
  - `src/core/orchestrator.py` — tool registry wired; two-pass inference loop; `utterance_id` UUID threaded
  - `src/core/config.py` + `config.yaml` — `ToolsConfig` + `VisionConfig` added
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

## ~~16. No Fine-Tuning Pipeline~~ — PARTIALLY DONE (Ring 2)

* **Context:** Lumi currently uses a stock base model with no personality, no Lumi identity, and no OS tool-call schema. Out of the box it will claim to be "a large language model by Microsoft" and refuse benign OS operations.
* **Why it matters:** Without fine-tuning, the user experience is degraded. With fine-tuning, Lumi becomes a coherent character with predictable behavior.
* **Items:**
  - ~~QLoRA training script (`scripts/train_lumi.py`) — SFTTrainer with 90/10 train/eval split, r=16 for personality, r=32 for tool-use~~ — **DONE** (Ring 2)
  - ~~Dataset generation (synthetic + manual + live mining) — ~1000–1200 examples across 6 categories~~ — **DONE** (Ring 2; `scripts/synth_dataset.py`)
  - ~~GGUF export pipeline — merge LoRA → convert → quantize → evaluate (Q4_K_M vs FP16 baseline)~~ — **DONE** (Ring 2; `scripts/merge_and_quantize.py`; `[qlora]` extra in `pyproject.toml`). Requires `llama.cpp` built locally (pass `--llama-cpp-dir`).
  - ~~Evaluation suite (`tests/test_model_quality.py`) — automated assertions (identity, tool calls, brevity) + manual checklist~~ — **DONE**
  - ~~Domain router (`src/llm/domain_router.py`) — Option A (regex, <1ms), Option B (embedding, ~20ms), decision gate at 20% miss rate~~ — **DONE** (shipped; `DomainRouter.classify()`, 6 domains, safety-first priority order; 39 tests passing)
  - LoRA hot-swap architecture — verify `llama_lora_adapter_set` API in `llama-cpp-python>=0.2.90`; fallback to ModelRegistry if unavailable
  - ~~ModelRegistry (`src/llm/model_registry.py`) — Full GGUF swapping (2.5–7s) if LoRA API missing~~ — **DONE** (shipped; `register()`, `load()`, `unload()`, `current_name`, `is_loaded`, `model`, `list_registered()`; 11 tests passing)
  - Versioning scheme: `lumi-phi35-v{N}-Q4_K_M.gguf` + specialist variants (`lumi-phi35-chat-v1`, `lumi-phi35-os-v1`)
* **v1 delivered (Ring 2):** `models/llm/lumi-phi35-v1-Q4_K_M.gguf` (2.4 GB, gitignored). Known quality regressions: identity consistency, refusal discipline, filler-opener patterns. These are deferred to **persona v2**, which is the final pre-MVP task after Ring 3.
* **Phased rollout:** ~~v0~~ → **v1 DONE** (identity + brevity, with known regressions) → v2 (+ OS tools + regression fixes) → v3 (+ code + multi-turn) → v4 (+ internet tools)
* **Reference:** See `ARCHITECTURE.md` Section 5 for full strategy: LoRA config table, dataset category specs, training workflow, tool palette, proof-of-concept experiment gate, and open questions.

**GPU status (checked 2026-04-23):** RTX 4070, 12 GB VRAM — **UNBLOCKED** (requirement was ≥8 GB).

**Still open:**
  - Wave H6 — Hot-swap wiring into orchestrator (LoRA adapter live swap)
  - Wave I3 — Avatar sprite integration (`app/src/assets/`): blocked on external PNG delivery; target is React `LumiAvatar.tsx` component
  - TurboQuant activation — uncomment `kv_cache_quant: "turbo3"` in `config.yaml`: blocked on llama.cpp PR #21089 shipping in `llama-cpp-python`
  - Wave J1+ — pip-installable wheel: not yet scoped
  - **Persona v2** — See Ring 3 section above

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
  - `src/core/event_bridge.py` — `on_rag_retrieval()`, `on_rag_status()` outbound; `rag_set_enabled` inbound
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

## ~~23. Citation Panel UI~~ — SUPERSEDED (Godot deleted)

* **Context:** Phase 7 RAG retrieval returns `Citation` objects surfaced via the `rag_retrieval` wire event.
* **Status:** Godot `ui/` directory deleted in Phase 9.5. `rag_retrieval` wire event is still sent by `EventBridge.on_rag_retrieval()`. Citation display in the Tauri/React frontend is a Ring 2+ item (see `MVP_REPORT.md` C2/C5).

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
  - `app/src/components/SettingsPanel.tsx` — gear icon / Ctrl+, entry; 7 tabs; 7 control types

## 24. Real Avatar Artwork — AWAITING ASSETS

* **Context:** The Tauri/React frontend (`app/src/components/LumiAvatar.tsx`) currently uses placeholder static images. Phase 5 and 6 deferred real artwork.
* **Decision (2026-04-20):** Option A — static sprite sheets. Live2D / 3D VRM deferred indefinitely. Note: Godot `ui/` was deleted in Phase 9.5; artwork delivery target is now `app/src/assets/` and the React avatar component.
* **Items:**
  - Commission or create PNG/SVG assets for 4 states (IDLE, LISTENING, PROCESSING, SPEAKING) and optionally 8 viseme overlays.
  - Drop assets into `app/src/assets/`, wire into `LumiAvatar.tsx`.
* **Blocker:** Artwork — waiting on art delivery.
