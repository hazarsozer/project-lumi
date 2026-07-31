# Definition of Done — Project Lumi

> Codified 2026-05-29. Required by ADR 0010 (CR-41 / #42) following the
> 2026-05-27 Crucible review which found that three live integration regressions
> were hidden behind a unit-green test suite. Eval-against-the-direct-LLM is
> necessary but not sufficient. Every ship claim must pass the end-to-end live
> test below.

---

## 1. Per-Change Criteria (every PR / commit)

Before any change is merged to a shared branch:

- [ ] Tests written (new behaviour = new test; bug fix = regression test).
- [ ] Full suite passes: `uv run pytest` — no `--continue-on-collection-errors` masking, no skipped modules that touch the changed code.
- [ ] Frontend typecheck + lint + tests green: `cd app && npm run typecheck && npm run lint && npm run test`.
- [ ] Code reviewed: a second pass (human or agent) on the diff before merge.
- [ ] No hardcoded secrets, debug print statements, or commented-out dead code introduced.

---

## 2. End-to-End Live-Test Gate (required before any "done" / "shipping" claim)

**Before asserting that a feature, release, or the product itself is shippable, you must run the full voice pipeline and observe it complete one turn.** Eval scripts that call the model directly bypass the orchestrator, the audio pipeline, and the IPC handshake — they are necessary but not sufficient.

### Pass criteria (ALL must be true in a single run)

1. Start the Brain:
   ```bash
   uv run python -m src.main
   ```
   The Brain must start without errors and log that the WebSocket server is listening on `ws://127.0.0.1:5556`.

2. Start or connect the Tauri client:
   ```bash
   cd app && npm run dev
   ```
   The Tauri frontend must connect on cold start **without** a 1008 close code. The `hello` / `hello_ack` IPC handshake must complete within the `HANDSHAKE_TIMEOUT_S` budget.

3. Trigger one full voice turn via wake word ("Hey Lumi") or push-to-talk (Ctrl+Space):
   - The Brain must post `WakeDetectedEvent` and enter LISTENING state.
   - VAD-gated recording must complete and produce a non-empty transcript.
   - The LLM must emit a **non-empty** reply (not `""`, not `tts_start.text = ""`).
   - TTS audio must play to completion; the Brain must return to IDLE.

4. Alternatively, use the `scripts/chat_ws.py` batch client for a headless check:
   ```bash
   uv run python scripts/chat_ws.py --message "Hey Lumi, what is two plus two?"
   ```
   The response text must be non-empty and contain a recognizable answer.

### What this gate catches (lessons from 2026-05-27 regressions)

| Regression | What it broke | Unit tests missed it |
|---|---|---|
| R1 — Audio pipeline orphan | `_consumer_loop` never called `record_command_with_vad`; state machine stuck in LISTENING | Yes — wake event dispatch was unit-tested, but the full wake→record→complete chain was not |
| R2 — Empty LLM replies | `create_completion(max_tokens=1)` streaming loop exited on EOS token; every reply was `""` | Yes — streaming loop logic was not exercised in integration |
| R3 — IPC handshake race | Brain's 3 s timeout shorter than Tauri's cold-start `read_ipc_token`; frontend got 1008 | Yes — handshake unit tests used mock timings; real cold-start latency was never measured |

**The live-test gate is the only check that exercises all three of these failure modes simultaneously.** Run it after every R1/R2/R3-class fix and before every release claim.

---

## 3. v1.0 Ship Gate (all criteria must be green simultaneously)

The project is ready to tag and merge `hardening/crucible-v1.0` → `main` when:

- [ ] **All 42 Crucible findings closed.** Issues `#3–#44` (label `crucible-audit`) are all closed or deliberately deferred with documented rationale. No P0 or P1 issue open.
- [ ] **End-to-end live test passes** (Section 2 above), recorded with a log snippet or screen capture as evidence.
- [ ] **CI green on the branch.** `uv run pytest` passes (all modules, no masking); frontend typecheck + lint + test passes; coverage gate ≥ 80%.
- [ ] **Persona frozen.** No retraining, no dataset regeneration, no cvec re-export. The shipping model is `models/llm/lumi-phi35-v2.1-abliterated-a15-Q5_K_M.gguf` with `identity_guard_enabled: true` in `config.yaml`. Per ADR 0010 (`docs/wiki/decisions/0010-v1-hardening-release-and-persona-freeze.md`), persona improvements are deferred to a post-v1.0 ADR.
- [ ] **Final review done.** A fresh review of the full diff from `main` to the hardening branch, confirming no unresolved CRITICAL or HIGH findings.
- [ ] **No regression from baseline.** Persona headline eval ≥ 77.8%; Category A ≥ 92%; PHI_PRIOR brand strings = 0. (Run `uv run python scripts/eval_identity.py` against the locked GGUF to confirm.)

---

## 4. Current Status (as of 2026-05-29)

v1.0 hardening is **IN PROGRESS** on branch `hardening/crucible-v1.0`.

- Integration regressions R1 (audio orphan), R2 (empty LLM replies), R3 (IPC handshake race) are **fixed**.
- Most P0 and P1 Crucible findings are closed. P2/P3 (performance + structural refactors) may still be in progress.
- The live-test gate has not yet been formally signed off on the hardening branch.
- The branch has **not** been merged to `main`; the product is **not yet shipped**.

See `TODO.md` for the regression fix log and `docs/wiki/decisions/0010-v1-hardening-release-and-persona-freeze.md` for the full scope-reset decision.
