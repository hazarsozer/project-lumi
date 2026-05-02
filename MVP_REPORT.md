# Project Lumi — Architecture & Product Readiness Report

> **This document is the active work backlog. All development work prioritises items here
> until the MVP checklist is fully complete.**
>
> Generated: 2026-05-01 | Reviewer: Senior Architect (Opus 4.7)

---

## Scores

| Dimension | Score | Notes |
|---|---|---|
| Architecture quality | **7/10** | Clean event bus, typed events, hot-reload config, real test discipline |
| Public MVP readiness | **3/10** | Distribution, cross-platform, persona, onboarding all unsolved |

---

## What Is Good (Do Not Break)

1. **Event bus + state machine.** Frozen-dataclass events, single `queue.Queue`, dispatch table indexed by type, `_OUTBOUND`/`_inbound` dicts in `event_bridge.py`. The spine holds.
2. **Test discipline.** 900 passing, 7 skipped, 80% coverage gate in CI, named regression contract suite. Ring 1 audit confirmed no tests were silenced or deleted to mark items DONE.
3. **Hot-reload config system.** `ConfigManager` + `dataclasses.replace()` + observer protocol + `FIELD_META` schema + atomic YAML writer. Correct and complete.
4. **Resource hygiene.** VRAM mutex, model hibernate/wake, inference cancel flag, watchdog timeout, daemon threads.
5. **Startup checks.** `startup_check.py` with hard vs. soft failure distinction and actionable remediation in every error.

---

## Blockers — Cannot Ship Without (B-tier)

### B1 — Linux-only at the OS-tools layer
- **Evidence:** `src/tools/os_actions.py` uses `xclip`, `wmctrl`, `grim`/`scrot`. `scripts/run_lumi.sh:52` hardcodes `GDK_BACKEND=x11`. `scripts/run_lumi.sh:26` uses `setsid`. `src/core/config.py` queries `nvidia-smi`.
- **Impact:** 90 % of potential users are on macOS or Windows.
- **Fix:** Strategy pattern in `src/tools/os_actions.py`. `sys.platform` adapter selected at startup. macOS: `pbpaste`/`osascript`/`screencapture`. Windows: `pyperclip`/`pygetwindow`/`PIL.ImageGrab` or `mss`. Each adapter < 50 lines.

### B2 — No installer, no distribution, no model bundling
- **Evidence:** User must install `uv`, `cmake`, `build-essential`, `portaudio19-dev`, Rust, webkit2gtk libs, `uv sync --extra llm --extra tts`, manually download 2.4 GB GGUF + Kokoro ONNX + voices.bin, edit `config.yaml`, run a Python TTY wizard, then start 3 processes via `run_lumi.sh`.
- **Impact:** Impossible for a normal user. 30–60 min even for an experienced developer.
- **Fix:** Tauri bundle with first-run model downloader UI. Target one platform first (Linux AppImage). Run `tauri build` end-to-end; force every rough edge to surface.

### B3 — Three-process startup with sleep-based ordering
- **Evidence:** `scripts/run_lumi.sh` boots Brain → `sleep 2` → boots `ws_bridge` → `sleep 1` → boots Tauri. `src/ipc/ws_bridge.py` exists solely because the WebView cannot speak raw TCP.
- **Impact:** 3 PIDs, 2 race conditions, 1 `kill -- -$$` hack. Fragile, unshippable.
- **Fix:** Drop `ws_bridge`. Move WebSocket server into the Brain process (replace `IPCTransport` with `websockets` directly — already a runtime dependency). Result: 2-PID architecture. `run_lumi.sh` becomes optional.
- **Alternate:** Move IPC into Tauri's Rust side via `invoke()`; use Tauri sidecar for the Brain.

### B4 — The LLM has no persona
- **Evidence:** TODO.md item 16: stock Phi-3.5 introduces itself as "a large language model by Microsoft" and refuses benign OS operations. `scripts/train_lumi.py` and `synth_dataset.py` exist; no training has been run.
- **Impact:** Every user's first impression is "I am Phi-3.5". Loses 50 % of users in 30 seconds.
- **Fix:** Train the persona LoRA. 200–300 examples: identity + brevity + 5 OS tool-call formats. One QLoRA run on RTX 4070 (12 GB VRAM, confirmed unblocked). Merge → Q4_K_M GGUF → bundle in installer. Reference: TODO.md #16 Wave plan.

### B5 — Wake word is a personal asset, with no fallback
- **Evidence:** `config.yaml → audio.wake_word_model_path` points to a custom `hey_lumi.onnx` trained for the developer. No push-to-talk fallback exists.
- **Impact:** Shipping this file publicly requires license + reproducibility verification. Custom wake word accuracy varies widely between speakers.
- **Fix:** Add push-to-talk global hotkey (default: Ctrl+Space) as the primary entry point. Wake word becomes opt-in. Document `hey_lumi.onnx` provenance.

---

## Critical Issues — Severely Degrade MVP (C-tier)

### C1 — No first-run UX
- **Evidence:** `scripts/setup_wizard.py` is a Python TTY interactive prompt. No GUI onboarding.
- **Fix:** Tauri first-run screen: detect missing models, show download progress, test microphone, done. The setup wizard becomes a UI flow, not a REPL.

### C2 — Avatar is two static PNGs
- **Evidence:** `app/src/components/LumiAvatar.tsx:21` — swaps between `lumi-idle.png` and `lumi-speaking.png`. All viseme pipeline work is live in the backend but not rendered in the frontend.
- **Fix (option A):** Commission sprite sheets per `ui/assets/sprites/SPRITE_SPEC.md`. Wire into `LumiAvatar`. **(option B):** Pivot to animated SVG / Lottie for MVP, defer sprites. Either way: stop paying the viseme wire-event cost for a feature the UI doesn't render.

### C3 — No web search, no datetime/timer tools
- **Evidence:** Current tool palette: `launch_app`, `clipboard`, `file_info`, `window_list`, `screenshot`. None of those are canonical voice-assistant queries.
- **Fix:** Add two tools: (1) DuckDuckGo HTML scrape, no API key required. (2) datetime + countdown timer. Each < 100 lines.

### C4 — No streaming TTS
- **Evidence:** `src/llm/reasoning_router.py` buffers the full LLM response and only then fires `LLMResponseReadyEvent`, which triggers TTS. Perceived latency: 4–6 s silence.
- **Fix:** Split on terminal punctuation (`. ! ?`) inside `ReasoningRouter`; fire TTS on first sentence completion. Expected perceived latency drop: 5 s → 800 ms.

### C5 — No end-to-end integration smoke test
- **Evidence:** Documented in the 2026-04-30 session: "no test that starts Brain + connects a fake frontend client + exercises a full user_text → LLM response → TTS cycle."
- **Fix:** `tests/integration/test_brain_e2e.py` — start Brain in a thread, open TCP connection, post `user_text`, assert `llm_token` stream, assert `tts_start` + `tts_stop`. Mock the LLM with a stub; mock TTS as a no-op.

---

## Important Architectural Debt (I-tier)

### I1 — Orchestrator is still a god class
- **Evidence:** `src/core/orchestrator.py`, 745 lines. `__init__` constructs ~15 subsystems and registers 14 handlers. Owns LLM, audio, RAG, tools, IPC, and config bootstrapping.
- **Fix (not MVP-blocking but do it in Ring 3):** Repeat the `LLMInferenceDispatcher` extraction pattern for `RagSubsystem`, `ToolsSubsystem`, `AudioSubsystem`. Orchestrator becomes ~250 lines of pure event routing.

### I2 — IPC threat model undocumented and unprotected
- **Evidence:** `src/core/config.py` — `IPCConfig.address: str = "127.0.0.1"`. Any local process can connect and send `user_text` events to trigger OS tools.
- **Fix:** (1) Write a 2-page threat-model doc. (2) Add a token handshake: Brain writes a single-use bearer token to `~/.lumi/ipc_token` (chmod 0600) on startup; every connecting client must present it in the `hello` frame.

### I3 — Privacy story is marketed but undocumented
- **Evidence:** README headline: "local, privacy-first". No documentation exists on: where audio buffers live, whether transcripts persist, conversation memory encryption, telemetry, data wipe procedure.
- **Fix:** `PRIVACY.md` at repo root. Two pages. Covers all of the above. Explicitly states telemetry = none.

### I4 — `openwakeword==0.4.0` exact pin + monkey-patch
- **Evidence:** `src/audio/ears.py` monkey-patches `openwakeword.utils.AudioFeatures`; `pyproject.toml` pins exact version; Python 3.13 breaks the build.
- **Fix:** Push upstream PR to add `inference_framework` kwarg. Until merged, document the constraint explicitly in `CONTRIBUTING.md`.

### I5 — Conversation memory has no rotation
- **Evidence:** `src/llm/memory.py` appends indefinitely to a JSON file. No summarization.
- **Fix:** When `len(history) > N` (suggest N=40 turns), call LLM to summarize oldest `M` turns into a single system-message paragraph. Replace entries. Keep newest 20 verbatim.

### I6 — Two frontends in the repo
- **Evidence:** `app/` (Tauri/React, canonical) + `ui/` (Godot, legacy, `ui/SUPERSEDED.md` exists).
- **Fix:** Delete `ui/` entirely. Remove all references from README. 

### I7 — Repo hygiene
- **Evidence:** Committed: `.coverage` (53 KB), `validation_set_features.npy` (185 MB), `test_canwrite.txt` (21 KB), `.codex` (empty), `*.tmp` files in `tests/`, `.run_logs/`.
- **Fix:** Add to `.gitignore`, remove from history with `git filter-repo --path validation_set_features.npy --invert-paths`.

---

## Items to Cut from MVP

| Item | Reason |
|---|---|
| Vision tool (`moondream2`) | 1.5 GB extra model, slow cold start, low UX value |
| Lip-sync visemes in wire protocol | Backend sends, frontend ignores; net cost, zero gain until avatar ships |
| Multi-edition hardware detection (light/standard/pro) | Just document a 4/8/12 GB recommended config in README |
| Training scripts (`train_lumi.py`, `synth_dataset.py`, `eval_persona.py`) | Move to a separate `lumi-training` repo; keeps public repo runtime-focused |
| Three separate Tauri windows on first launch | Collapse Chat + Settings into collapsible overlay panels |

---

## Complete Prioritised Action Plan

### Ring 1 — Make it installable (target: 2 weeks)

**Status (post-audit, 2026-05-02):** 1 of 5 items genuinely shipped. The other 4 were marked DONE prematurely — code landed, tests are green, but user-facing goals are unmet. See **Ring 1.5 Punch List** below for the corrective work required before Ring 2 can begin.

| Priority | Task | Linked Issue | Status |
|---|---|---|---|
| 1 | Drop `ws_bridge`; move WebSocket server into Brain | B3 | ✅ **DONE** (2026-05-02) |
| 2 | Cross-platform OS-tool adapters (strategy pattern, Linux + Windows first) | B1 | ⚠️ **PARTIAL** — macOS GUI launch broken; mypy unconfigured |
| 3 | Push-to-talk global hotkey (Ctrl+Space) as wake-word fallback | B5 | ⚠️ **PARTIAL** — new fields not in `FIELD_META`; UI cannot toggle |
| 4 | First-run model downloader as a Tauri UI screen | C1 | ❌ **REOPENED** — setup screen unreachable for new users |
| 5 | Tauri build artifact for one platform (Linux AppImage) | B2 | ❌ **REOPENED** — bundle has no Brain; users get an empty shell |

---

### Ring 1.5 — Corrective Work (must complete before Ring 2)

**Definition of Done (Ring 1):** A new user downloading `Lumi_0.1.0_amd64.AppImage` on a clean machine can complete first-run setup *without* editing `config.yaml` or running terminal commands. Until that test passes, Ring 1 is not done.

**Punch List — order by leverage:**

| # | Linked Issue | Task | File(s) | Severity |
|---|---|---|---|---|
| 1.5.a | C1 | ~~Convert `_check_llm_package`, `_check_microphone`, `_check_wake_word_model` to soft returns.~~ **DONE 2026-05-02** — All three now return `list[str]`. OWW version check still hard-fails but only when model is present (wrong version = real deployment bug). `main.py` gates Ears on absence of wake-word missing items. | `src/core/startup_check.py` | CRITICAL |
| 1.5.b | C1 | ~~Default `ipc.enabled: true` in `config.yaml`.~~ **DONE 2026-05-02** | `config.yaml` | CRITICAL |
| 1.5.c | B5 | ~~Add `audio.wake_word_enabled`, `audio.ptt_enabled`, `audio.ptt_hotkey` to `FIELD_META`.~~ **DONE 2026-05-02** — Three entries added to `config_schema.py`. | `src/core/config_schema.py` | CRITICAL |
| 1.5.d | B2 | **RE-SCOPED** — Brain sidecar bundling (PyInstaller + Tauri `externalBin`) is ~2-3 days of dedicated work and is promoted to Ring 2 as an explicit item. The current AppImage remains a dev-only artifact. See Ring 2 table. | Ring 2 | CRITICAL |
| 1.5.e | C1 | ~~Rescope C1 description.~~ **DONE 2026-05-02** — `SetupPanel` docstring updated to accurately describe it as "static guidance panel" (no download progress, no mic test). Those features follow from 1.5.d. | `app/src/components/SetupPanel.tsx` | CRITICAL |
| 1.5.f | C1 | ~~Add `core:clipboard:allow-write-text` to Tauri capabilities.~~ **DONE 2026-05-02** | `app/src-tauri/capabilities/default.json` | HIGH |
| 1.5.g | B1 | ~~Fix `AppLaunchTool` macOS path.~~ **DONE 2026-05-02** — `_MACOS_BUNDLE_APPS` dict + `_launch_macos_bundle()` method added; `open -a BundleName` used for `.app` bundles on Darwin. | `src/tools/os_actions.py` | HIGH |
| 1.5.h | B1 | ~~Add `pyperclip.*` and `pygetwindow.*` to mypy overrides.~~ **DONE 2026-05-02** | `pyproject.toml` | HIGH |
| 1.5.i | B5 | ~~Validate hotkey input in `_to_pynput_hotkey`.~~ **DONE 2026-05-02** — Raises `ValueError` on empty/blank input; strips empty `+` segments. | `src/audio/hotkey.py` | MEDIUM |
| 1.5.j | B5 | ~~Make `PTTListener.start()` idempotent.~~ **DONE 2026-05-02** — Returns early if `_listener is not None`. | `src/audio/hotkey.py` | MEDIUM |
| 1.5.k | B5 | ~~Document `hey_lumi.onnx` provenance in `CONTRIBUTING.md`.~~ **DONE 2026-05-02** — New file with provenance, license, and Python version constraint note. | `CONTRIBUTING.md` | MEDIUM |
| 1.5.l | B3 | ~~Update stale docstrings in `event_bridge.py` and `handshake.py`.~~ **DONE 2026-05-02** — All 4 `IPCTransport` docstring mentions replaced with `WSTransport`. | `src/core/event_bridge.py`, `src/core/handshake.py` | LOW |
| 1.5.m | B2 | Tighten CSP from `null` to `default-src 'self'; connect-src ws://127.0.0.1:*` — **deferred until 1.5.d** (Brain sidecar changes the final connect-src). | `app/src-tauri/tauri.conf.json` | LOW (after 1.5.d) |
| 1.5.n | B2 | Add `libportaudio2` to `.deb` `Depends:` — **deferred until 1.5.d**. | Tauri deb config | LOW (after 1.5.d) |
| 1.5.o | C1 | ~~Move `system_status` handling into `useLumiState`.~~ **DONE 2026-05-02** — `systemStatus: SystemStatus | null` field added to `LumiState`; `OverlayRoot` reads from hook instead of local state. | `app/src/state/useLumiState.ts`, `app/src/roots/OverlayRoot.tsx` | LOW |

**Acceptance test for Ring 1 closure (must pass on a fresh machine):**
1. `chmod +x Lumi_0.1.0_amd64.AppImage && ./Lumi_0.1.0_amd64.AppImage`
2. App opens; SetupPanel appears listing missing models.
3. User downloads models per the panel's instructions (or the panel downloads them itself, per 1.5.e).
4. After restart, app shows the live overlay with no setup screen.
5. PTT toggle in Settings works (per 1.5.c) and Ctrl+Space activates the assistant.

If any of those steps require the user to open a terminal, edit a file, or run a Python command, Ring 1 is not done.

---

### Ring 2 — Make it useful + complete the installer (target: weeks 3-4)

| Priority | Task | Linked Issue |
|---|---|---|
| 6 | **Brain sidecar bundling** — PyInstaller standalone binary + Tauri `externalBin`; Brain auto-starts with the app; AppImage becomes self-contained | B2 (promoted from 1.5.d) |
| 7 | Train persona LoRA (200–300 examples, Q4_K_M, bundle) | B4 |
| 8 | Streaming TTS on sentence boundaries | C4 |
| 9 | Web search tool (DuckDuckGo, no API key) + datetime/timer tools | C3 |
| 10 | End-to-end integration smoke test | C5 |

### Ring 3 — Make it polished (target: week 4)

| Priority | Task | Linked Issue |
|---|---|---|
| 10 | Real avatar artwork or procedural animated fallback | C2 |
| 11 | Privacy/threat-model docs + IPC token handshake | I2, I3 |
| 12 | Conversation memory rotation + LLM summarisation | I5 |
| 13 | Delete `ui/` (Godot legacy) | I6 |
| 14 | Repo hygiene: `.gitignore` additions, scrub binaries | I7 |
| 15 | Orchestrator decomposition (RagSubsystem, ToolsSubsystem, AudioSubsystem) | I1 |
| 16 | `openwakeword` upstream PR or vendor fork | I4 |

---

## Three Highest-Leverage Moves (Updated post-audit, 2026-05-02)

1. ✅ **Drop `ws_bridge`** — DONE. Architecture is now 2-process; B3 cleanly delivered.
2. ❌ **Train the persona LoRA** — Still pending; remains the single biggest UX uplift. (B4, Ring 2)
3. ⚠️ **Ship a Tauri AppImage** — Build pipeline works, but the bundle ships an empty shell (no Brain). The "rough edges" the build was supposed to surface are still latent because no one can actually use the artifact end-to-end yet. Re-scope or extend in Ring 1.5 (item 1.5.d).

**Revised timeline:**
- Ring 1.5 punch list: ~1 day (most fixes are config/schema)
- Brain sidecar bundling (1.5.d): ~2–3 days standalone, the long pole
- Ring 2 (B4 persona LoRA, C4 streaming TTS, C3 web search/datetime, C5 E2E smoke test): ~1.5 weeks once Ring 1.5 is complete
- Realistic public MVP: 4–5 weeks total from 2026-05-02

---

## What NOT to Do Next

- Do not refactor the orchestrator further. Ring 1 doesn't depend on it.
- Do not implement more wire events. The protocol is stable and complete.
- Do not add more unit tests to existing modules. Add the one E2E smoke test (C5) and stop.
- Do not finish avatar lip-sync. Visemes are wire cost with zero UI benefit until artwork ships.
- Do not chase new `MetricsCollector`-style infrastructure. Add it only when users report performance issues you cannot diagnose.

**Stop polishing. Start shipping.**
