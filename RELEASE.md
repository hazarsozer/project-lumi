# Lumi v1.0 Release Definition of Done

This document defines what "shipped" means for the v1.0 tag and provides a
manual E2E checklist to run before tagging. Work through every flow and confirm
every pass criterion before pushing the release tag.

---

## v1.0 Definition of Done

### Features complete

- [ ] Wake-word detection active in IDLE state (openWakeWord, `hey_lumi.onnx`)
- [ ] Voice activity detection stops recording on silence
- [ ] Speech-to-text transcription via faster-whisper (tiny.en, int8)
- [ ] LLM inference via llama-cpp-python (Phi-3.5 Mini Q4_K_M or equivalent GGUF)
- [ ] Reflex router handles greetings and time queries without LLM invocation
- [ ] Two-pass tool-call loop: `<tool_call>` emitted by LLM, executed, result injected, second pass produces final response
- [ ] OS tools available: `launch_app`, `clipboard`, `file_info`, `window_list`
- [ ] TTS playback via Kokoro-82M ONNX, non-blocking SpeakerThread
- [ ] Viseme events posted per phoneme for avatar lip-sync
- [ ] LLM token streaming via `llm_token` IPC wire frame
- [ ] IPC server active and connected to Tauri/React frontend (WebSocket on `ws://127.0.0.1:5556`; `ipc.enabled: true` default)
- [ ] Tauri/React overlay renders avatar in four states: IDLE, LISTENING, PROCESSING, SPEAKING
- [ ] Settings panel (gear icon / Ctrl+,) applies live config changes and marks restart-required fields
- [ ] RAG retriever available as an opt-in toggle (disabled by default)
- [ ] `scripts/ingest_docs.py` indexes documents into `~/.lumi/rag.db`
- [ ] Interrupt handling: new wake word or Escape key cancels in-flight LLM/TTS and returns to IDLE
- [ ] `scripts/doctor.py` pre-flight diagnostics pass on clean install

### Tests

- [ ] `uv run pytest tests/ --cov=src --cov-fail-under=80` exits 0
- [ ] No test failures (skipped tests documented in CI)
- [ ] `tests/test_e2e_smoke.py` passes

### Smoke test

- [ ] `scripts/smoke_live.py` completes one full voice turn without errors
- [ ] `scripts/measure_base_latency.py` reports p95 < 1.7 s on target hardware
- [ ] `scripts/measure_rag_latency.py` reports p95 < 2.0 s when RAG is enabled

### Docs

- [ ] `ARCHITECTURE.md` reflects actual file tree and phase status
- [ ] `config.yaml` documents every key with inline comments
- [ ] `app/README.md` (or repo root `README.md`) covers Tauri/React setup steps
- [ ] `RELEASE.md` manual checklist is completed and all flows below are signed off

---

## 5-Flow E2E Checklist

Each flow is independent. Work through them in order since later flows depend on
a correctly running system established by earlier ones.

---

### Flow 1: Cold Start

**What it tests:** Fresh process start, startup validation, all subsystems reach
ready state without crashing.

**Prerequisites:**

- `models/hey_lumi.onnx` present
- `models/faster-whisper-tiny.en/` directory present (or Hugging Face download allowed)
- `models/llm/phi-3.5-mini.gguf` (or whichever GGUF is configured in `config.yaml`)
- `models/tts/kokoro-v1_0.onnx` and `models/tts/voices.bin` present
- `config.yaml` has `tts.enabled: true`, `ipc.enabled: false` for this flow
- A microphone present and accessible

**Procedure:**

1. Run `uv run python scripts/doctor.py` and verify all hard checks pass.
2. Run `uv run python -m src.main`.
3. Observe the log output for the following lines (order matters):
   - `Startup checks passed`
   - `StateMachine initialised — state: IDLE`
   - `Ears thread started`
4. Confirm no `ERROR` or `CRITICAL` log lines appear during the first 5 seconds.
5. Confirm the process remains running (no exit code).

**Pass criteria:**

- `doctor.py` reports no hard failures.
- `src.main` reaches `IDLE` state and holds it.
- No exceptions logged during startup.
- Process is responsive (does not deadlock or spin at 100% CPU).

---

### Flow 2: Wake-Word Turn

**What it tests:** End-to-end voice interaction — wake-word detection, VAD
recording, STT transcription, LLM inference, TTS playback.

**Prerequisites:**

- Flow 1 passed.
- Working microphone and speakers.
- `tts.enabled: true` in `config.yaml`.
- LLM model loaded (`llm.model_path` points to a valid GGUF).
- `ipc.enabled: false` for this flow (audio-only, no UI needed).

**Procedure:**

1. Start `uv run python -m src.main` (or use the session from Flow 1).
2. Wait for `IDLE` state in the logs.
3. Say "Hey Lumi" clearly into the microphone.
4. Wait for the ready-sound ping (880 Hz tone).
5. Say a short factual query, e.g. "What time is it?" or "Hello."
6. Observe logs for the following sequence:
   - `WakeDetectedEvent` posted
   - State transition: `IDLE -> LISTENING`
   - `RecordingCompleteEvent` posted (after silence timeout)
   - State transition: `LISTENING -> PROCESSING`
   - `TranscriptReadyEvent` with your spoken text
   - Either `CommandResultEvent` (reflex) or `LLMResponseReadyEvent` (reasoning)
   - State transition: `PROCESSING -> SPEAKING`
   - `SpeechCompletedEvent`
   - State transition: `SPEAKING -> IDLE`
7. Confirm you hear a spoken response through the speakers.

**Pass criteria:**

- All six state transitions occur in order.
- Transcript contains recognisable words from what you said.
- TTS produces audible output.
- System returns to `IDLE` after speech completes.
- No `ERROR` events logged during the turn.

---

### Flow 3: Text Turn

**What it tests:** Text input path via IPC (`user_text` wire event from the
Tauri/React frontend), LLM response, streamed tokens displayed in the chat panel.

**Prerequisites:**

- Flow 1 passed.
- Node.js and Rust toolchain installed; `app/` dependencies installed (`cd app && npm install`).
- `ipc.enabled: true` in `config.yaml` (default).
- `tts.enabled: true` or `false` (either is acceptable for this flow).
- LLM model loaded.

**Procedure:**

1. Start `uv run python -m src.main` (Brain starts WebSocket server on `ws://127.0.0.1:5556`).
2. In a second terminal: `cd app && npm run tauri dev`.
3. Confirm the Brain log shows `Client connected` and the overlay window appears.
4. In the chat panel, type a short message, e.g. "What is 2 plus 2?" and submit.
5. Observe the Brain logs:
   - `UserTextEvent` received
   - State transitions: `IDLE -> PROCESSING -> SPEAKING` (or `IDLE -> PROCESSING -> IDLE` if TTS disabled)
   - `LLMTokenEvent` entries streaming tokens
6. Confirm tokens appear progressively in the chat panel.
7. Confirm the response makes sense for the query.

**Pass criteria:**

- Brain receives `UserTextEvent` and enters `PROCESSING`.
- At least one `LLMTokenEvent` reaches the frontend and renders in the chat panel.
- Complete response is coherent and displayed in the panel.
- System returns to `IDLE` after the turn completes.

---

### Flow 4: Tool Call

**What it tests:** LLM-generated `<tool_call>` block is parsed, executed through
`ToolExecutor`, result injected into context, second LLM pass produces a spoken
confirmation.

**Prerequisites:**

- Flow 2 or Flow 3 passed (system running with LLM active).
- `tools.enabled: true` in `config.yaml`.
- `launch_app` in `tools.allowed_tools` list.
- A known application exists on the system path (e.g. `gnome-calculator`,
  `kcalc`, or `xcalc` — verify with `which gnome-calculator`).
- The application name is in the `AppLaunchTool` internal allowlist defined in
  `src/tools/os_actions.py`.

**Procedure:**

1. Start the Brain (with IPC enabled if using text input, otherwise voice).
2. Send the query: "Open the calculator" (voice or text).
3. Observe logs for:
   - `TranscriptReadyEvent` or `UserTextEvent` containing "calculator"
   - `ReasoningRouter` entering inference
   - Log line containing `<tool_call>` with `"tool": "launch_app"`
   - `ToolExecutor` log: `Executing tool: launch_app`
   - `ToolResultEvent` posted with success status
   - Second LLM inference pass
   - `LLMResponseReadyEvent` with a confirmation message
4. Confirm the calculator application opens on the desktop.
5. Confirm Lumi speaks or displays a confirmation ("Opened the calculator" or
   similar).

**Pass criteria:**

- `<tool_call>` block is present in the raw LLM output (visible in DEBUG logs).
- Calculator process launches visibly on the desktop.
- `ToolResultEvent` carries `success: true`.
- Lumi's final response acknowledges the action.
- System returns to `IDLE`.

**If tool call is not triggered:** The stock model may not reliably emit
`<tool_call>` blocks without fine-tuning (see Known Limitations). Try a more
direct phrasing: "Use the launch_app tool to open gnome-calculator." If that
fails, verify `tools.enabled: true` and that the tool name is in `allowed_tools`.

---

### Flow 5: RAG Query

**What it tests:** Document ingest, hybrid BM25 + vector retrieval, context
injection into the LLM prompt, cited response returned to the user.

**Prerequisites:**

- Flow 3 passed (IPC working, or voice working from Flow 2).
- `rag.enabled: false` initially in `config.yaml` (will be toggled at runtime).
- `sentence-transformers/all-MiniLM-L6-v2` downloaded or accessible via
  Hugging Face (first run downloads ~80 MB).
- `sqlite-vec` native extension available (`uv run python -c "import sqlite_vec"` exits 0).
- A plain-text or Markdown document prepared for ingest (e.g. a `notes.md` file
  with at least 3–4 paragraphs of unique content).

**Procedure:**

1. Place your test document in `~/.lumi/docs/` (or the path set by `rag.corpus_dir`).
2. Run the ingest script:
   ```
   uv run python scripts/ingest_docs.py
   ```
   Confirm log output shows chunks embedded and stored (e.g. "Stored N chunks for notes.md").
3. Start the Brain with IPC enabled.
4. Enable RAG via the Settings panel (RAG toggle) or by sending the
   `rag_set_enabled` wire event, or restart with `rag.enabled: true` in `config.yaml`.
5. Confirm the Brain logs: `RAG retriever enabled`.
6. Ask a question whose answer is only in the ingested document, e.g.
   "According to my notes, what is the project deadline?" (adjust to match your
   test document content).
7. Observe Brain logs:
   - `route_rag_intent` returns `rag` intent (or explicit trigger phrase used)
   - `RAGRetriever` log: retrieved N chunks, scores above `min_score`
   - `RAGRetrievalEvent` posted
   - `rag_retrieval` wire frame sent to frontend (citation panel populated)
   - `LLMResponseReadyEvent` containing content from the document
8. Confirm the response contains information drawn from your document.
9. Check the citation panel for source document name and relevance score.

**Pass criteria:**

- Ingest completes with no errors; chunk count > 0.
- RAG retriever returns at least one chunk with score above `min_score`.
- LLM response contains content that was only in the ingested document (not
  general LLM knowledge).
- Citation panel displays the source document name.
- `RAGRetrievalEvent` visible in Brain logs with non-empty citations list.
- System returns to `IDLE` after the turn.

---

## Known Limitations (v1.0)

### Model behavior

- **No fine-tuning shipped with v1.0.** The LLM is a stock Phi-3.5 Mini (or
  equivalent) with no Lumi personality, no brevity training, and no
  purpose-trained OS tool-call schema. The model may:
  - Identify itself as "a large language model by Microsoft" instead of "Lumi".
  - Use filler phrases ("Certainly!", "Of course!") that the architecture
    explicitly trains against.
  - Fail to emit well-formed `<tool_call>` blocks without explicit prompting.
  - Include markdown syntax (`**bold**`, `# headers`) in TTS-destined responses,
    causing the TTS to read symbols aloud.
- Fine-tuning pipeline (QLoRA via `scripts/train_lumi.py`) is implemented but
  requires a trained GGUF to be dropped into `models/llm/` — this is a v1.1
  deliverable.

### Frontend / display

- **Avatar artwork is placeholder.** The Tauri/React overlay (`app/src/components/LumiAvatar.tsx`) uses static placeholder images. Commissioned artwork is a v1.1 deliverable.
- **Citation panel** — the `rag_retrieval` wire event is forwarded correctly by the Brain but the React component may need wiring to display citations.
- **Wayland transparency** may not work on all compositors. The transparent overlay is tested on X11. On Wayland (Sway, GNOME Wayland) the window may render with a solid background. Use X11 or XWayland if transparency is required.

### Audio

- **Wake-word model accuracy** is subject to acoustic environment. The `hey_lumi.onnx`
  model may false-trigger on background speech or music. Raise `audio.sensitivity`
  (toward 1.0) in `config.yaml` to reduce false positives at the cost of recall.
- `ears.py` contains an upstream monkey-patch for `openwakeword==0.4.0`. The
  exact version must be installed; mismatches raise a hard `RuntimeError` at
  startup (enforced by `startup_check.py`).
- **Kokoro phoneme tuple format** (`(phoneme_str, start_ms, duration_ms)`) is
  assumed based on the Kokoro ONNX API. Viseme timing may be inaccurate if the
  upstream format differs; verify against a real Kokoro model run.

### Tools

- **`xclip` required** for `ClipboardTool` on Linux. Not installed by default on all distributions. Install with `sudo apt install xclip` or equivalent. On macOS and Windows, `pyperclip` is used instead.
- **`wmctrl` required** for `WindowListTool` on Linux. Install with `sudo apt install wmctrl`. On macOS, `osascript` is used; on Windows, `pygetwindow`.
- **`grim` or `scrot` required** for `ScreenshotTool`. `grim` for Wayland, `scrot`
  for X11. The tool falls back to `Pillow` for basic capture if neither is present,
  but description quality degrades.
- **`moondream2.gguf` not bundled.** `vision.enabled: false` by default; set to
  `true` only after downloading the model to `models/vision/moondream2.gguf`.

### RAG

- **`sqlite-vec` native extension** must be installed on the system or available
  as a Python wheel. CI installs it explicitly; local installs may need
  `uv pip install sqlite-vec`.
- **Citation panel** is not yet rendered in the React frontend (see Frontend above).
- Automatic RAG routing (trigger-word-free) is not implemented. Explicit trigger
  phrases ("search my docs", "look up in my notes") are required to activate
  retrieval.

### TurboQuant KV cache

- `kv_cache_quant: "turbo3"` is commented out in `config.yaml`. Activation is
  blocked on llama.cpp PR #21089 propagating into a `llama-cpp-python` release.
  Do not uncomment until the upstream release ships.

---

## Upgrade Path (v1.1+)

- **Fine-tuned model (v1.1 priority):** QLoRA training pipeline is ready
  (`scripts/train_lumi.py`). Once a trained `lumi-phi35-v1-Q4_K_M.gguf` is
  available, drop it into `models/llm/` and update `llm.model_path`. No code
  changes required.
- **Real avatar artwork (v1.1):** Replace placeholder with commissioned assets
  and wire them into the React avatar component.
- **TurboQuant KV cache:** Uncomment `kv_cache_quant: "turbo3"` in `config.yaml`
  once the upstream `llama-cpp-python` release ships PR #21089. Expected ~0.3 GB
  VRAM saving at 4096-token context.
- **LoRA hot-swap:** After fine-tuning v1, investigate `llama_lora_adapter_set`
  API availability in `llama-cpp-python >= 0.2.90`. If exposed, domain-specific
  LoRA adapters can be swapped in < 100 ms. If not, `ModelRegistry` (already
  implemented) provides full GGUF swapping at 2.5–7 s latency.
- **Automatic RAG routing:** Embedding-based classifier (all-MiniLM-L6-v2) to
  trigger retrieval without explicit trigger phrases. Gated on > 90% precision
  in production traffic.
- **Internet tools (v2+):** Web search and fetch tools. Training category 5
  dataset already specified in `ARCHITECTURE.md` Section 5.
- **pip-installable wheel:** Not yet scoped. Requires resolving native dependency
  packaging for llama-cpp-python, sqlite-vec, and onnxruntime across target
  platforms.
