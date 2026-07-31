# Privacy — Project Lumi

Lumi runs **entirely on your device**. There is no cloud connection, no
account registration, no telemetry, and no external data transfer — with
one explicit exception documented below.

---

## What Lumi processes and where

### Audio

- Your microphone is read by the wake-word engine and the push-to-talk
  trigger. Audio samples are held in RAM as NumPy arrays for the duration
  of a single recognition window (≤ 10 seconds by default).
- Audio buffers are **never written to disk**. Nothing is logged, uploaded,
  or cached beyond the active processing window.

### Speech transcripts

- When speech is detected, faster-whisper transcribes it to text **locally**
  using a bundled model. The raw audio is discarded immediately after
  transcription.
- Transcripts are passed to the local LLM for response generation. They are
  **not persisted separately** — only the final conversation turn is saved
  to the conversation history file (see below).

### Conversation history

- Your conversation is saved to a JSON file at
  `~/.lumi/memory/conversation.json`. This file is **local only** — it is
  never transmitted.
- The file is not encrypted. It is readable by any user with access to your
  home directory.
- When the conversation exceeds 40 turns, Lumi summarizes the oldest turns
  using the local LLM and replaces them with a single summary entry. No
  external service is involved in this summarization.
- **Retention (opt-in):** By default, conversation history is kept
  indefinitely. You can enable automatic age-based purging by setting
  `llm.memory_max_age_days` in `config.yaml` to a positive number of days
  (e.g. `30.0`). Entries older than that threshold are removed on every
  load and save. The default value is `0.0`, which disables retention
  entirely.

### Tool execution logging

- When a tool is invoked (e.g. clipboard, file info, screenshot), Lumi
  logs the tool **name** and the **keys** of its arguments at the `INFO`
  level. Raw argument values — such as search queries, file paths, or
  clipboard text — are recorded only at the `DEBUG` level, which is not
  emitted in the default (`INFO`) log configuration.
- Logs are written to the local application log file only, never
  transmitted.

### Screen capture and vision (opt-in)

- Lumi includes a **screenshot and vision tool** (`ScreenshotTool`) that,
  when triggered by the LLM, captures the full screen and returns a text
  description of what is visible.
- **How capture works:** Lumi uses `grim` (Wayland), `scrot` (X11), or
  Pillow's `ImageGrab` — whichever is available — to capture a PNG of the
  entire display. If the captured image exceeds `max_resolution` (default
  1280 px on the longest side), it is downscaled proportionally before
  processing.
- **Where the image goes:** The PNG bytes are passed entirely **locally** to
  a moondream2 GGUF model stored in `models/vision/moondream2.gguf` on your
  machine. The image is held in RAM only for the duration of the inference
  call. It is **never written to disk** and **never transmitted off-device**.
- **Opt-in by default:** Vision is disabled unless you set
  `vision.enabled: true` in `config.yaml` and download the moondream2 GGUF
  model manually. When disabled, no screenshots are ever taken. The tool
  must also be present in `tools.allowed_tools` for the LLM to invoke it.
- **VRAM management:** When vision runs, the main LLM is temporarily
  unloaded to free VRAM. The vision model is unloaded automatically
  30 seconds after its last use.
- **Sensitive windows:** The capture includes everything visible on screen
  at the time of the call — including any open windows, documents, or
  browser tabs. Only enable vision if you are comfortable with the local
  LLM receiving a text description of your screen contents.

### LLM inference

- All language model inference runs locally via `llama-cpp-python`. The
  model weights are stored in `models/llm/` on your machine.
- Your prompts and responses **never leave your device**.

### RAG document store (opt-in)

- If you enable the RAG (knowledge retrieval) feature, Lumi indexes
  documents from `~/.lumi/docs/` into a SQLite database at `~/.lumi/rag.db`.
  This file is **local only** — it is never transmitted.
- **Retention (opt-in):** By default, indexed documents are kept
  indefinitely. You can enable automatic age-based purging by setting
  `rag.rag_max_age_days` in `config.yaml` to a positive number of days.
  The default value is `0.0`, which disables retention entirely.

### Web search tool (opt-in)

- If you configure `web_search` in `tools.allowed_tools` and ask Lumi to
  search the web, it sends your search query to
  [DuckDuckGo's HTML endpoint](https://html.duckduckgo.com/html/) over
  HTTPS. **This is the only network request Lumi makes.** No account or
  API key is required.
- The `web_search` tool is **not** in the default `tools.allowed_tools`
  list — you must explicitly add it to enable web search. If you do not
  enable and use the web search tool, no network requests are made.

### IPC token

- On each startup, Lumi writes a single-use 64-character bearer token to
  `~/.lumi/ipc_token` (permissions: `0600`, owner read-only). This token
  is used to authenticate the Tauri frontend process against the Brain's
  WebSocket server, preventing other local processes from sending commands.
  The token is regenerated on every launch.

### File access tool (security-restricted)

- The `file_info` tool allows Lumi to inspect filesystem metadata. It is
  restricted to an allowlist of safe roots under your home directory
  (Documents, Downloads, Desktop, Music, Pictures, Videos). Paths outside
  this allowlist — including system directories, `~/.ssh`, `~/.lumi`, and
  other sensitive home subtrees — are rejected before any filesystem
  operation. Absolute paths are never included in tool output; only paths
  relative to the allowed root are surfaced.

---

## Telemetry

**None.** Lumi contains no analytics, crash reporting, usage metrics, or
any other form of telemetry. The codebase has no calls to external
analytics endpoints.

---

## Data wipe procedure

To completely remove all data Lumi has stored on your machine:

```bash
# Remove conversation history, RAG database, and IPC token
rm -rf ~/.lumi/

# Remove model weights (large — only if you want to reclaim disk space)
rm -rf /path/to/lumi/models/

# Uninstall the application
# Linux .deb:   sudo dpkg -r lumi
# AppImage:     delete the .AppImage file
```

---

## IPC threat model

See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for a detailed analysis
of the local IPC attack surface and implemented mitigations.
