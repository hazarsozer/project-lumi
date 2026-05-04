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

### LLM inference

- All language model inference runs locally via `llama-cpp-python`. The
  model weights are stored in `models/llm/` on your machine.
- Your prompts and responses **never leave your device**.

### Web search tool (opt-in)

- If you ask Lumi to search the web, it sends your search query to
  [DuckDuckGo's HTML endpoint](https://html.duckduckgo.com/html/) over
  HTTPS. **This is the only network request Lumi makes.** No account or
  API key is required.
- If you do not use the web search tool, no network requests are made.

### IPC token

- On each startup, Lumi writes a single-use 64-character bearer token to
  `~/.lumi/ipc_token` (permissions: `0600`, owner read-only). This token
  is used to authenticate the Tauri frontend process against the Brain's
  WebSocket server, preventing other local processes from sending commands.
  The token is regenerated on every launch.

---

## Telemetry

**None.** Lumi contains no analytics, crash reporting, usage metrics, or
any other form of telemetry. The codebase has no calls to external
analytics endpoints.

---

## Data wipe procedure

To completely remove all data Lumi has stored on your machine:

```bash
# Remove conversation history and IPC token
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
