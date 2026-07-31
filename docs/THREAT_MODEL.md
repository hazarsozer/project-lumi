# IPC Threat Model — Project Lumi

_Last updated: 2026-05-30 (Ring 3 + hardening/crucible-v1.0)_

---

## Overview

The Lumi Brain process exposes a WebSocket server on `ws://127.0.0.1:5556`
for communication with the Tauri/React frontend. This document describes
the attack surface of that interface, the threat actors, the mitigations
in place, and the known remaining gaps.

---

## Assets at risk

| Asset | Value |
|---|---|
| OS tools | Clipboard read/write, file metadata, window listing, app launch — triggered by inbound `user_text` events |
| Screen contents (vision, opt-in) | Full-screen PNG capture passed to local moondream2 model; only active when `vision.enabled: true` and `screenshot` is in `tools.allowed_tools` |
| Conversation history | JSON file at `~/.lumi/memory/conversation.json` — readable via IPC indirectly |
| RAG document store | SQLite at `~/.lumi/rag.db` — readable via IPC indirectly when RAG is enabled |
| LLM inference budget | CPU/GPU compute — any connected client can trigger inference |

---

## Threat actors

### T1 — Malicious local process

A process running on the same machine (e.g., malware, a malicious browser
extension with local loopback access, a compromised application) connects
to `ws://127.0.0.1:5556` and sends `user_text` events to trigger OS tools
or extract conversation history.

**Severity:** High. OS tools include clipboard (credential theft),
file metadata inspection, and app launches. If vision is enabled, a
successful connection could trigger a screen capture.

### T2 — Drive-by web page via localhost loopback

A malicious web page loaded in a browser may attempt to connect to
`ws://127.0.0.1:5556` via JavaScript's WebSocket API. Modern browsers
enforce the Private Network Access spec (Chrome 98+, Firefox 120+), which
blocks cross-origin requests to localhost by default. However, older
browsers and non-browser WebSocket clients are not protected.

**Severity:** Medium (depends on browser version and configuration).

---

## Mitigations implemented (Ring 3)

### M1 — Localhost binding

The WebSocket server binds exclusively to `127.0.0.1` (loopback). It is
unreachable from other machines on the network.

**Residual risk:** Does not protect against local processes (T1).

### M2 — Single-client enforcement

`WSTransport` rejects any second connection attempt with WebSocket close
code 1008. Only one client can be connected at a time.

**Residual risk:** An attacker process that connects before the legitimate
frontend can hold the connection.

### M3 — Bearer token handshake

On each Brain startup:
1. A 64-character cryptographically random hex token is generated via
   `secrets.token_hex(32)`.
2. It is written to `~/.lumi/ipc_token` with permissions `0600`
   (owner-read only).
3. The Tauri frontend reads this file (via a Rust command `read_ipc_token`)
   and presents the token in its `hello_ack` frame.
4. `HandshakeHandler` verifies the token using `hmac.compare_digest`
   (constant-time — no timing oracle).
5. On mismatch: the client is disconnected with WebSocket close code 1008.
6. On timeout (no `hello_ack` within 3 seconds): with token auth enabled,
   the client is disconnected (fail-closed). Without token auth (dev mode),
   the connection is allowed to continue with a warning log.

A process without access to the current user's home directory cannot read
the token and cannot pass the handshake.

**Residual risk:** See known gaps below.

### M4 — Tool allowlist

`ToolExecutor` enforces a `tools.allowed_tools` allowlist. Any tool name
not in the list is rejected before execution. The `web_search` and
`screenshot` tools are **not** in the default allowlist — they require
explicit user opt-in in `config.yaml`. This limits the blast radius of a
successful IPC compromise to the tools the user has deliberately enabled.

### M5 — Tool-argument logging redaction

`ToolExecutor` logs only the tool **name** and argument **keys** at the
`INFO` level. Raw argument values (search queries, file paths, clipboard
content) are logged only at `DEBUG` level, which is disabled in the
default configuration. This limits the exposure of sensitive content in
application logs.

### M6 — FileInfoTool path restriction

`FileInfoTool` resolves symlinks and checks the result against an allowlist
of safe roots (`~/Documents`, `~/Downloads`, `~/Desktop`, `~/Music`,
`~/Pictures`, `~/Videos`). Sensitive home subtrees (`~/.ssh`, `~/.lumi`,
`~/.aws`, `~/.gnupg`, etc.) are denied even if they fall under a permitted
root. Absolute paths are stripped from tool output — only paths relative to
the allowed root are surfaced to the LLM or UI.

---

## Known gaps

### G1 — No TLS on localhost

Traffic between the Brain and the frontend is unencrypted plaintext
WebSocket. On the loopback interface, eavesdropping requires kernel-level
privileges (root or the same UID). For a local desktop app, TLS on
loopback is generally not considered necessary, but it is a gap for
environments where multiple UIDs share a machine.

**Accepted for MVP.** TLS on localhost adds certificate management
complexity with minimal practical gain for the single-user desktop use
case.

### G2 — Token file race window

Between the Brain writing `~/.lumi/ipc_token` and the frontend connecting,
there is a small window (< 1 second) during which the file exists but the
server is not yet listening. A racing attacker process could read the token
before the frontend and race to connect first.

**Accepted for MVP.** Exploiting this race requires the attacker to
already be running as the same user, which grants equivalent access to the
home directory regardless.

### G3 — OS tool inputs not fully sanitized for injection

The OS tools (`AppLaunchTool`, `ClipboardTool`, etc.) accept arbitrary
string inputs from the LLM. `AppLaunchTool` enforces an application-name
allowlist and resolves only known binaries via `shutil.which()` — no
user-supplied string is ever passed directly to subprocess. `FileInfoTool`
enforces path allowlisting and path-traversal rejection. However, clipboard
content length and encoding are not validated against injection patterns,
and `WindowListTool` has no sanitization (read-only, low risk).

**Partially mitigated (M4, M6).** Full prompt-injection sanitization is
tracked as a post-MVP hardening item.

### G4 — Vision model: full-screen capture scope

When vision is enabled and the `screenshot` tool is invoked, the entire
screen is captured regardless of what is currently displayed. There is no
window-selection filter, no PII scrubbing, and no user confirmation prompt
before capture. The captured image is processed entirely locally (moondream2
GGUF on-device), but a prompt-injection attack that triggers the tool could
capture sensitive screen content and include it in the LLM context.

**Accepted for MVP.** Vision is opt-in (`vision.enabled: false` by
default) and requires explicit allowlisting in `tools.allowed_tools`.
Future work: per-invocation user confirmation prompt.

---

## Recommended future work

1. **Audit clipboard content** — validate content length and encoding on
   clipboard write; consider a size cap for clipboard read results returned
   to the LLM.
2. **Log rejected connections** — emit a structured warning event when a
   client fails the token handshake, so users can detect unauthorized
   access attempts.
3. **Short token lifetime** — regenerate the token after each successful
   connection rather than once per startup. Reduces the replay window.
4. **Unix socket option** — for Linux, a Unix domain socket at
   `~/.lumi/brain.sock` with `0600` permissions would replace the localhost
   TCP port entirely, eliminating loopback eavesdropping risk and the T2
   threat.
5. **Vision: per-invocation consent** — prompt the user before each
   screenshot capture so that prompt-injection attacks cannot silently
   capture screen contents.
