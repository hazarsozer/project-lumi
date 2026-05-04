# IPC Threat Model — Project Lumi

_Last updated: 2026-05-04 (Ring 3)_

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
| OS tools | Clipboard read/write, screenshot capture, app launch — trigged by inbound `user_text` events |
| Conversation history | JSON file at `~/.lumi/memory/conversation.json` — readable via IPC indirectly |
| LLM inference budget | CPU/GPU compute — any connected client can trigger inference |

---

## Threat actors

### T1 — Malicious local process

A process running on the same machine (e.g., malware, a malicious browser
extension with local loopback access, a compromised application) connects
to `ws://127.0.0.1:5556` and sends `user_text` events to trigger OS tools
or extract conversation history.

**Severity:** High. OS tools include clipboard (credential theft),
screenshots (screen capture), and app launches.

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

A process without access to the current user's home directory cannot read
the token and cannot pass the handshake.

**Residual risk:** See known gaps below.

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

### G3 — OS tool inputs not sanitized for injection

The OS tools (`AppLaunchTool`, `ClipboardTool`, etc.) accept arbitrary
string inputs from the LLM. If a compromised prompt causes the LLM to emit
a crafted `<tool_call>` payload, the tool executes it. There is no
allowlist of allowed app names or clipboard content shapes.

**Deferred.** Requires a more complete threat model of prompt injection
scenarios. Tracked as a post-MVP hardening item.

---

## Recommended future work

1. **Audit OS tool inputs** — add allowlisting for `AppLaunchTool` target
   names; validate clipboard content length and encoding.
2. **Log rejected connections** — emit a structured warning event when a
   client fails the token handshake, so users can detect unauthorized
   access attempts.
3. **Short token lifetime** — regenerate the token after each successful
   connection rather than once per startup. Reduces the replay window.
4. **Unix socket option** — for Linux, a Unix domain socket at
   `~/.lumi/brain.sock` with `0600` permissions would replace the localhost
   TCP port entirely, eliminating loopback eavesdropping risk and the T2
   threat.
