# Manual Smoke Test Protocol — Phase 5

## Prerequisites

- Python Brain running: `uv run python -m src.main` (from repo root)
- Godot project opened from `ui/` directory

## Steps

1. **Start Brain** — `uv run python -m src.main` (from Lumi repo root).
   Expected: Brain starts, logs "IPCTransport listening on 127.0.0.1:5555".

2. **Open Godot project** — Open `ui/project.godot` in Godot 4 editor, press F5 to run.
   Expected: Transparent overlay window appears; console logs "connected to Brain".

3. **Verify idle state** — Avatar should show idle_breathe animation.

4. **Say wake word** — Speak "Hey Lumi" into microphone.
   Expected: Avatar transitions to listening_pulse animation.

5. **Speak a command** — Say any command.
   Expected: Avatar transitions to processing_spin during STT+LLM.

6. **Verify speaking** — When LLM responds and TTS plays.
   Expected: Avatar transitions to speaking_lipsync; mouth open/close with viseme events.

7. **Press Escape** — While Brain is in PROCESSING or SPEAKING.
   Expected: Brain receives interrupt, transitions to idle; avatar returns to idle_breathe.

## Notes

- Transparent window requires a compositor (X11: Picom/Compton; Wayland: native).
- If the window is not transparent, check `Project Settings → Display → Window → Transparent`.
- Placeholder sprites are colored circles. Replace with real assets in Phase 6.

---

## Citation Panel — Wave F3 Polish

These tests verify the `rag_retrieval` wire event drives the polished `CitationPanel`.

### Prerequisites

- Brain is running with RAG enabled (`rag.enabled = true` in config).
- At least one document has been indexed so retrieval returns results.

### Checklist

- [ ] **Header shows result count and latency** — Ask Lumi a question that triggers
  retrieval.
  Expected: Panel header reads e.g. "Sources (3 results, 42ms)" — not just "Sources".

- [ ] **Query subtitle appears** — The panel shows the query string in smaller grey
  italic text directly below the header row.
  Expected: Subtitle matches the text of the query that triggered retrieval.
  If the payload has an empty `query` field, the subtitle row is hidden.

- [ ] **Paths are truncated to 2 components** — Source rows show only the last two
  path segments, e.g. `notes/ideas.txt` not the full absolute path.
  Expected: No leading slashes or intermediate directory components visible.

- [ ] **Close button (X) dismisses panel** — Click the "X" button in the panel header.
  Expected: Panel hides immediately. The button is labelled "X", not "Hide".

- [ ] **Auto-hide after 8 seconds (no interaction)** — Trigger a RAG query and do
  not move the mouse over the panel.
  Expected: Panel disappears automatically after approximately 8 seconds.

- [ ] **Auto-hide cancelled by hover** — Trigger a RAG query and move the mouse
  cursor over the panel within 8 seconds. Keep it there beyond 8 seconds.
  Expected: Panel does NOT auto-hide while the mouse is over it.

- [ ] **Panel re-shows on second RAG query** — Without restarting Lumi, trigger
  another RAG-backed query.
  Expected: Panel reappears and is repopulated with the new query's sources.
  No stale rows from the previous query should remain.

- [ ] **Panel absent for non-RAG responses** — Ask a question that does not trigger
  retrieval (e.g. a simple greeting when RAG is disabled or no docs are indexed).
  Expected: CitationPanel stays hidden; no "rag_retrieval" event is received.

- [ ] **Malformed payload is silently skipped** — Manually send a `rag_retrieval`
  frame with `top_doc_paths` omitted (e.g. using a raw TCP test client).
  Expected: A `push_warning` appears in the Godot console; panel does not show
  and the application does not crash.

---

## RAG Toggle — Wave F4

These tests verify the Ctrl+R toggle and the status pill.

### Prerequisites

- Godot project running (Brain does not need to be active for pill/shortcut tests,
  but must be running to verify `set_config` wire events).

### Checklist

- [ ] **Pill visible at startup** — Launch the Godot project.
  Expected: A small pill in the top-left corner reads "RAG ON" with a green
  background (default state on first launch).

- [ ] **Ctrl+R toggles to OFF** — Press Ctrl+R once.
  Expected: Pill changes to "RAG OFF" with a grey background immediately.

- [ ] **Ctrl+R toggles back to ON** — Press Ctrl+R again.
  Expected: Pill returns to "RAG ON" green.

- [ ] **set_config event sent to Brain** — With Brain running, press Ctrl+R.
  Expected: Brain console/logs show a `set_config` frame received with
  `{"key": "rag_enabled", "value": false}` (or `true` on the next toggle).

- [ ] **State persists across restarts** — Toggle RAG to OFF (Ctrl+R), then close
  and relaunch the Godot project.
  Expected: Pill shows "RAG OFF" grey on startup — the preference was saved.

- [ ] **set_config synced on reconnect** — With Brain stopped, toggle RAG state via
  Ctrl+R, then start the Brain.
  Expected: When the Body reconnects, it immediately sends a `set_config` event
  matching the current persisted state.

- [ ] **Pill does not steal focus** — While a text input field in the UI is focused,
  pressing Ctrl+R should still toggle RAG.
  Expected: Toggle fires; text field focus is not disrupted.
  Note: `_unhandled_input` is used so the shortcut only fires when no other
  Control has consumed the event. Verify by typing in any input field first.
