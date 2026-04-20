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

## Citation Panel (TODO 23)

These tests verify the `rag_retrieval` wire event drives `CitationPanel` correctly.

### Prerequisites

- Brain is running with RAG enabled (`rag.enabled = true` in config).
- At least one document has been indexed so retrieval returns results.

### Checklist

- [ ] **Panel appears on RAG query** — Ask Lumi a question that triggers retrieval.
  Expected: The CitationPanel slides into view on the top-right of the overlay,
  titled "Sources", listing each retrieved document filename (one per row) and
  a footer showing hit count and latency (e.g. "3 source(s) — 45 ms").

- [ ] **Hide button dismisses panel** — Click the "Hide" button in the panel header.
  Expected: Panel hides immediately. Avatar and text bubble are unaffected.

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
