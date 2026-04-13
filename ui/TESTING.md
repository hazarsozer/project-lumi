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
