# Phase 9 Scope — UI Design Overhaul

**Decision date:** 2026-04-23
**Status:** In progress — awaiting Claude Design mockups

---

## Core Decision: Overlay Architecture

**Chosen: Compact overlay + separate floating panels.**

Lumi renders as a small always-on-top widget in the **bottom-left corner** of the screen.
No full-window mode. The overlay is never a "normal app window."

### Overlay layout (resting state, ~180×220 px)

```
┌──────────────┐
│   Avatar     │  ← semi-transparent (40% opacity) when idle
│   (◉◡◉)     │     full opacity (100%) when speaking
│              │     ambient glow colour changes by state:
│              │       idle=soft blue, listening=green pulse,
│              │       processing=amber, speaking=white glow
│  [⚙]  [💬]  │  ← icon buttons: settings, chat
└──────────────┘
```

### Floating panels (opened from buttons, independently moveable)

| Trigger | Panel | Size |
|---------|-------|------|
| Click ⚙ | Settings Panel | ~720×560 |
| Click 💬 | Chat Panel | ~400×560 |

Both panels are independently moveable and closeable (× button top-right).
They float free — the compact overlay always stays visible behind them.

---

## What is NOT in the Overlay

- **No on-screen text bubble during voice responses.** LLM response text is not
  displayed outside the Chat panel. Voice is the primary output channel; text is
  available on demand in the Chat panel.
- **No citation overlay.** Citations appear inline inside the Chat panel messages
  (as small pill tags, e.g. "from notes.md"), not as a separate floating element.

---

## Surfaces to Design (for Claude Design brief)

| Surface | Notes |
|---------|-------|
| **Compact Overlay** | Avatar + 2 icon buttons, ~180×220 |
| **Chat Panel** | Session history (voice + text), text input, inline citations |
| **Settings Panel** | 7 tabs, 47 fields, 7 control types |

The `text_bubble.tscn` and `citation_panel.tscn` (previously standalone) are
**absorbed into the Chat Panel**. They become dead code once the Chat Panel is built.

---

## Existing Godot Components — Phase 9 Disposition

| Component | Disposition |
|-----------|-------------|
| `settings_panel.gd/.tscn` | Keep — restyle with new theme |
| `setting_row.gd/.tscn` | Keep — restyle with new theme |
| `lumi_client.gd` | Keep — no changes needed |
| `main.gd` | Refactor — remove TextBubble routing, add ChatPanel routing |
| `main.tscn` | Refactor — remove TextBubble node, add ChatPanel |
| `text_bubble.gd/.tscn` | Remove — replaced by Chat Panel |
| `citation_panel.gd/.tscn` | Remove — absorbed into Chat Panel |
| `avatar_controller.gd` | Keep — wire opacity + glow to state events |
| `rag_toggle.gd` | Defer — RAG toggle moves into Settings Panel tab |

---

## Wayland Transparency

Deferred to Phase 11. For Phase 9, the overlay uses an opaque dark background
with a simulated drop shadow inside the texture. X11 users get real per-pixel
alpha; Wayland/XWayland users get the opaque fallback. Both look correct.

---

## GPU Status (checked 2026-04-23)

**RTX 4070 — 12 GB VRAM, 11.5 GB free.**

Wave H3–H6 (QLoRA fine-tune pipeline) is **unblocked**. Minimum requirement was 8 GB.
This runs in parallel as Track C during Phase 9.

---

## Phase 9 Tracks

| Track | Content | Serial/Parallel |
|-------|---------|-----------------|
| **A** | Commit Phase 8.5, GPU check, smoke test, this doc | Serial — done first |
| **B** | Claude Design → tokens → Godot theme → restyle 3 surfaces | Parallel |
| **C** | QLoRA dataset + train skeleton + eval harness | Parallel |

---

## Definition of Done

- [ ] Phase 8.5 PR merged to `main`
- [ ] End-to-end voice smoke test passes (script at `scripts/smoke_test_voice.py`)
- [ ] `ui/themes/lumi_dark.theme` exists
- [ ] `ui/themes/design_tokens.json` committed (Claude Design export)
- [ ] Compact overlay resized to ~180×220, anchored bottom-left
- [ ] Chat Panel scene + script implemented
- [ ] Settings Panel restyled with new theme
- [ ] `text_bubble.tscn/.gd` and `citation_panel.tscn/.gd` removed
- [ ] Avatar opacity + glow wired to 4 state events
- [ ] All 4 avatar states have visual treatment
- [ ] Python tests still 896+ passing, ≥80% coverage
- [ ] `docs/phase9_screenshots/` before/after screenshots committed
- [ ] `ARCHITECTURE.md` Section 4 updated (window size, theme path, Wayland note)
