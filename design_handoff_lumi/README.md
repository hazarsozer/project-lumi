# Handoff: Lumi — Desktop AI Assistant Overlay

## Overview
Lumi is a desktop AI assistant that runs as an always-on-top transparent overlay. The UI consists of three surfaces: a compact character overlay anchored to the screen edge, a floating chat panel, and a floating settings panel. It is a desktop application (Electron or Tauri) — not a web page.

## About the Design Files
The files in this bundle (`Lumi Design.html`, `lumi-components.jsx`, `lumi-surfaces.jsx`) are **design references built in HTML/React** — they demonstrate intended look, layout, and behavior. They are **not production code to ship directly**. Your task is to recreate these designs in your actual application environment (Electron + React, Tauri + React/Svelte, or equivalent) using its established patterns and libraries.

The HTML files can be opened in any browser to interact with the design. Use the **Tweaks** panel (bottom-right toggle) to cycle the avatar through its four states.

## Fidelity
**High-fidelity.** Colors, typography, spacing, border radii, shadow values, and interactions are all final. Recreate pixel-precisely using the design tokens in this document.

---

## Design Tokens

### Colors (all in oklch)
```json
{
  "background":        "oklch(11% 0.018 245)",
  "surface":           "oklch(15.5% 0.022 245)",
  "surfaceElevated":   "oklch(19% 0.026 245)",
  "surfaceTop":        "oklch(23% 0.03 245)",
  "border":            "oklch(28% 0.03 245)",
  "borderSubtle":      "oklch(22% 0.025 245)",
  "textPrimary":       "oklch(91% 0.01 240)",
  "textSecondary":     "oklch(58% 0.025 240)",
  "textMuted":         "oklch(36% 0.022 240)",
  "accentIdle":        "oklch(62% 0.18 222)",
  "accentListening":   "oklch(64% 0.18 152)",
  "accentProcessing":  "oklch(72% 0.17 65)",
  "accentSpeaking":    "oklch(94% 0.012 240)",
  "danger":            "oklch(60% 0.18 25)"
}
```

### Spacing Scale (px)
| Token | Value |
|-------|-------|
| xs    | 4     |
| sm    | 8     |
| md    | 12    |
| lg    | 16    |
| xl    | 24    |
| xxl   | 32    |

### Border Radius (px)
| Token | Value |
|-------|-------|
| sm    | 4     |
| md    | 8     |
| lg    | 12    |
| xl    | 16    |
| pill  | 9999  |

### Font Size (px)
| Token | Value |
|-------|-------|
| xs    | 10    |
| sm    | 11    |
| md    | 13    |
| lg    | 15    |
| xl    | 18    |
| xxl   | 22    |

**Font family:** `'Manrope', system-ui, sans-serif`  
**Weights used:** 400 (body), 500 (label), 600 (heading), 700 (display)

### Opacity
| Token    | Value |
|----------|-------|
| idle     | 0.42  |
| active   | 1.0   |
| disabled | 0.32  |

### Shadows
```
sm:        0 2px 8px oklch(0% 0 0 / 0.45)
md:        0 4px 20px oklch(0% 0 0 / 0.55)
lg:        0 8px 36px oklch(0% 0 0 / 0.65)
glowBlue:  0 0 18px oklch(62% 0.18 222 / 0.45)
glowGreen: 0 0 18px oklch(64% 0.18 152 / 0.50)
glowAmber: 0 0 18px oklch(72% 0.17 65 / 0.45)
glowWhite: 0 0 22px oklch(94% 0.012 240 / 0.55)
```

---

## Avatar States
The assistant has four states, each changing the accent color, glow, and opacity of the character:

| State      | Color token       | Opacity | Behavior |
|------------|-------------------|---------|----------|
| idle       | accentIdle (blue) | 0.42    | Slow ambient glow pulse (2.8s cycle) |
| listening  | accentListening (green) | 1.0 | Outer ring pulses (1.1s loop); character floats up/down (1.4s ease-in-out loop, 5px travel) |
| processing | accentAmber       | 1.0     | Steady glow, no motion |
| speaking   | accentSpeaking (near-white) | 1.0 | Full brightness |

---

## Surface 1 — Compact Overlay

**Window size:** ~160px wide. No fixed height — height is determined by content.  
**Window type:** Always-on-top, transparent background, no frame/chrome. Anchored to bottom-left of screen.

### Layout
Two layers stacked vertically with overlap:

```
┌─────────────────────┐
│   Character art      │  ← transparent bg, ~140×210px
│   (overlaps tray     │
│    by 28px)          │
│                      │
└──────────────────────┘
        ↕ -28px overlap
┌────────────────────────────┐
│ ● Idle  │  ⚙   │  💬      │  ← button tray pill
└────────────────────────────┘
```

### Character Area
- **Size:** 140×210px
- **Background:** fully transparent (no fill)
- **Content:** Character PNG/WebP with alpha transparency — drop-in replacement for the SVG placeholder in the reference file
- **Opacity:** 0.42 when idle, 1.0 otherwise
- **Transition:** `opacity 0.5s ease`
- **Glow effect:** CSS `filter: drop-shadow(0 0 14px <accentColor>)` at active; `drop-shadow(0 0 6px <accentColor>80)` at idle
- **Listening animation:** `translateY` tween, 0→-5px→0, 1.4s ease-in-out, infinite
- **z-index:** above the button tray (z-index: 2 vs 1)
- **Margin-bottom:** -28px to create the overlap

### Button Tray
- **Shape:** pill (`border-radius: 9999px`)
- **Background:** `oklch(14% 0.022 245 / 0.95)` with `backdrop-filter: blur(18px)`
- **Border:** `1px solid oklch(30% 0.035 245 / 0.7)`
- **Shadow:** `md` shadow + `inset 0 1px 0 oklch(100% 0 0 / 0.05)`
- **Padding:** 12px 16px
- **Gap between items:** 8px
- **Content (left to right):**
  1. Status dot — 6×6px circle, background = current accent color, box-shadow = current glow
  2. State label — font-size 11px, color textSecondary, letter-spacing 0.02em, margin-right 8px
  3. Divider — 1px wide × 18px tall, color borderSubtle
  4. Settings button ⚙ — 34×34px, border-radius 8px
  5. Chat button 💬 — 34×34px, border-radius 8px

### Icon Button States
|           | Background    | Border        |
|-----------|---------------|---------------|
| Normal    | surfaceElevated | borderSubtle |
| Hover     | surfaceTop    | border        |
| Active    | surfaceTop    | border        |
| Disabled  | transparent   | transparent, opacity 0.32 |

---

## Surface 2 — Chat Panel

**Window size:** 380×540px  
**Window type:** Floating, draggable, closeable. Appears when 💬 is clicked.

### Layout (flex column)
```
┌──────────────────────────────┐
│ Header (avatar + name + ×)   │  48px, border-bottom
├──────────────────────────────┤
│                              │
│  Message list (scrollable)   │  flex: 1
│                              │
├──────────────────────────────┤
│ Input row                    │  ~56px, border-top
└──────────────────────────────┘
```

### Window Chrome
- Background: `surface`
- Border-radius: 16px
- Border: `1px solid border`
- Shadow: `lg`

### Header
- Padding: 12px 16px
- Border-bottom: `1px solid borderSubtle`
- Left: 28×28px circle avatar (background surfaceTop, border 1.5px accentIdle, box-shadow glowBlue, content: ✦ glyph at 12px)
- Next to avatar: name "Lumi" (15px, weight 600, textPrimary) + online indicator below ("● Online", 10px, accentGreen)
- Right: close button ×, 26×26px, border-radius 8px, background surfaceTop, border border, color textSecondary

### Message List
- Padding: 16px
- Gap between bubbles: 16px
- Overflow-y: scroll

#### Lumi bubble (left-aligned)
- Layout: row, gap 10px, align items to flex-end
- Avatar: 28×28px circle, surfaceTop bg, border, ✦ glyph
- Bubble: max-width 80%, padding 8×12px
  - Background: surfaceElevated
  - Border: `1px solid borderSubtle`
  - Border-radius: `12px 12px 12px 4px`
  - Font: 13px, textPrimary, line-height 1.55
- Citation pills (below text, gap 5px):
  - Font: 10px, accentBlue
  - Background: `oklch(62% 0.18 222 / 0.12)`
  - Border: `1px solid oklch(62% 0.18 222 / 0.25)`
  - Border-radius: pill
  - Padding: 2px 7px
- Timestamp: 10px, textMuted, margin-top 3px, left-aligned

#### User bubble (right-aligned)
- Layout: row-reverse, same gap
- No avatar
- Bubble:
  - Background: `oklch(62% 0.18 222 / 0.18)`
  - Border: `1px solid oklch(62% 0.18 222 / 0.3)`
  - Border-radius: `12px 12px 4px 12px`
- Timestamp: right-aligned

#### Typing indicator
- Same layout as Lumi bubble
- Bubble contains 3 dots (5×5px circles, textMuted)
- Each dot animates: `translateY(0) → translateY(-4px) → translateY(0)`, 1.2s ease-in-out, staggered by 0.2s each

### Input Row
- Padding: 12px
- Inner container: flex row, gap 8px, background surfaceTop, border-radius 12px, border, padding 8×12px
- Placeholder text input (flex: 1): transparent bg, no border, 13px, textSecondary
- Mic button: 28×28px, fontSize 14, color textMuted, emoji 🎙
- Send button: 28×28px, border-radius 8px, background accentBlue, contains right-arrow SVG in white

---

## Surface 3 — Settings Panel

**Window size:** 680×540px  
**Window type:** Floating, draggable, closeable.

### Layout
```
┌──────────────────────────────────────────┐
│ Header ("Settings" + ×)                  │  44px
├────────────┬─────────────────────────────┤
│            │                             │
│  Sidebar   │   Tab content (scrollable)  │
│  (140px)   │   (flex: 1)                 │
│            │                             │
├────────────┴─────────────────────────────┤
│ Footer (↻ note + Cancel / Apply / Save)  │  52px
└──────────────────────────────────────────┘
```

### Window Chrome
- Same as Chat Panel: surface bg, radius 16px, border, lg shadow

### Header
- Padding: 12px 16px
- "Settings" label: 15px, weight 600, textPrimary
- Close button ×: same as Chat Panel

### Sidebar
- Width: 140px
- Border-right: `1px solid borderSubtle`
- Padding: 8px (all sides)
- Gap between tabs: 2px
- **7 tabs:** General, Voice, Model, Context, Privacy, Appearance, Advanced

#### Tab item
- Padding: 8px 12px, border-radius 8px
- Font: 13px
- **Inactive:** background transparent, border transparent, color textSecondary
- **Active:** background surfaceTop, border `1px solid border`, color textPrimary
- Transition: `background 0.15s`

### Tab Content
- Padding: 8px 0 (top/bottom only — rows handle left/right padding internally)
- Overflow-y: scroll

#### Setting Row
- Padding: 8px 12px, border-radius 8px
- Layout: flex row, space-between, gap 16px, align-items center
- Left side (flex:1): label (13px, textPrimary) + optional description below (11px, textMuted, margin-top 2px)
- Restart badge: shown inline after label — `↻`, 10px, accentAmber, background `oklch(72% 0.17 65 / 0.15)`, padding 1px 5px, border-radius pill
- Right side: control (see control specs below)

### Footer
- Padding: 12px 16px
- Border-top: `1px solid borderSubtle`
- Left: "↻ Requires restart" in 11px, textMuted
- Right: three buttons with gap 8px between:

| Button | Background | Border | Color | Weight |
|--------|-----------|--------|-------|--------|
| Cancel | transparent | border | textSecondary | 400 |
| Apply  | surfaceTop  | border | textSecondary | 400 |
| Save   | accentBlue  | none   | white | 600, box-shadow glowBlue |

Button padding: 7px 16px, border-radius 8px, font 13px

---

## Controls — 7 Types

### 1. Toggle
- Track: 36×20px, border-radius pill
- On background: accentBlue; Off background: surfaceTop
- Thumb: 14×14px circle, top/bottom offset 3px
- On: left 19px, background textPrimary, box-shadow glowBlue; Off: left 3px, background textMuted
- Transition: `background 0.22s, left 0.22s`
- Disabled: opacity 0.32, cursor not-allowed

### 2. Slider
- Track height: 4px, border-radius pill, background surfaceTop
- Fill: left-anchored, background accentBlue
- Thumb: 12×12px circle, background textPrimary, border `2px solid accentBlue`, cursor ew-resize
- Optional min/max labels: 10px, textMuted, below track
- Value readout: 11px, accentBlue, right-aligned above track

### 3. Dropdown
- Padding: 7px 10px, border-radius 8px
- Background: surfaceTop, border: border
- Value text: 13px, textPrimary
- Chevron: 10×6px SVG arrow, stroke textMuted
- Min-width: 160px

### 4. Text Input
- Padding: 7px 10px, border-radius 8px
- Background: surfaceTop, border: border
- Font: 13px, textPrimary (textMuted when disabled)
- Placeholder color: textMuted
- Optional suffix: 11px, textMuted

### 5. Number Input
- Width: 90px
- Left section: centered number value, 13px
- Right section: + and − buttons stacked, separated by border, 7px horizontal padding
- Border between sections: borderSubtle

### 6. File Path Input
- Two-part: path field (flex:1, monospace 11px, textSecondary) + Browse button
- Path field border-radius: `8px 0 0 8px`
- Browse button: surfaceElevated bg, border-radius `0 8px 8px 0`, 11px, textSecondary

### 7. Multi-select Checkbox
- Checkbox box: 15×15px, border-radius 4px
- Checked: background + border = accentBlue, contains white checkmark SVG (9×7px)
- Unchecked: transparent background, border = border color
- Label: 13px, textPrimary (textMuted when disabled)
- Transition: `background 0.15s`

---

## Animations

All animations are simple CSS tweens — no springs or physics.

| Name | Property | From | To | Duration | Easing | Trigger |
|------|----------|------|----|----------|--------|---------|
| fadeIn | opacity + translateY | 0, 6px | 1, 0 | 280ms | ease | panel mount |
| avatarPulse | opacity + scale | 0.55, 1 | 0.9, 1.06 | 1100ms | ease-in-out | listening state, infinite |
| idlePulse | outer ring opacity | 0.25 | 0.5 | 2800ms | ease-in-out | idle state, infinite |
| float | translateY | 0 | -5px | 1400ms | ease-in-out | listening state, infinite |
| dotBounce | translateY + opacity | 0, 0.4 | -4px, 1 | 1200ms | ease-in-out | typing indicator, staggered 0.2s |

---

## State Management

```ts
type AvatarState = 'idle' | 'listening' | 'processing' | 'speaking';

interface AppState {
  avatarState: AvatarState;
  chatOpen: boolean;
  settingsOpen: boolean;
  activeSettingsTab: number; // 0–6
  messages: Message[];
  inputDraft: string;
}

interface Message {
  id: string;
  from: 'lumi' | 'user';
  text: string;
  timestamp: Date;
  citations?: string[]; // e.g. ['notes.md', 'calendar.json']
  isTyping?: boolean;   // true while Lumi is generating
}
```

Avatar state transitions:
- `idle` → `listening`: wake word detected or mic button pressed
- `listening` → `processing`: user stops speaking
- `processing` → `speaking`: response ready, TTS begins
- `speaking` → `idle`: TTS ends

---

## Assets
- **Character art:** Not included. Drop in a transparent PNG/WebP. The design is built to accept any character — anime, cartoon, 3D render, etc. Recommended size: at least 280×420px @2x. The `filter: drop-shadow(...)` glow applies automatically via CSS.
- **✦ glyph:** Unicode U+2736 (used as Lumi's avatar icon in chat; fallback to any star/sparkle glyph)
- **Icons:** Settings ⚙ (U+2699), Chat 💬 (U+1F4AC) — replace with icon library equivalents if preferred (SF Symbols, Lucide, etc.)

---

## Files in This Package
| File | Purpose |
|------|---------|
| `Lumi Design.html` | Main interactive design reference — open in browser |
| `lumi-components.jsx` | All 7 control components + shared tokens |
| `lumi-surfaces.jsx` | CompactOverlay, ChatPanel, SettingsPanel, LumiAvatar |
| `README.md` | This document |

Open `Lumi Design.html` in Chrome or Safari to interact with the full design. Use the **Tweaks** panel (bottom-right) to toggle avatar states.
