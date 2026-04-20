# Lumi Avatar Sprite Specification

Art style decision: **static sprite sheets** (`AnimatedSprite2D`) — Option A.
Live2D / 3D VRM deferred indefinitely.

---

## Delivery format

- **PNG sprite sheets** — one sheet per animation group
- **Recommended frame size:** 256 × 256 px per frame (can scale; keep square)
- **Frames per sheet:** horizontal strip (frame 0 left → frame N right)
- **Colour space:** sRGB, transparent background (RGBA)
- **Naming:** match the animation name exactly (see below)

---

## Required animations (state)

These drive `AvatarController.on_state_change()`. All four are mandatory.

| File | Animation name in Godot | Description | Suggested frames |
|------|------------------------|-------------|-----------------|
| `idle_breathe.png` | `idle_breathe` | Gentle breathing / blinking loop | 8–16 |
| `listening_pulse.png` | `listening_pulse` | Attentive expression, ear/pulse indicator | 4–8 |
| `processing_spin.png` | `processing_spin` | Thinking expression, eye movement or spinner overlay | 6–12 |
| `speaking_lipsync.png` | `speaking_lipsync` | Base speaking pose (viseme overlays on top) | 4–8 |

---

## Optional animations (viseme mouth overlays)

These are played by `AvatarController.on_viseme()` during TTS speech.
Missing animations are silently skipped — ship state animations first.

| File | Animation name | Mouth shape |
|------|---------------|-------------|
| `mouth_rest.png` | `mouth_rest` | Closed / neutral |
| `mouth_open.png` | `mouth_open` | Open vowel (A, E) |
| `mouth_narrow.png` | `mouth_narrow` | Narrow (EE, IH) |
| `mouth_round.png` | `mouth_round` | Round (OH, OO) |
| `mouth_wide.png` | `mouth_wide` | Wide smile (AE) |
| `mouth_teeth.png` | `mouth_teeth` | Teeth visible (S, Z) |
| `mouth_tongue.png` | `mouth_tongue` | Tongue tip (TH, L) |
| `mouth_lips.png` | `mouth_lips` | Lip press (B, P, M) |

---

## How to import into Godot

1. Copy PNG files into `ui/assets/sprites/`.
2. Open `ui/scenes/avatar.tscn` in the Godot editor.
3. Select the `AnimatedSprite2D` node → open its `SpriteFrames` resource.
4. For each animation: **Add animation** → name it exactly as above → **Add frames from sprite sheet** → set the frame count to match the PNG strip.
5. Set **FPS** to 8–12 for state animations, 24 for mouth visemes.
6. No code changes needed — `avatar_controller.gd` picks up animations by name.
