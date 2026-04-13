# Lumi — Godot Frontend (Phase 5)

Transparent 200x200 overlay window. Connects to the Python Brain via raw TCP (127.0.0.1:5555).

## Requirements

- Godot 4.x — download from https://godotengine.org
- Python Brain running first (see main README.md)

## Running

1. Open `ui/project.godot` in the Godot 4 editor.
2. Press F5 (or Run -> Play) to launch the overlay.
3. The client auto-connects and retries every 2 seconds if the Brain is not yet running.

## Project structure

- `scripts/ipc_protocol.gd` — length-prefixed JSON frame encoding/decoding
- `scripts/lumi_client.gd` — StreamPeerTCP client with auto-reconnect
- `scripts/avatar_controller.gd` — drives AnimatedSprite2D from Brain state events
- `scripts/main.gd` — root scene: wires client signals, handles Escape -> interrupt

## Placeholder sprites

Phase 5 uses colored circles in `assets/sprites/`. Replace with real artwork in Phase 6.

## Compositor (Linux)

Transparent windows require a compositor:
- X11: install Picom (`sudo apt install picom`) and run `picom &`
- Wayland: transparency is natively supported by most compositors (Sway, Mutter, KWin)
