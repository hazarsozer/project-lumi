"""
ws_bridge.py — DEPRECATED. Removed in B3.

The three-process architecture (Brain TCP + ws_bridge + Tauri) has been
replaced by a two-process architecture: the Brain serves WebSocket clients
directly via WSTransport (src/core/ws_transport.py).

This file is kept as a stub to avoid stale import errors during the
transition.  It will be deleted in the Ring 3 repo-hygiene pass (I7).
"""
