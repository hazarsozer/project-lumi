"""
Tests for ws_bridge.py — superseded by B3 (drop ws_bridge).

ws_bridge.py was removed in B3: the Brain process now serves WebSocket
clients directly via WSTransport (src/core/ws_transport.py).  These
tests are replaced by the integration tests in
tests/integration/test_ipc_full_turn.py which exercise the full
WSTransport → EventBridge stack.

This file is kept as a placeholder and will be deleted in the Ring 3
repo-hygiene pass (I7).
"""
