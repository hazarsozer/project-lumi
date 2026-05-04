"""IPC bearer-token generation and verification for Project Lumi.

On each Brain startup a fresh 32-byte random token is written to
~/.lumi/ipc_token (chmod 0600). The Tauri frontend reads this file
and presents the token in its hello_ack frame. The HandshakeHandler
verifies it with a constant-time comparison.

This prevents arbitrary local processes from sending commands to the
Brain's WebSocket server and triggering OS tools (clipboard, screenshots,
app launches, etc.).
"""

from __future__ import annotations

import hmac
import logging
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_and_write_token(token_path: Path) -> str:
    """Generate a 64-char hex token, write to *token_path* (chmod 0600), return it."""
    token = secrets.token_hex(32)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    logger.info("IPC token written to %s", token_path)
    return token


def read_token(token_path: Path) -> str | None:
    """Read the token from *token_path*. Returns None if the file is missing or unreadable."""
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def verify_token(provided: str, expected: str) -> bool:
    """Constant-time token comparison (prevents timing oracles)."""
    return hmac.compare_digest(
        provided.encode("utf-8"),
        expected.encode("utf-8"),
    )
