"""
Push-to-talk global hotkey listener for Project Lumi.

Posts WakeDetectedEvent to the event queue whenever the configured hotkey
is pressed.  This allows Lumi to be activated without a custom wake-word
model — the hotkey is the primary entry point; wake-word is opt-in.

Requires pynput.  Install with: uv sync --extra ptt (or: pip install pynput)
If pynput is not installed, start() logs a warning and becomes a no-op so
the rest of the app continues without PTT.

Usage
─────
    from src.audio.hotkey import PTTListener
    listener = PTTListener(event_queue, hotkey="ctrl+space")
    listener.start()   # non-blocking; runs pynput's GlobalHotKeys thread
    ...
    listener.stop()

Hotkey format
─────────────
Plain English key names joined by "+":
    "ctrl+space"        "alt+shift+l"        "ctrl+alt+p"
Recognised modifier names: ctrl, alt, shift, cmd, super
Recognised special keys:   space, tab, enter, esc, f1–f12
Single printable characters are passed through as-is: "ctrl+a"
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

from src.core.events import WakeDetectedEvent

logger = logging.getLogger(__name__)

# Map plain key names → pynput GlobalHotKeys bracket notation.
_PYNPUT_KEY_MAP: dict[str, str] = {
    "ctrl": "<ctrl>",
    "alt": "<alt>",
    "shift": "<shift>",
    "cmd": "<cmd>",
    "super": "<super>",
    "space": "<space>",
    "tab": "<tab>",
    "enter": "<enter>",
    "return": "<enter>",
    "esc": "<esc>",
    "escape": "<esc>",
    **{f"f{n}": f"<f{n}>" for n in range(1, 13)},
}


def _to_pynput_hotkey(hotkey: str) -> str:
    """Convert plain-text hotkey string to pynput GlobalHotKeys format.

    Examples:
        "ctrl+space"   → "<ctrl>+<space>"
        "alt+shift+l"  → "<alt>+<shift>+l"
        "ctrl+a"       → "<ctrl>+a"

    Raises:
        ValueError: if ``hotkey`` is empty or produces no valid key parts.
    """
    if not hotkey or not hotkey.strip():
        raise ValueError(f"PTTListener: hotkey must not be empty, got {hotkey!r}")
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"PTTListener: no valid key segments in hotkey {hotkey!r}")
    return "+".join(_PYNPUT_KEY_MAP.get(p, p) for p in parts)


class PTTListener:
    """Global hotkey listener that posts WakeDetectedEvent on hotkey press.

    Args:
        event_queue: The orchestrator's central event queue.
        hotkey:      Plain-text hotkey string (default: "ctrl+space").
    """

    def __init__(
        self,
        event_queue: queue.Queue[Any],
        hotkey: str = "ctrl+space",
    ) -> None:
        self._event_queue = event_queue
        self._hotkey = hotkey
        self._listener: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the global hotkey listener on a background thread.

        Idempotent: if already running, logs a debug message and returns.
        No-op (with a warning) if pynput is not installed.
        """
        if self._listener is not None:
            logger.debug("PTTListener.start() called while already active; ignoring.")
            return

        try:
            pynput_key = _to_pynput_hotkey(self._hotkey)
        except ValueError as exc:
            logger.warning("PTTListener: invalid hotkey %r: %s", self._hotkey, exc)
            return

        try:
            from pynput import keyboard as kb  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "PTTListener: pynput not installed — push-to-talk disabled. "
                "Enable with: uv sync --extra ptt"
            )
            return

        def _on_activate() -> None:
            logger.info(
                "PTT hotkey %r activated — posting WakeDetectedEvent", self._hotkey
            )
            self._event_queue.put(WakeDetectedEvent(timestamp=time.monotonic()))

        try:
            self._listener = kb.GlobalHotKeys({pynput_key: _on_activate})
            self._listener.start()
            logger.info(
                "PTTListener: active — press %r to activate Lumi", self._hotkey
            )
        except Exception as exc:
            logger.warning("PTTListener: failed to start hotkey listener: %s", exc)
            self._listener = None

    def stop(self) -> None:
        """Stop the hotkey listener. Safe to call even if never started."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:
                logger.debug("PTTListener: error during stop: %s", exc)
            self._listener = None
        logger.info("PTTListener: stopped.")

    @property
    def is_active(self) -> bool:
        """True if the listener thread is currently running."""
        return self._listener is not None
