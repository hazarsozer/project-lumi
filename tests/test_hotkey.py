"""
Unit tests for src.audio.hotkey — PTTListener and _to_pynput_hotkey.

pynput is NOT required to be installed; all tests mock it via sys.modules.
"""

from __future__ import annotations

import queue
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.audio.hotkey import PTTListener, _to_pynput_hotkey
from src.core.events import WakeDetectedEvent


# ---------------------------------------------------------------------------
# _to_pynput_hotkey
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_pynput_hotkey_ctrl_space() -> None:
    """ctrl+space → <ctrl>+<space>"""
    assert _to_pynput_hotkey("ctrl+space") == "<ctrl>+<space>"


@pytest.mark.unit
def test_to_pynput_hotkey_alt_shift_l() -> None:
    """alt+shift+l → <alt>+<shift>+l"""
    assert _to_pynput_hotkey("alt+shift+l") == "<alt>+<shift>+l"


@pytest.mark.unit
def test_to_pynput_hotkey_ctrl_a() -> None:
    """ctrl+a → <ctrl>+a (single char passes through)"""
    assert _to_pynput_hotkey("ctrl+a") == "<ctrl>+a"


@pytest.mark.unit
def test_to_pynput_hotkey_f5() -> None:
    """ctrl+f5 → <ctrl>+<f5>"""
    assert _to_pynput_hotkey("ctrl+f5") == "<ctrl>+<f5>"


@pytest.mark.unit
def test_to_pynput_hotkey_escape() -> None:
    """ctrl+escape → <ctrl>+<esc> (escape alias)"""
    assert _to_pynput_hotkey("ctrl+escape") == "<ctrl>+<esc>"


@pytest.mark.unit
def test_to_pynput_hotkey_case_insensitive() -> None:
    """Key names are normalised to lowercase."""
    assert _to_pynput_hotkey("Ctrl+Space") == "<ctrl>+<space>"


# ---------------------------------------------------------------------------
# PTTListener — start() with pynput available
# ---------------------------------------------------------------------------


def _make_mock_kb() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_pynput_package, mock_kb_module, mock_listener_instance).

    ``from pynput import keyboard`` resolves via attribute access on the pynput
    package object (``sys.modules["pynput"].keyboard``), so mock_pynput.keyboard
    must point at mock_kb — not just sys.modules["pynput.keyboard"].
    """
    mock_kb = MagicMock()
    mock_listener = MagicMock()
    mock_kb.GlobalHotKeys.return_value = mock_listener
    mock_pynput = MagicMock()
    mock_pynput.keyboard = mock_kb
    return mock_pynput, mock_kb, mock_listener


@pytest.mark.unit
def test_ptt_listener_start_creates_global_hotkeys() -> None:
    """start() instantiates GlobalHotKeys with the correct pynput-format hotkey."""
    mock_pynput, mock_kb, _ = _make_mock_kb()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        listener.start()

    mock_kb.GlobalHotKeys.assert_called_once()
    hotkey_dict: dict[str, Any] = mock_kb.GlobalHotKeys.call_args[0][0]
    assert "<ctrl>+<space>" in hotkey_dict


@pytest.mark.unit
def test_ptt_listener_start_calls_listener_start() -> None:
    """start() calls .start() on the underlying GlobalHotKeys listener."""
    mock_pynput, _, mock_listener = _make_mock_kb()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        listener.start()

    mock_listener.start.assert_called_once()


@pytest.mark.unit
def test_ptt_listener_is_active_after_start() -> None:
    """is_active returns True after a successful start()."""
    mock_pynput, _, _ = _make_mock_kb()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        assert not listener.is_active
        listener.start()

    assert listener.is_active


# ---------------------------------------------------------------------------
# PTTListener — pynput missing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ptt_listener_start_pynput_missing_is_noop(caplog: Any) -> None:
    """start() is a no-op (with warning) when pynput is not installed."""
    with patch.dict("sys.modules", {"pynput": None}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        listener.start()  # must not raise

    assert not listener.is_active
    assert any("pynput" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# PTTListener — on_activate callback posts WakeDetectedEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ptt_listener_activate_posts_wake_event() -> None:
    """The activate callback posts a WakeDetectedEvent to the event queue."""
    mock_kb = MagicMock()
    captured_callback: list[Any] = []

    def capture_hotkeys(hotkey_dict: dict[str, Any]) -> MagicMock:
        captured_callback.extend(hotkey_dict.values())
        return MagicMock()

    mock_kb.GlobalHotKeys.side_effect = capture_hotkeys

    mock_pynput = MagicMock()
    mock_pynput.keyboard = mock_kb

    event_queue: queue.Queue[Any] = queue.Queue()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(event_queue, hotkey="ctrl+space")
        listener.start()

    assert captured_callback, "no callback was registered"
    captured_callback[0]()  # simulate key press

    event = event_queue.get_nowait()
    assert isinstance(event, WakeDetectedEvent)
    assert event.timestamp <= time.monotonic()


# ---------------------------------------------------------------------------
# PTTListener — stop()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ptt_listener_stop_calls_listener_stop() -> None:
    """stop() calls .stop() on the underlying listener and clears it."""
    mock_pynput, _, mock_listener = _make_mock_kb()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        listener.start()
        assert listener.is_active
        listener.stop()

    mock_listener.stop.assert_called_once()
    assert not listener.is_active


@pytest.mark.unit
def test_ptt_listener_stop_before_start_is_safe() -> None:
    """stop() before start() must not raise."""
    listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
    listener.stop()  # must not raise
    assert not listener.is_active


@pytest.mark.unit
def test_ptt_listener_double_stop_is_safe() -> None:
    """stop() called twice must not raise."""
    mock_pynput, _, _ = _make_mock_kb()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        listener.start()
        listener.stop()
        listener.stop()  # must not raise


@pytest.mark.unit
def test_ptt_listener_start_is_idempotent() -> None:
    """start() called twice creates exactly one GlobalHotKeys listener."""
    mock_pynput, mock_kb, _ = _make_mock_kb()

    with patch.dict("sys.modules", {"pynput": mock_pynput}):
        listener = PTTListener(queue.Queue(), hotkey="ctrl+space")
        listener.start()
        listener.start()  # second call must be a no-op

    mock_kb.GlobalHotKeys.assert_called_once()  # not twice


@pytest.mark.unit
def test_to_pynput_hotkey_raises_on_empty_string() -> None:
    """_to_pynput_hotkey raises ValueError for empty input."""
    with pytest.raises(ValueError, match="must not be empty"):
        _to_pynput_hotkey("")


@pytest.mark.unit
def test_to_pynput_hotkey_raises_on_blank_string() -> None:
    """_to_pynput_hotkey raises ValueError for whitespace-only input."""
    with pytest.raises(ValueError, match="must not be empty"):
        _to_pynput_hotkey("   ")


@pytest.mark.unit
def test_to_pynput_hotkey_strips_empty_segments() -> None:
    """Extra '+' separators are stripped, not passed through as empty keys."""
    # "ctrl++space" has an empty segment between the two '+'
    result = _to_pynput_hotkey("ctrl++space")
    assert result == "<ctrl>+<space>"


@pytest.mark.unit
def test_ptt_listener_start_logs_warning_on_invalid_hotkey() -> None:
    """start() logs a warning and is a no-op when hotkey is invalid."""
    listener = PTTListener(queue.Queue(), hotkey="")
    listener.start()  # must not raise
    assert not listener.is_active
