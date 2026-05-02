"""
Unit tests for src.tools.os_actions — all subprocess and filesystem calls
are mocked via unittest.mock.patch; no real system tools are invoked.

Platform-specific paths are tested by patching ``src.tools.os_actions._get_platform``
to return "darwin" or "win32"; all subprocess calls are mocked so no actual
system commands run.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.os_actions import (
    AppLaunchTool,
    ClipboardTool,
    FileInfoTool,
    WindowListTool,
    _get_platform,
)


# ---------------------------------------------------------------------------
# AppLaunchTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_app_launch_success() -> None:
    """Popen is called with the resolved binary path; returns success."""
    with (
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/firefox"),
        patch("src.tools.os_actions.subprocess.Popen") as mock_popen,
    ):
        tool = AppLaunchTool()
        result = tool.execute({"app": "firefox"})

    assert result.success is True
    assert "firefox" in result.output.lower()
    mock_popen.assert_called_once_with(
        ["/usr/bin/firefox"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.unit
def test_app_launch_file_not_found_error() -> None:
    """FileNotFoundError from Popen → failure ToolResult with descriptive message."""
    with (
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/firefox"),
        patch(
            "src.tools.os_actions.subprocess.Popen",
            side_effect=FileNotFoundError("no such file"),
        ),
    ):
        tool = AppLaunchTool()
        result = tool.execute({"app": "firefox"})

    assert result.success is False
    assert "firefox" in result.output.lower()


@pytest.mark.unit
def test_app_launch_not_in_allowlist() -> None:
    """App names not in the allowlist are rejected before any subprocess call."""
    with patch("src.tools.os_actions.subprocess.Popen") as mock_popen:
        tool = AppLaunchTool()
        result = tool.execute({"app": "rm"})  # definitely not allowed

    assert result.success is False
    assert "not allowed" in result.output.lower()
    mock_popen.assert_not_called()


@pytest.mark.unit
def test_app_launch_macos_bundle_safari() -> None:
    """On darwin, 'safari' is launched via `open -a Safari` (not shutil.which)."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/open"),
        patch("src.tools.os_actions.subprocess.Popen") as mock_popen,
    ):
        tool = AppLaunchTool()
        result = tool.execute({"app": "safari"})

    assert result.success is True
    assert "safari" in result.output.lower()
    mock_popen.assert_called_once_with(
        ["/usr/bin/open", "-a", "Safari"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# ClipboardTool — read
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clipboard_read_success() -> None:
    """xclip stdout is returned as the output string."""
    mock_proc = MagicMock()
    mock_proc.stdout = "hello clipboard"
    mock_proc.returncode = 0

    with (
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/xclip"),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc),
    ):
        tool = ClipboardTool()
        result = tool.execute({"action": "read"})

    assert result.success is True
    assert result.output == "hello clipboard"


@pytest.mark.unit
def test_clipboard_read_file_not_found() -> None:
    """xclip not on PATH → failure ToolResult with install hint."""
    with patch("src.tools.os_actions.shutil.which", return_value=None):
        tool = ClipboardTool()
        result = tool.execute({"action": "read"})

    assert result.success is False
    assert "xclip" in result.output.lower()
    assert "install" in result.output.lower()


# ---------------------------------------------------------------------------
# ClipboardTool — write
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clipboard_write_success() -> None:
    """xclip is called with the text piped via stdin; returns success."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/xclip"),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc) as mock_run,
    ):
        tool = ClipboardTool()
        result = tool.execute({"action": "write", "text": "test content"})

    assert result.success is True
    assert "updated" in result.output.lower()
    # xclip should receive the text via input=
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs.get("input") == "test content"


@pytest.mark.unit
def test_clipboard_write_file_not_found() -> None:
    """xclip not on PATH → failure ToolResult with install hint."""
    with patch("src.tools.os_actions.shutil.which", return_value=None):
        tool = ClipboardTool()
        result = tool.execute({"action": "write", "text": "stuff"})

    assert result.success is False
    assert "xclip" in result.output.lower()
    assert "install" in result.output.lower()


# ---------------------------------------------------------------------------
# FileInfoTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_file_info_existing_file() -> None:
    """Stat on an existing regular file returns size, is_dir=False, exists=True."""
    fake_stat = MagicMock()
    fake_stat.st_size = 1234

    with patch("src.tools.os_actions.Path.stat", return_value=fake_stat), \
         patch("src.tools.os_actions.Path.is_dir", return_value=False):
        tool = FileInfoTool()
        result = tool.execute({"path": "/tmp/somefile.txt"})

    assert result.success is True
    assert result.data["exists"] is True
    assert result.data["is_dir"] is False
    assert result.data["size"] == 1234


@pytest.mark.unit
def test_file_info_path_traversal_rejected() -> None:
    """Paths containing '..' components are rejected without touching filesystem."""
    with patch("src.tools.os_actions.Path.stat") as mock_stat:
        tool = FileInfoTool()
        result = tool.execute({"path": "/tmp/../etc/passwd"})

    assert result.success is False
    assert "invalid path" in result.output.lower()
    mock_stat.assert_not_called()


@pytest.mark.unit
def test_file_info_non_existent_path() -> None:
    """FileNotFoundError from stat → returns exists=False with success=True."""
    with patch("src.tools.os_actions.Path.stat", side_effect=FileNotFoundError):
        tool = FileInfoTool()
        result = tool.execute({"path": "/tmp/definitely_does_not_exist_xyz.txt"})

    assert result.success is True
    assert result.data["exists"] is False


# ---------------------------------------------------------------------------
# WindowListTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_window_list_success() -> None:
    """wmctrl output is parsed into a window list."""
    wmctrl_output = (
        "0x00400003  0  hostname  Firefox\n"
        "0x00600001  0  hostname  Terminal\n"
    )
    mock_proc = MagicMock()
    mock_proc.stdout = wmctrl_output
    mock_proc.returncode = 0

    with (
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/wmctrl"),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc),
    ):
        tool = WindowListTool()
        result = tool.execute({})

    assert result.success is True
    assert result.data["windows"] == [
        {"id": "0x00400003", "desktop": "0", "host": "hostname", "title": "Firefox"},
        {"id": "0x00600001", "desktop": "0", "host": "hostname", "title": "Terminal"},
    ]
    assert "2 windows" in result.output


@pytest.mark.unit
def test_window_list_wmctrl_not_found() -> None:
    """wmctrl not on PATH → failure ToolResult with install hint."""
    with patch("src.tools.os_actions.shutil.which", return_value=None):
        tool = WindowListTool()
        result = tool.execute({})

    assert result.success is False
    assert "wmctrl" in result.output.lower()
    assert "install" in result.output.lower()


# ---------------------------------------------------------------------------
# ClipboardTool — macOS (pbpaste / pbcopy)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clipboard_read_darwin_success() -> None:
    """pbpaste output is returned as clipboard text on macOS."""
    mock_proc = MagicMock()
    mock_proc.stdout = "macOS clipboard"

    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/pbpaste"),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc),
    ):
        result = ClipboardTool().execute({"action": "read"})

    assert result.success is True
    assert result.output == "macOS clipboard"


@pytest.mark.unit
def test_clipboard_read_darwin_pbpaste_not_found() -> None:
    """pbpaste absent on macOS → failure with descriptive message."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value=None),
    ):
        result = ClipboardTool().execute({"action": "read"})

    assert result.success is False
    assert "pbpaste" in result.output.lower()


@pytest.mark.unit
def test_clipboard_write_darwin_success() -> None:
    """pbcopy is called with text via stdin on macOS."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/pbcopy"),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc) as mock_run,
    ):
        result = ClipboardTool().execute({"action": "write", "text": "hello mac"})

    assert result.success is True
    assert "updated" in result.output.lower()
    assert mock_run.call_args.kwargs.get("input") == "hello mac"


@pytest.mark.unit
def test_clipboard_write_darwin_pbcopy_not_found() -> None:
    """pbcopy absent on macOS → failure."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value=None),
    ):
        result = ClipboardTool().execute({"action": "write", "text": "hi"})

    assert result.success is False
    assert "pbcopy" in result.output.lower()


# ---------------------------------------------------------------------------
# ClipboardTool — Windows (pyperclip)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clipboard_read_win32_success() -> None:
    """pyperclip.paste() result is returned on Windows."""
    mock_pyperclip = MagicMock()
    mock_pyperclip.paste.return_value = "windows clipboard"

    with (
        patch("src.tools.os_actions._get_platform", return_value="win32"),
        patch.dict("sys.modules", {"pyperclip": mock_pyperclip}),
    ):
        result = ClipboardTool().execute({"action": "read"})

    assert result.success is True
    assert result.output == "windows clipboard"


@pytest.mark.unit
def test_clipboard_read_win32_pyperclip_missing() -> None:
    """ImportError for pyperclip → failure with install hint on Windows."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="win32"),
        patch.dict("sys.modules", {"pyperclip": None}),
    ):
        result = ClipboardTool().execute({"action": "read"})

    assert result.success is False
    assert "pyperclip" in result.output.lower()
    assert "install" in result.output.lower()


@pytest.mark.unit
def test_clipboard_write_win32_success() -> None:
    """pyperclip.copy() is called with text on Windows."""
    mock_pyperclip = MagicMock()

    with (
        patch("src.tools.os_actions._get_platform", return_value="win32"),
        patch.dict("sys.modules", {"pyperclip": mock_pyperclip}),
    ):
        result = ClipboardTool().execute({"action": "write", "text": "win text"})

    assert result.success is True
    mock_pyperclip.copy.assert_called_once_with("win text")


# ---------------------------------------------------------------------------
# ClipboardTool — Linux wl-clipboard fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clipboard_read_linux_wl_paste_fallback() -> None:
    """When xclip is absent but wl-paste is present, wl-paste is used."""
    mock_proc = MagicMock()
    mock_proc.stdout = "wayland clipboard"

    def which_side_effect(name: str) -> str | None:
        return "/usr/bin/wl-paste" if name == "wl-paste" else None

    with (
        patch("src.tools.os_actions._get_platform", return_value="linux"),
        patch("src.tools.os_actions.shutil.which", side_effect=which_side_effect),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc),
    ):
        result = ClipboardTool().execute({"action": "read"})

    assert result.success is True
    assert result.output == "wayland clipboard"


@pytest.mark.unit
def test_clipboard_write_linux_wl_copy_fallback() -> None:
    """When xclip absent but wl-copy present, wl-copy is used for write."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    def which_side_effect(name: str) -> str | None:
        return "/usr/bin/wl-copy" if name == "wl-copy" else None

    with (
        patch("src.tools.os_actions._get_platform", return_value="linux"),
        patch("src.tools.os_actions.shutil.which", side_effect=which_side_effect),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc),
    ):
        result = ClipboardTool().execute({"action": "write", "text": "wayland write"})

    assert result.success is True


@pytest.mark.unit
def test_clipboard_linux_no_tool_found() -> None:
    """Neither xclip nor wl-clipboard on PATH → failure with both mentioned."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="linux"),
        patch("src.tools.os_actions.shutil.which", return_value=None),
    ):
        result = ClipboardTool().execute({"action": "read"})

    assert result.success is False
    assert "xclip" in result.output.lower()
    assert "wl-clipboard" in result.output.lower()


# ---------------------------------------------------------------------------
# WindowListTool — macOS (osascript)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_window_list_darwin_success() -> None:
    """osascript output is parsed into window list on macOS."""
    mock_proc = MagicMock()
    mock_proc.stdout = "Firefox: New Tab\nTerminal: bash\n"

    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value="/usr/bin/osascript"),
        patch("src.tools.os_actions.subprocess.run", return_value=mock_proc),
    ):
        result = WindowListTool().execute({})

    assert result.success is True
    assert result.data["windows"] == [
        {"app": "Firefox", "title": "New Tab"},
        {"app": "Terminal", "title": "bash"},
    ]
    assert "2 windows" in result.output


@pytest.mark.unit
def test_window_list_darwin_osascript_not_found() -> None:
    """osascript absent on macOS → failure with descriptive message."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="darwin"),
        patch("src.tools.os_actions.shutil.which", return_value=None),
    ):
        result = WindowListTool().execute({})

    assert result.success is False
    assert "osascript" in result.output.lower()


# ---------------------------------------------------------------------------
# WindowListTool — Windows (pygetwindow)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_window_list_win32_success() -> None:
    """pygetwindow.getAllTitles() result is returned on Windows."""
    mock_pgw = MagicMock()
    mock_pgw.getAllTitles.return_value = ["Notepad", "Firefox", ""]

    with (
        patch("src.tools.os_actions._get_platform", return_value="win32"),
        patch.dict("sys.modules", {"pygetwindow": mock_pgw}),
    ):
        result = WindowListTool().execute({})

    assert result.success is True
    assert result.data["windows"] == [{"title": "Notepad"}, {"title": "Firefox"}]
    assert "2 windows" in result.output


@pytest.mark.unit
def test_window_list_win32_pygetwindow_missing() -> None:
    """ImportError for pygetwindow → failure with install hint on Windows."""
    with (
        patch("src.tools.os_actions._get_platform", return_value="win32"),
        patch.dict("sys.modules", {"pygetwindow": None}),
    ):
        result = WindowListTool().execute({})

    assert result.success is False
    assert "pygetwindow" in result.output.lower()
    assert "install" in result.output.lower()
