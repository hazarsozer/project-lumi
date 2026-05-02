"""
OS action tools for Project Lumi.

Concrete Tool implementations for common desktop operations:
- AppLaunchTool  (name="launch_app")  — launch an application via Popen
- ClipboardTool  (name="clipboard")   — read/write system clipboard
- FileInfoTool   (name="file_info")   — stat a file path safely
- WindowListTool (name="window_list") — list open windows

Platform dispatch
─────────────────
ClipboardTool and WindowListTool select a backend at call time based on
the value returned by _get_platform().  Adapters:

  Linux   — xclip (X11) or wl-clipboard (Wayland) / wmctrl
  macOS   — pbpaste + pbcopy (built-in) / osascript
  Windows — pyperclip (optional) / pygetwindow (optional)

AppLaunchTool and FileInfoTool are already cross-platform; no dispatch
is required.

Security decisions
──────────────────
- AppLaunchTool: app_name is NEVER passed directly to subprocess; only the
  resolved binary path of a known-safe executable is passed, verified via
  shutil.which().
- FileInfoTool: paths containing ".." components are rejected before any
  filesystem operation.
- No shell=True anywhere — all subprocess calls use explicit argument lists.
- All inputs are validated before any subprocess or filesystem operation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.tools.base import ToolResult

logger = logging.getLogger(__name__)


def _get_platform() -> str:
    """Return sys.platform. Isolated so tests can patch it without side-effects."""
    return sys.platform


# macOS GUI apps live in .app bundles and are NOT on $PATH.
# shutil.which("safari") returns None. Use `open -a BundleName` instead.
# Maps plain name (in ALLOWED_APPS) → the .app bundle name for `open -a`.
_MACOS_BUNDLE_APPS: dict[str, str] = {
    "safari":   "Safari",
    "terminal": "Terminal",
    "textedit": "TextEdit",
    "finder":   "Finder",
    "iterm2":   "iTerm2",
}


# ---------------------------------------------------------------------------
# AppLaunchTool
# ---------------------------------------------------------------------------


class AppLaunchTool:
    """Launch a desktop application by name.

    Schema:
        name: "launch_app"
        args:
            app (str, required): Application name to launch.
                                 Must be in ALLOWED_APPS.

    Returns:
        ToolResult(success=True, output="Launched: <app>", data={})
        ToolResult(success=False, output="App not found: <app>", data={}) if not on PATH
        ToolResult(success=False, output="App not allowed: <app>", data={}) if not in allowlist
    """

    name: str = "launch_app"
    description: str = (
        "Launch a desktop application by name. "
        'Args: {"app": "<name>"}. '
        "Common apps: firefox, terminal, file manager, text editor, vlc."
    )

    # Cross-platform allowlist — the only apps this tool will ever launch.
    # Security rationale: user-controlled strings are NEVER passed directly
    # to subprocess; we look up the canonical binary of the allow-listed name.
    ALLOWED_APPS: frozenset[str] = frozenset(
        {
            # Linux
            "firefox",
            "thunar",
            "gedit",
            "gnome-terminal",
            "konsole",
            "nautilus",
            "libreoffice",
            "vlc",
            "xterm",
            "mousepad",
            # macOS (command-line binaries on PATH)
            "open",
            "safari",
            "terminal",
            "textedit",
            "finder",
            "iterm2",
            "code",
            # Windows (command-line binaries on PATH)
            "notepad",
            "explorer",
            "calc",
            "powershell",
            "cmd",
            "mspaint",
        }
    )

    def execute(self, args: dict[str, Any]) -> ToolResult:
        app: Any = args.get("app", "")

        if not isinstance(app, str) or not app:
            logger.warning("AppLaunchTool: missing or non-string 'app' arg.")
            return ToolResult(
                success=False, output="Missing required arg: app", data={}
            )

        app = app.strip()

        if app not in self.ALLOWED_APPS:
            logger.warning(
                "AppLaunchTool: app '%s' is not in allowlist; rejected.", app
            )
            return ToolResult(success=False, output=f"App not allowed: {app}", data={})

        # macOS GUI apps live in .app bundles — launch via `open -a` instead of
        # PATH lookup, which would return None for every bundle-based app.
        if _get_platform() == "darwin" and app in _MACOS_BUNDLE_APPS:
            return self._launch_macos_bundle(app, _MACOS_BUNDLE_APPS[app])

        binary = shutil.which(app)
        if binary is None:
            logger.warning("AppLaunchTool: app '%s' not found on PATH.", app)
            return ToolResult(success=False, output=f"App not found: {app}", data={})

        try:
            subprocess.Popen(  # noqa: S603 — explicit list, no shell
                [binary],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning("AppLaunchTool: FileNotFoundError launching '%s'.", app)
            return ToolResult(success=False, output=f"App not found: {app}", data={})
        except OSError as exc:
            logger.error("AppLaunchTool: OSError launching '%s': %s.", app, exc)
            return ToolResult(
                success=False, output=f"Failed to launch {app}: {exc}", data={}
            )

        logger.info("AppLaunchTool: launched '%s' (%s).", app, binary)
        return ToolResult(success=True, output=f"Launched: {app}", data={})

    @staticmethod
    def _launch_macos_bundle(app: str, bundle_name: str) -> ToolResult:
        """Launch a macOS .app bundle via `open -a BundleName`."""
        open_bin = shutil.which("open")
        if open_bin is None:
            return ToolResult(
                success=False, output="'open' not found (expected on macOS)", data={}
            )
        try:
            subprocess.Popen(  # noqa: S603
                [open_bin, "-a", bundle_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.error(
                "AppLaunchTool: OSError launching macOS bundle '%s': %s", bundle_name, exc
            )
            return ToolResult(
                success=False, output=f"Failed to launch {app}: {exc}", data={}
            )
        logger.info("AppLaunchTool: launched macOS bundle '%s'.", bundle_name)
        return ToolResult(success=True, output=f"Launched: {app}", data={})


# ---------------------------------------------------------------------------
# ClipboardTool
# ---------------------------------------------------------------------------


class ClipboardTool:
    """Read from or write to the system clipboard.

    Schema:
        name: "clipboard"
        args:
            action (str, required): "read" | "write"
            text   (str, optional): Required when action="write". Max 10,000 chars.

    Returns (read):
        ToolResult(success=True, output="<clipboard text>", data={})

    Returns (write):
        ToolResult(success=True, output="Clipboard updated.", data={})

    Platform backends:
        Linux   — xclip (X11) or wl-clipboard (Wayland)
        macOS   — pbpaste / pbcopy (built-in)
        Windows — pyperclip (optional; install with: pip install pyperclip)
    """

    name: str = "clipboard"
    description: str = (
        "Read from or write to the system clipboard. "
        'Args: {"action": "read"} or {"action": "write", "text": "<content>"}.'
    )

    _MAX_TEXT_LEN: int = 10_000

    def execute(self, args: dict[str, Any]) -> ToolResult:
        action: Any = args.get("action", "")

        if not isinstance(action, str) or action not in ("read", "write"):
            return ToolResult(
                success=False,
                output="Invalid or missing 'action'. Use 'read' or 'write'.",
                data={},
            )

        if action == "read":
            return self._read()
        return self._write(args.get("text", ""))

    # ------------------------------------------------------------------
    # Read dispatch
    # ------------------------------------------------------------------

    def _read(self) -> ToolResult:
        platform = _get_platform()
        if platform == "darwin":
            return self._read_darwin()
        if platform == "win32":
            return self._read_win32()
        return self._read_linux()

    def _read_linux(self) -> ToolResult:
        xclip = shutil.which("xclip")
        if xclip is not None:
            return self._run_read([xclip, "-selection", "clipboard", "-o"])
        wl_paste = shutil.which("wl-paste")
        if wl_paste is not None:
            return self._run_read([wl_paste])
        logger.warning("ClipboardTool: no clipboard tool found on Linux PATH.")
        return ToolResult(
            success=False,
            output=(
                "No clipboard tool found. "
                "Install xclip (sudo apt install xclip) "
                "or wl-clipboard (sudo apt install wl-clipboard)."
            ),
            data={},
        )

    def _read_darwin(self) -> ToolResult:
        pbpaste = shutil.which("pbpaste")
        if pbpaste is None:
            logger.warning("ClipboardTool: pbpaste not found on macOS PATH.")
            return ToolResult(
                success=False,
                output="pbpaste not found (expected built into macOS).",
                data={},
            )
        return self._run_read([pbpaste])

    def _read_win32(self) -> ToolResult:
        try:
            import pyperclip

            return ToolResult(success=True, output=pyperclip.paste(), data={})
        except ImportError:
            logger.warning("ClipboardTool: pyperclip not installed on Windows.")
            return ToolResult(
                success=False,
                output="pyperclip not installed. Run: pip install pyperclip",
                data={},
            )
        except Exception as exc:
            logger.warning("ClipboardTool: Windows clipboard read failed: %s", exc)
            return ToolResult(
                success=False, output=f"Clipboard read failed: {exc}", data={}
            )

    def _run_read(self, cmd: list[str]) -> ToolResult:
        try:
            proc = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=5
            )
            text = proc.stdout
            logger.info("ClipboardTool: read %d chars from clipboard.", len(text))
            return ToolResult(success=True, output=text, data={})
        except FileNotFoundError:
            logger.warning("ClipboardTool: %s disappeared during read.", cmd[0])
            return ToolResult(
                success=False, output=f"{cmd[0]} not found.", data={}
            )
        except subprocess.TimeoutExpired:
            logger.warning("ClipboardTool: clipboard read timed out.")
            return ToolResult(
                success=False, output="Clipboard read timed out.", data={}
            )

    # ------------------------------------------------------------------
    # Write dispatch
    # ------------------------------------------------------------------

    def _write(self, text: Any) -> ToolResult:
        if not isinstance(text, str):
            return ToolResult(
                success=False,
                output="'text' must be a string for clipboard write.",
                data={},
            )

        if len(text) > self._MAX_TEXT_LEN:
            logger.warning(
                "ClipboardTool: write rejected — text length %d > %d limit.",
                len(text),
                self._MAX_TEXT_LEN,
            )
            return ToolResult(
                success=False,
                output=f"Text too long: {len(text)} chars (max {self._MAX_TEXT_LEN}).",
                data={},
            )

        platform = _get_platform()
        if platform == "darwin":
            return self._write_darwin(text)
        if platform == "win32":
            return self._write_win32(text)
        return self._write_linux(text)

    def _write_linux(self, text: str) -> ToolResult:
        xclip = shutil.which("xclip")
        if xclip is not None:
            return self._run_write([xclip, "-selection", "clipboard"], text)
        wl_copy = shutil.which("wl-copy")
        if wl_copy is not None:
            return self._run_write([wl_copy], text)
        logger.warning("ClipboardTool: no clipboard tool found on Linux PATH.")
        return ToolResult(
            success=False,
            output=(
                "No clipboard tool found. "
                "Install xclip (sudo apt install xclip) "
                "or wl-clipboard (sudo apt install wl-clipboard)."
            ),
            data={},
        )

    def _write_darwin(self, text: str) -> ToolResult:
        pbcopy = shutil.which("pbcopy")
        if pbcopy is None:
            logger.warning("ClipboardTool: pbcopy not found on macOS PATH.")
            return ToolResult(
                success=False,
                output="pbcopy not found (expected built into macOS).",
                data={},
            )
        return self._run_write([pbcopy], text)

    def _write_win32(self, text: str) -> ToolResult:
        try:
            import pyperclip

            pyperclip.copy(text)
            logger.info("ClipboardTool: wrote %d chars to clipboard (Windows).", len(text))
            return ToolResult(success=True, output="Clipboard updated.", data={})
        except ImportError:
            logger.warning("ClipboardTool: pyperclip not installed on Windows.")
            return ToolResult(
                success=False,
                output="pyperclip not installed. Run: pip install pyperclip",
                data={},
            )
        except Exception as exc:
            logger.warning("ClipboardTool: Windows clipboard write failed: %s", exc)
            return ToolResult(
                success=False, output=f"Clipboard write failed: {exc}", data={}
            )

    def _run_write(self, cmd: list[str], text: str) -> ToolResult:
        try:
            proc = subprocess.run(  # noqa: S603
                cmd, input=text, capture_output=True, text=True, timeout=5
            )
            if proc.returncode != 0:
                logger.warning(
                    "ClipboardTool: %s write exited with code %d.",
                    cmd[0],
                    proc.returncode,
                )
                return ToolResult(
                    success=False,
                    output=f"{cmd[0]} write failed (exit {proc.returncode}).",
                    data={},
                )
            logger.info("ClipboardTool: wrote %d chars to clipboard.", len(text))
            return ToolResult(success=True, output="Clipboard updated.", data={})
        except FileNotFoundError:
            logger.warning("ClipboardTool: %s disappeared during write.", cmd[0])
            return ToolResult(
                success=False, output=f"{cmd[0]} not found.", data={}
            )
        except subprocess.TimeoutExpired:
            logger.warning("ClipboardTool: clipboard write timed out.")
            return ToolResult(
                success=False, output="Clipboard write timed out.", data={}
            )


# ---------------------------------------------------------------------------
# FileInfoTool
# ---------------------------------------------------------------------------


class FileInfoTool:
    """Return stat information about a filesystem path.

    Schema:
        name: "file_info"
        args:
            path (str, required): Absolute or relative path to inspect.

    Returns:
        ToolResult(
            success=True,
            output="<human-readable summary>",
            data={"size": int, "is_dir": bool, "exists": bool}
        )

    Security: paths containing ".." are rejected before any os.stat() call.
    Cross-platform: uses pathlib — works on Linux, macOS, and Windows.
    """

    name: str = "file_info"
    description: str = (
        "Return filesystem metadata for a path (size, type, exists). "
        'Args: {"path": "<absolute or relative path>"}.'
    )

    def execute(self, args: dict[str, Any]) -> ToolResult:
        raw_path: Any = args.get("path", "")

        if not isinstance(raw_path, str) or not raw_path:
            return ToolResult(
                success=False, output="Missing required arg: path", data={}
            )

        if ".." in Path(raw_path).parts:
            logger.warning("FileInfoTool: path traversal rejected for '%s'.", raw_path)
            return ToolResult(success=False, output="Invalid path", data={})

        resolved = Path(raw_path).resolve()

        try:
            stat_result = resolved.stat()
        except FileNotFoundError:
            logger.info("FileInfoTool: path '%s' does not exist.", resolved)
            return ToolResult(
                success=True,
                output=f"Path does not exist: {resolved}",
                data={"size": 0, "is_dir": False, "exists": False},
            )
        except PermissionError as exc:
            logger.warning(
                "FileInfoTool: permission denied for '%s': %s.", resolved, exc
            )
            return ToolResult(
                success=False,
                output=f"Permission denied: {resolved}",
                data={},
            )
        except OSError as exc:
            logger.error("FileInfoTool: OSError for '%s': %s.", resolved, exc)
            return ToolResult(
                success=False,
                output=f"Error accessing path: {exc}",
                data={},
            )

        is_dir = resolved.is_dir()
        size = stat_result.st_size
        kind = "directory" if is_dir else "file"
        output = f"{kind}: {resolved} ({size} bytes)"
        logger.info("FileInfoTool: %s", output)

        return ToolResult(
            success=True,
            output=output,
            data={"size": size, "is_dir": is_dir, "exists": True},
        )


# ---------------------------------------------------------------------------
# WindowListTool
# ---------------------------------------------------------------------------

# AppleScript that collects "app: window" pairs for visible processes on macOS.
_DARWIN_WINDOW_SCRIPT = """\
set out to ""
tell application "System Events"
  repeat with p in (every process whose visible is true)
    set pName to name of p
    try
      repeat with w in every window of p
        set out to out & pName & ": " & (name of w) & "\\n"
      end repeat
    end try
  end repeat
end tell
return out"""


class WindowListTool:
    """List currently open windows on the desktop.

    Schema:
        name: "window_list"
        args: {} (no arguments required)

    Returns:
        ToolResult(
            success=True,
            output="N windows",
            data={"windows": [...]}
        )

    Platform backends:
        Linux   — wmctrl; each entry has id/desktop/host/title keys
        macOS   — osascript; each entry has app/title keys
        Windows — pygetwindow (optional); each entry has title key
    """

    name: str = "window_list"
    description: str = (
        "List all open windows on the desktop. "
        "Args: {} (no arguments needed)."
    )

    def execute(self, args: dict[str, Any]) -> ToolResult:  # args unused
        platform = _get_platform()
        if platform == "darwin":
            return self._list_darwin()
        if platform == "win32":
            return self._list_win32()
        return self._list_linux()

    # ------------------------------------------------------------------
    # Platform backends
    # ------------------------------------------------------------------

    def _list_linux(self) -> ToolResult:
        wmctrl = shutil.which("wmctrl")
        if wmctrl is None:
            logger.warning("WindowListTool: wmctrl not found on PATH.")
            return ToolResult(
                success=False,
                output="wmctrl not found. Install with: sudo apt install wmctrl",
                data={},
            )

        try:
            proc = subprocess.run(  # noqa: S603
                [wmctrl, "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError:
            logger.warning("WindowListTool: wmctrl disappeared during run.")
            return ToolResult(
                success=False,
                output="wmctrl not found. Install with: sudo apt install wmctrl",
                data={},
            )
        except subprocess.TimeoutExpired:
            logger.warning("WindowListTool: wmctrl timed out.")
            return ToolResult(success=False, output="wmctrl timed out.", data={})

        windows = self._parse_wmctrl(proc.stdout)
        count = len(windows)
        logger.info("WindowListTool: found %d windows (Linux).", count)
        return ToolResult(
            success=True,
            output=f"{count} windows",
            data={"windows": windows},
        )

    def _list_darwin(self) -> ToolResult:
        osascript = shutil.which("osascript")
        if osascript is None:
            logger.warning("WindowListTool: osascript not found on macOS PATH.")
            return ToolResult(
                success=False,
                output="osascript not found (expected built into macOS).",
                data={},
            )

        try:
            proc = subprocess.run(  # noqa: S603
                [osascript, "-e", _DARWIN_WINDOW_SCRIPT],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            logger.warning("WindowListTool: osascript timed out.")
            return ToolResult(success=False, output="osascript timed out.", data={})

        windows = self._parse_osascript(proc.stdout)
        count = len(windows)
        logger.info("WindowListTool: found %d windows (macOS).", count)
        return ToolResult(
            success=True,
            output=f"{count} windows",
            data={"windows": windows},
        )

    def _list_win32(self) -> ToolResult:
        try:
            import pygetwindow

            titles = [t for t in pygetwindow.getAllTitles() if t]
            windows = [{"title": t} for t in titles]
            count = len(windows)
            logger.info("WindowListTool: found %d windows (Windows).", count)
            return ToolResult(
                success=True,
                output=f"{count} windows",
                data={"windows": windows},
            )
        except ImportError:
            logger.warning("WindowListTool: pygetwindow not installed on Windows.")
            return ToolResult(
                success=False,
                output="pygetwindow not installed. Run: pip install pygetwindow",
                data={},
            )
        except Exception as exc:
            logger.warning("WindowListTool: pygetwindow failed: %s", exc)
            return ToolResult(
                success=False, output=f"Window list failed: {exc}", data={}
            )

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_wmctrl(output: str) -> list[dict[str, str]]:
        """Parse ``wmctrl -l`` output (space-separated: id desktop host title)."""
        windows: list[dict[str, str]] = []
        for line in output.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            windows.append(
                {
                    "id": parts[0],
                    "desktop": parts[1],
                    "host": parts[2],
                    "title": parts[3],
                }
            )
        return windows

    @staticmethod
    def _parse_osascript(output: str) -> list[dict[str, str]]:
        """Parse ``osascript`` output: one 'app: window' pair per line."""
        windows: list[dict[str, str]] = []
        for line in output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            app, sep, title = line.partition(": ")
            if sep:
                windows.append({"app": app, "title": title})
            else:
                windows.append({"title": line})
        return windows
