"""
OS action tools for Project Lumi.

Concrete Tool implementations for common desktop operations:
- AppLaunchTool  (name="launch_app")  — launch an application via Popen
- ClipboardTool  (name="clipboard")   — read/write system clipboard via xclip
- FileInfoTool   (name="file_info")   — stat a file path safely
- WindowListTool (name="window_list") — list open windows via wmctrl

Security decisions:
- AppLaunchTool: app_name is NEVER passed directly to subprocess; only the
  resolved binary path of a known-safe executable is passed, and that binary
  must exist on the system PATH via shutil.which().
- FileInfoTool: paths containing ".." components are rejected before any
  filesystem operation; pathlib.Path.resolve() is used for canonical paths.
- No shell=True anywhere — all subprocess calls use explicit argument lists.
- All inputs are validated before any subprocess or filesystem operation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


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
        "Launch a desktop application. "
        'Args: {"app": "<name>"}. '
        "Allowed apps: firefox, thunar, gedit, gnome-terminal, konsole, "
        "nautilus, libreoffice, vlc, xterm, mousepad."
    )

    # Hardcoded allowlist — the only apps this tool will ever launch.
    # Security rationale: user-controlled strings are NEVER passed directly
    # to subprocess; we look up the canonical binary of the allow-listed name.
    ALLOWED_APPS: frozenset[str] = frozenset(
        {
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

        # Allowlist gate — reject before any subprocess interaction.
        if app not in self.ALLOWED_APPS:
            logger.warning(
                "AppLaunchTool: app '%s' is not in allowlist; rejected.", app
            )
            return ToolResult(
                success=False, output=f"App not allowed: {app}", data={}
            )

        # Resolve binary — never pass user string directly to Popen.
        binary = shutil.which(app)
        if binary is None:
            logger.warning(
                "AppLaunchTool: app '%s' not found on PATH.", app
            )
            return ToolResult(
                success=False, output=f"App not found: {app}", data={}
            )

        try:
            subprocess.Popen(  # noqa: S603 — explicit list, no shell
                [binary],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning(
                "AppLaunchTool: FileNotFoundError launching '%s'.", app
            )
            return ToolResult(
                success=False, output=f"App not found: {app}", data={}
            )
        except OSError as exc:
            logger.error("AppLaunchTool: OSError launching '%s': %s.", app, exc)
            return ToolResult(
                success=False, output=f"Failed to launch {app}: {exc}", data={}
            )

        logger.info("AppLaunchTool: launched '%s' (%s).", app, binary)
        return ToolResult(success=True, output=f"Launched: {app}", data={})


# ---------------------------------------------------------------------------
# ClipboardTool
# ---------------------------------------------------------------------------


class ClipboardTool:
    """Read from or write to the system clipboard using xclip.

    Schema:
        name: "clipboard"
        args:
            action (str, required): "read" | "write"
            text   (str, optional): Required when action="write". Max 10,000 chars.

    Returns (read):
        ToolResult(success=True, output="<clipboard text>", data={})

    Returns (write):
        ToolResult(success=True, output="Clipboard updated.", data={})

    On xclip not installed:
        ToolResult(success=False, output="xclip not found. Install with: sudo apt install xclip", data={})
    """

    name: str = "clipboard"
    description: str = (
        "Read from or write to the system clipboard via xclip. "
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

    def _read(self) -> ToolResult:
        """Read clipboard contents via xclip."""
        xclip = shutil.which("xclip")
        if xclip is None:
            logger.warning("ClipboardTool: xclip not found on PATH.")
            return ToolResult(
                success=False,
                output="xclip not found. Install with: sudo apt install xclip",
                data={},
            )

        try:
            proc = subprocess.run(  # noqa: S603
                [xclip, "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            text = proc.stdout
            logger.info("ClipboardTool: read %d chars from clipboard.", len(text))
            return ToolResult(success=True, output=text, data={})
        except FileNotFoundError:
            logger.warning("ClipboardTool: xclip disappeared during read.")
            return ToolResult(
                success=False,
                output="xclip not found. Install with: sudo apt install xclip",
                data={},
            )
        except subprocess.TimeoutExpired:
            logger.warning("ClipboardTool: xclip read timed out.")
            return ToolResult(success=False, output="Clipboard read timed out.", data={})

    def _write(self, text: Any) -> ToolResult:
        """Write text to clipboard via xclip."""
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

        xclip = shutil.which("xclip")
        if xclip is None:
            logger.warning("ClipboardTool: xclip not found on PATH.")
            return ToolResult(
                success=False,
                output="xclip not found. Install with: sudo apt install xclip",
                data={},
            )

        try:
            proc = subprocess.run(  # noqa: S603
                [xclip, "-selection", "clipboard"],
                input=text,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode != 0:
                logger.warning(
                    "ClipboardTool: xclip write exited with code %d.", proc.returncode
                )
                return ToolResult(
                    success=False,
                    output=f"xclip write failed (exit {proc.returncode}).",
                    data={},
                )
            logger.info("ClipboardTool: wrote %d chars to clipboard.", len(text))
            return ToolResult(success=True, output="Clipboard updated.", data={})
        except FileNotFoundError:
            logger.warning("ClipboardTool: xclip disappeared during write.")
            return ToolResult(
                success=False,
                output="xclip not found. Install with: sudo apt install xclip",
                data={},
            )
        except subprocess.TimeoutExpired:
            logger.warning("ClipboardTool: xclip write timed out.")
            return ToolResult(success=False, output="Clipboard write timed out.", data={})


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

        # Security: reject path traversal attempts before touching the filesystem.
        if ".." in Path(raw_path).parts:
            logger.warning(
                "FileInfoTool: path traversal rejected for '%s'.", raw_path
            )
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
            logger.warning("FileInfoTool: permission denied for '%s': %s.", resolved, exc)
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


class WindowListTool:
    """List currently open windows using wmctrl.

    Schema:
        name: "window_list"
        args: {} (no arguments required)

    Returns:
        ToolResult(
            success=True,
            output="N windows",
            data={"windows": [{"id": str, "desktop": str, "host": str, "title": str}, ...]}
        )

    On wmctrl not installed:
        ToolResult(success=False, output="wmctrl not found. Install with: sudo apt install wmctrl", data={})
    """

    name: str = "window_list"
    description: str = (
        "List all open windows on the desktop using wmctrl. "
        "Args: {} (no arguments needed)."
    )

    def execute(self, args: dict[str, Any]) -> ToolResult:  # args unused
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
            return ToolResult(
                success=False, output="wmctrl timed out.", data={}
            )

        windows = self._parse_wmctrl(proc.stdout)
        count = len(windows)
        logger.info("WindowListTool: found %d windows.", count)

        return ToolResult(
            success=True,
            output=f"{count} windows",
            data={"windows": windows},
        )

    @staticmethod
    def _parse_wmctrl(output: str) -> list[dict[str, str]]:
        """Parse ``wmctrl -l`` output into a list of window dicts.

        wmctrl -l format (space-separated, title may contain spaces):
            <id>  <desktop>  <host>  <title...>
        """
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
