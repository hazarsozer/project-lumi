"""
Project Lumi — Tool Framework
==============================

This package provides the OS action tool layer for Project Lumi's local
desktop assistant. Tools are dispatched by the ReasoningRouter when the LLM
emits a tool call in the JSON schema::

    {"tool": "<tool_name>", "args": {...}}

Public API
----------
- ``ToolResult``   — frozen dataclass returned by every tool (success, output, data)
- ``Tool``         — Protocol that every concrete tool must satisfy
- ``ToolRegistry`` — maps tool names → Tool implementations
- ``ToolExecutor`` — dispatches tool call lists with allowlist + timeout enforcement

Registered Tools (source of truth for llm-engineer system prompt)
------------------------------------------------------------------

launch_app
~~~~~~~~~~
Launch an allowed desktop application.

Schema::

    {"tool": "launch_app", "args": {"app": "<name>"}}

Allowed app names: firefox, thunar, gedit, gnome-terminal, konsole,
nautilus, libreoffice, vlc, xterm, mousepad.

Returns: "Launched: <app>" on success.
Failure cases: app not in allowlist, app binary not on PATH.

clipboard
~~~~~~~~~
Read from or write to the system clipboard (requires xclip).

Read schema::

    {"tool": "clipboard", "args": {"action": "read"}}

Write schema::

    {"tool": "clipboard", "args": {"action": "write", "text": "<content>"}}

Constraints: text max 10,000 characters.
Returns (read): clipboard text content.
Returns (write): "Clipboard updated."
Failure cases: xclip not installed, text too long.

file_info
~~~~~~~~~
Return filesystem metadata for a path.

Schema::

    {"tool": "file_info", "args": {"path": "/absolute/or/relative/path"}}

Returns: JSON-friendly data with keys: size (int bytes), is_dir (bool),
exists (bool), plus a human-readable output string.
Failure cases: path traversal attempt (".."), permission denied.

window_list
~~~~~~~~~~~
List all open windows on the desktop (requires wmctrl).

Schema::

    {"tool": "window_list", "args": {}}

Returns: count string + data.windows list, each entry has:
id, desktop, host, title.
Failure cases: wmctrl not installed.

screenshot
~~~~~~~~~~
Capture the current screen and return a text description (requires vision to be
enabled in VisionConfig and the moondream2 GGUF model file to be present).

Schema::

    {"tool": "screenshot", "args": {}}

No arguments are required or accepted.

Returns: Text description of screen content on success.
Returns: "Screenshot captured but vision model not available." when model is absent.
Failure cases: no screenshot backend available (grim/scrot/Pillow), model load error.

Default ToolsConfig allowlist: ("launch_app", "clipboard", "file_info", "window_list")
Note: "screenshot" is registered separately when VisionConfig.enabled=True.
"""

from src.tools.base import Tool, ToolResult
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
]
