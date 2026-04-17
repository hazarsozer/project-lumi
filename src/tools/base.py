"""
Base types for the Project Lumi tool framework.

All tool implementations return a ToolResult — they never raise for
user-facing errors. The Tool Protocol defines the interface that every
concrete tool must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResult:
    """Immutable result returned by every tool execution.

    Attributes:
        success: True when the tool completed without error.
        output:  Human-readable result string for the LLM to reason about.
        data:    Structured data dict (empty on failure).
    """

    success: bool
    output: str
    data: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Protocol that every concrete tool must satisfy.

    Implementors must expose:
        name        — unique string identifier used in tool call JSON
        description — one-line description embedded in the LLM system prompt
        execute()   — synchronous execution; returns ToolResult, never raises
    """

    name: str
    description: str

    def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            args: Dictionary of arguments parsed from the LLM tool call.

        Returns:
            ToolResult with success=True on success, success=False on any
            validation or execution failure (never raises).
        """
        ...
