"""
Tool registry for Project Lumi.

Maintains a name → Tool mapping. Tools are registered at startup and
looked up by name during execution. Thread-safe for read-heavy workloads
(registration happens once at startup; concurrent reads need no locking).
"""

from __future__ import annotations

import logging

from src.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry mapping tool names to Tool implementations.

    Usage:
        registry = ToolRegistry()
        registry.register(my_tool)
        tool = registry.get("my_tool_name")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool under its ``name`` attribute.

        If a tool with the same name is already registered, it is silently
        overwritten (last write wins). Logs a warning on overwrite.

        Args:
            tool: Any object satisfying the Tool Protocol.
        """
        if tool.name in self._tools:
            logger.warning("ToolRegistry: overwriting existing tool '%s'.", tool.name)
        self._tools[tool.name] = tool
        logger.debug("ToolRegistry: registered tool '%s'.", tool.name)

    def get(self, name: str) -> Tool | None:
        """Return the Tool registered under ``name``, or None if unknown.

        Args:
            name: Tool name string.

        Returns:
            The Tool instance, or None.
        """
        return self._tools.get(name)

    def is_registered(self, name: str) -> bool:
        """Return True if a tool with the given name is registered.

        Args:
            name: Tool name string.
        """
        return name in self._tools

    def list_tools(self) -> list[dict[str, str]]:
        """Return a list of ``{"name": ..., "description": ...}`` dicts.

        Suitable for embedding in an LLM system prompt so the model knows
        which tools are available.

        Returns:
            List of dicts, one per registered tool, sorted by name.
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in sorted(self._tools.values(), key=lambda t: t.name)
        ]
