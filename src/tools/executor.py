"""
Tool executor for Project Lumi.

Dispatches validated tool call dicts to registered Tool implementations,
enforcing the allowed-tools allowlist, per-call timeout, and cancel flag.

Threading model: execute() is called synchronously from the orchestrator
worker thread. It must not spawn threads or use asyncio.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from src.core.config import ToolsConfig

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Dispatch tool calls from the LLM to registered Tool implementations.

    Args:
        registry: Populated ToolRegistry with all available tools.
        config:   ToolsConfig controlling the allowlist and timeout.
    """

    def __init__(self, registry: ToolRegistry, config: "ToolsConfig") -> None:
        self._registry = registry
        self._config = config

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
        cancel_flag: threading.Event,
    ) -> list[ToolResult]:
        """Execute a sequence of tool calls, returning results in order.

        Processing rules (applied in order):
        1. If ``cancel_flag`` is set before a call, stop and return what
           has been collected so far.
        2. If the tool name is not in ``config.allowed_tools``, return a
           failure ToolResult (tool is not executed).
        3. If the tool name is not registered, return a failure ToolResult.
        4. Run the tool with a threading.Timer enforcing
           ``config.execution_timeout_s``; on timeout return a failure
           ToolResult.
        5. If ``tool.execute()`` raises any exception, return a failure
           ToolResult (never propagate).

        Args:
            tool_calls: List of dicts each with ``"tool"`` and ``"args"``
                        keys, matching the LLM output schema
                        ``{"tool": "<name>", "args": {...}}``.
            cancel_flag: threading.Event; if set, execution stops early.

        Returns:
            List of ToolResult objects, one per tool call processed.
        """
        results: list[ToolResult] = []

        for call in tool_calls:
            # --- 1. Cancel check ---
            if cancel_flag.is_set():
                logger.info(
                    "ToolExecutor: cancel_flag set; stopping before '%s'.",
                    call.get("tool", "<unknown>"),
                )
                break

            tool_name: str = call.get("tool", "")
            args: dict[str, Any] = call.get("args") or {}

            logger.info(
                "ToolExecutor: attempting tool '%s' with args %s.",
                tool_name,
                args,
            )

            # --- 2. Allowlist check ---
            if tool_name not in self._config.allowed_tools:
                logger.warning(
                    "ToolExecutor: tool '%s' is not in allowed_tools; blocked.",
                    tool_name,
                )
                results.append(
                    ToolResult(
                        success=False,
                        output=f"Tool not allowed: {tool_name}",
                        data={},
                    )
                )
                continue

            # --- 3. Registry check ---
            tool = self._registry.get(tool_name)
            if tool is None:
                logger.warning(
                    "ToolExecutor: tool '%s' is in allowlist but not registered.",
                    tool_name,
                )
                results.append(
                    ToolResult(
                        success=False,
                        output=f"Tool not registered: {tool_name}",
                        data={},
                    )
                )
                continue

            # --- 4 & 5. Timed execution with exception guard ---
            result = self._run_with_timeout(tool_name, tool, args)
            results.append(result)

            logger.info(
                "ToolExecutor: tool '%s' finished success=%s.",
                tool_name,
                result.success,
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_with_timeout(
        self,
        tool_name: str,
        tool: Any,
        args: dict[str, Any],
    ) -> ToolResult:
        """Run tool.execute(args) with a hard timeout.

        Uses a threading.Event + threading.Timer to detect timeout without
        spawning a persistent thread per call — the worker thread runs the
        tool; the timer thread only sets a flag.

        Returns a failure ToolResult if the call exceeds
        ``config.execution_timeout_s`` or raises any exception.
        """
        result_holder: list[ToolResult] = []
        exc_holder: list[Exception] = []
        done_event = threading.Event()

        def _target() -> None:
            try:
                result_holder.append(tool.execute(args))
            except Exception as exc:  # noqa: BLE001
                exc_holder.append(exc)
            finally:
                done_event.set()

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()

        finished_in_time = done_event.wait(timeout=self._config.execution_timeout_s)

        if not finished_in_time:
            logger.warning(
                "ToolExecutor: tool '%s' timed out after %.1f s.",
                tool_name,
                self._config.execution_timeout_s,
            )
            return ToolResult(success=False, output="Timeout", data={})

        if exc_holder:
            exc = exc_holder[0]
            logger.error(
                "ToolExecutor: tool '%s' raised %s: %s.",
                tool_name,
                type(exc).__name__,
                exc,
            )
            return ToolResult(success=False, output=str(exc), data={})

        return result_holder[0]
