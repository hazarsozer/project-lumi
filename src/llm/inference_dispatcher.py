"""LLM inference dispatcher — owns the inference thread, watchdog, and tool pass."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from src.core.config import LLMConfig, LumiConfig
from src.core.events import LLMResponseReadyEvent
from src.core.state_machine import LumiState, StateMachine
from src.llm.reasoning_router import ReasoningRouter
from src.llm.reflex_router import ReflexRouter
from src.llm.tool_call_parser import parse_tool_calls

if TYPE_CHECKING:
    import queue
    from src.llm.memory import ConversationMemory
    from src.llm.model_loader import ModelLoader
    from src.tools import ToolExecutor

logger = logging.getLogger(__name__)


class LLMInferenceDispatcher:
    """Owns the inference thread, per-turn watchdog, and tool-call pass.

    Extracted from Orchestrator._dispatch_user_turn to reduce the god-class.
    The orchestrator retains aliases to cancel_flag and the lock objects for
    use by _handle_interrupt and _handle_llm_response.
    """

    def __init__(
        self,
        *,
        model_loader: ModelLoader,
        reflex_router: ReflexRouter,
        reasoning_router: ReasoningRouter,
        memory: ConversationMemory,
        tool_executor: ToolExecutor,
        state_machine: StateMachine,
        event_queue: queue.Queue[Any],
        llm_config: LLMConfig,
    ) -> None:
        self._model_loader = model_loader
        self._reflex_router = reflex_router
        self._reasoning_router = reasoning_router
        self._memory = memory
        self._tool_executor = tool_executor
        self._state_machine = state_machine
        self._event_queue = event_queue
        self._llm_config = llm_config

        self._llm_cancel_flag: threading.Event = threading.Event()
        self._llm_state_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public properties (aliased by Orchestrator for interrupt wiring)
    # ------------------------------------------------------------------

    @property
    def cancel_flag(self) -> threading.Event:
        return self._llm_cancel_flag

    @property
    def llm_state_lock(self) -> threading.Lock:
        return self._llm_state_lock

    def reconfigure(self, new_config: LumiConfig) -> None:
        self._llm_config = new_config.llm

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        text: str,
        source: str,
        rag_runtime_enabled: bool,
        post_event: Any,
    ) -> None:
        """Run reflex fast-path or launch reasoning slow-path in a daemon thread.

        Args:
            text:                The user's input text.
            source:              Label for log messages ("transcript" or "user_text").
            rag_runtime_enabled: Whether RAG retrieval is currently active.
            post_event:          ``Orchestrator.post_event`` callable for posting events.
        """
        self._llm_cancel_flag.clear()

        # Reflex fast-path — no model required.
        reflex_response = self._reflex_router.route(text)
        if reflex_response is not None:
            logger.debug("Reflex hit for %r -> %r", text, reflex_response)
            self._memory.add_turn("user", text)
            self._memory.add_turn("assistant", reflex_response)
            self._memory.save()
            post_event(LLMResponseReadyEvent(text=reflex_response))
            self._state_machine.transition_to(LumiState.SPEAKING)
            return

        utterance_id = str(uuid.uuid4())
        use_rag = rag_runtime_enabled and self._reflex_router.route_rag_intent(text)

        def _run_inference() -> None:
            try:
                if not self._model_loader.is_loaded:
                    logger.info("Loading LLM model on first inference request...")
                    self._model_loader.load(self._llm_config)
                response = self._reasoning_router.generate(
                    text,
                    self._llm_cancel_flag,
                    utterance_id=utterance_id,
                    use_rag=use_rag,
                )
            except InterruptedError:
                logger.info("LLM generation cancelled for %r (source=%s)", text, source)
                return
            except Exception:
                logger.exception(
                    "LLM inference failed for %r (source=%s)", text, source
                )
                with self._llm_state_lock:
                    if self._state_machine.current_state == LumiState.PROCESSING:
                        self._state_machine.transition_to(LumiState.IDLE)
                return

            # Tool-call pass: execute any tool calls and do a follow-up inference.
            tool_calls = parse_tool_calls(response)
            if tool_calls:
                results = self._tool_executor.execute(tool_calls, self._llm_cancel_flag)
                result_lines = [
                    f"Tool {tc['tool']!r}: {'OK' if tr.success else 'FAIL'} — {tr.output}"
                    for tc, tr in zip(tool_calls, results, strict=False)
                ]
                followup_prompt = f"{text}\n\n[Tool results]\n" + "\n".join(result_lines)
                try:
                    response = self._reasoning_router.generate(
                        followup_prompt,
                        self._llm_cancel_flag,
                        utterance_id=utterance_id,
                    )
                except InterruptedError:
                    logger.info(
                        "LLM tool-followup cancelled for %r (source=%s)", text, source
                    )
                    return
                except Exception:
                    logger.exception(
                        "LLM tool-followup failed for %r (source=%s)", text, source
                    )
                    with self._llm_state_lock:
                        if self._state_machine.current_state == LumiState.PROCESSING:
                            self._state_machine.transition_to(LumiState.IDLE)
                    return

            with self._llm_state_lock:
                if self._state_machine.current_state != LumiState.PROCESSING:
                    logger.debug(
                        "State changed during inference, discarding response for %r (source=%s)",
                        text,
                        source,
                    )
                    return
                try:
                    self._memory.save()
                    post_event(LLMResponseReadyEvent(text=response))
                    self._state_machine.transition_to(LumiState.SPEAKING)
                except Exception:
                    logger.exception(
                        "Post-inference save/dispatch failed for %r (source=%s); returning to IDLE",
                        text,
                        source,
                    )
                    self._state_machine.transition_to(LumiState.IDLE)

        _timeout_s = self._llm_config.inference_timeout_s
        _watchdog_timer: threading.Timer | None = None

        if _timeout_s > 0.0:
            def _watchdog_fn() -> None:
                logger.warning(
                    "LLM inference watchdog fired after %.1f s for %r (source=%s) — "
                    "setting cancel flag and returning to IDLE",
                    _timeout_s,
                    text,
                    source,
                )
                self._llm_cancel_flag.set()
                with self._llm_state_lock:
                    if self._state_machine.current_state == LumiState.PROCESSING:
                        self._state_machine.transition_to(LumiState.IDLE)

            _watchdog_timer = threading.Timer(_timeout_s, _watchdog_fn)
            _watchdog_timer.daemon = True
            _watchdog_timer.start()

        def _run_inference_with_watchdog() -> None:
            try:
                _run_inference()
            finally:
                if _watchdog_timer is not None:
                    _watchdog_timer.cancel()

        thread = threading.Thread(
            target=_run_inference_with_watchdog, daemon=True, name="LLMInferenceThread"
        )
        thread.start()
