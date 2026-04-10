"""
Central event orchestrator for Project Lumi.

The orchestrator owns the event bus (a queue.Queue) and the state machine.
All components post events to the bus; the orchestrator dispatches them to
registered handlers in a single dedicated thread.

Usage:
    from src.core.orchestrator import Orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.run()  # blocks until ShutdownEvent
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from src.core.config import LumiConfig
from src.core.events import (
    InterruptEvent,
    LLMResponseReadyEvent,
    ShutdownEvent,
    TranscriptReadyEvent,
)
from src.core.state_machine import LumiState, StateMachine
from src.llm.memory import ConversationMemory
from src.llm.model_loader import ModelLoader
from src.llm.prompt_engine import PromptEngine
from src.llm.reasoning_router import ReasoningRouter
from src.llm.reflex_router import ReflexRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central coordinator that consumes events and dispatches to handlers.

    Args:
        config: The application configuration.
    """

    def __init__(self, config: LumiConfig) -> None:
        self._config: LumiConfig = config
        self._event_queue: queue.Queue[Any] = queue.Queue()
        self._state_machine: StateMachine = StateMachine()
        self._shutdown: bool = False
        self._handlers: dict[type, list[Callable[..., None]]] = {}

        # Cancel flag for in-flight LLM work. LLM workers should
        # periodically check this and abort when set.
        self._llm_cancel_flag: threading.Event = threading.Event()

        # Lock that makes the daemon thread's guard-check + transition_to
        # atomic with _handle_interrupt's set + transition, preventing an
        # illegal state transition when an interrupt and an inference
        # completion race each other.
        self._llm_state_lock: threading.Lock = threading.Lock()

        # LLM subsystem — components are created here; the model itself is
        # loaded on first use (ModelLoader.load() is deferred until inference).
        self._reflex_router: ReflexRouter = ReflexRouter()
        self._model_loader: ModelLoader = ModelLoader()
        self._prompt_engine: PromptEngine = PromptEngine()
        self._memory: ConversationMemory = ConversationMemory(
            config.llm.memory_dir
        )
        self._memory.load()
        self._reasoning_router: ReasoningRouter = ReasoningRouter(
            model_loader=self._model_loader,
            prompt_engine=self._prompt_engine,
            memory=self._memory,
            config=config.llm,
        )

        # Register built-in handlers.
        self.register_handler(ShutdownEvent, self._handle_shutdown)
        self.register_handler(InterruptEvent, self._handle_interrupt)
        self.register_handler(TranscriptReadyEvent, self._handle_transcript)

    @property
    def state_machine(self) -> StateMachine:
        """Expose the state machine for observer registration."""
        return self._state_machine

    @property
    def llm_cancel_flag(self) -> threading.Event:
        """Expose the LLM cancel flag for worker threads."""
        return self._llm_cancel_flag

    def post_event(self, event: Any) -> None:
        """Thread-safe: any component calls this to post an event.

        Args:
            event: A frozen dataclass event instance.
        """
        self._event_queue.put(event)

    def register_handler(
        self, event_type: type, handler: Callable[..., None]
    ) -> None:
        """Register a handler for a specific event type.

        Multiple handlers may be registered for the same event type;
        they are called in registration order.

        Args:
            event_type: The event class to handle.
            handler: A callable that receives the event instance.
        """
        self._handlers.setdefault(event_type, []).append(handler)

    def run(self) -> None:
        """Main event loop. Blocks until ShutdownEvent is received.

        Call from the main thread. Uses a 0.1 s timeout on queue.get
        so the loop checks _shutdown even when the queue is empty.
        """
        logger.info("Orchestrator starting event loop")

        while not self._shutdown:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._dispatch(event)

        logger.info("Orchestrator event loop exited")

    def _dispatch(self, event: Any) -> None:
        """Route an event to all registered handlers for its type.

        Args:
            event: The event instance to dispatch.
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug(
                "No handler registered for %s", event_type.__name__
            )
            return

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler %s raised an exception for %s",
                    handler.__name__,
                    event_type.__name__,
                )

    def _handle_transcript(self, event: TranscriptReadyEvent) -> None:
        """Handle TranscriptReadyEvent: route text to reflex or LLM.

        Fast-path: try ReflexRouter first (regex, no model needed).
        Slow-path: dispatch ReasoningRouter in a daemon thread so the
        event loop remains unblocked during inference.

        Args:
            event: The transcript event containing the user's spoken text.
        """
        self._state_machine.transition_to(LumiState.PROCESSING)
        self._llm_cancel_flag.clear()

        # Reflex fast-path — no model required.
        reflex_response = self._reflex_router.route(event.text)
        if reflex_response is not None:
            logger.debug("Reflex hit for %r -> %r", event.text, reflex_response)
            self._memory.add_turn("user", event.text)
            self._memory.add_turn("assistant", reflex_response)
            self._memory.save()
            self.post_event(LLMResponseReadyEvent(text=reflex_response))
            self._state_machine.transition_to(LumiState.SPEAKING)
            return

        # Reasoning slow-path — run in a daemon thread so the event loop
        # remains responsive to InterruptEvents during long inference.
        def _run_inference() -> None:
            try:
                response = self._reasoning_router.generate(
                    event.text, self._llm_cancel_flag
                )
            except InterruptedError:
                logger.info("LLM generation cancelled for %r", event.text)
                return
            except Exception:
                logger.exception("LLM inference failed for %r", event.text)
                with self._llm_state_lock:
                    if self._state_machine.current_state == LumiState.PROCESSING:
                        self._state_machine.transition_to(LumiState.IDLE)
                return

            # Hold the state lock so this check+transition is atomic with
            # _handle_interrupt's own set+transition block.  Without the lock,
            # the interrupt handler could transition to IDLE between the guard
            # check here and the transition_to(SPEAKING) call below, resulting
            # in an illegal IDLE→SPEAKING transition.
            with self._llm_state_lock:
                if self._state_machine.current_state != LumiState.PROCESSING:
                    logger.debug(
                        "State changed during inference, discarding response for %r",
                        event.text,
                    )
                    return

                self._memory.save()
                self.post_event(LLMResponseReadyEvent(text=response))
                self._state_machine.transition_to(LumiState.SPEAKING)

        thread = threading.Thread(target=_run_inference, daemon=True)
        thread.start()

    def _handle_shutdown(self, event: ShutdownEvent) -> None:
        """Handle ShutdownEvent: stop the event loop.

        Args:
            event: The shutdown event.
        """
        logger.info("Shutdown requested")
        self._shutdown = True

    def _handle_interrupt(self, event: InterruptEvent) -> None:
        """Handle InterruptEvent: cancel in-flight work, return to IDLE.

        Args:
            event: The interrupt event with source info.
        """
        current = self._state_machine.current_state
        logger.info(
            "Interrupt received (source=%s) in state %s",
            event.source,
            current.value,
        )

        if current == LumiState.IDLE:
            logger.debug("Already IDLE, ignoring interrupt")
            return

        if current == LumiState.PROCESSING:
            # Hold the state lock so the inference daemon cannot slip between
            # its guard check and transition_to(SPEAKING) while we cancel.
            # Clear the flag *before* transitioning so a new inference thread
            # started immediately after the interrupt does not see a stale set.
            with self._llm_state_lock:
                self._llm_cancel_flag.set()
                self._drain_event_types(
                    {"LLMResponseReadyEvent", "TranscriptReadyEvent"}
                )
                self._llm_cancel_flag.clear()
                self._state_machine.transition_to(LumiState.IDLE)
            return

        if current == LumiState.SPEAKING:
            self._drain_event_types({"TTSChunkReadyEvent"})

        # SPEAKING → IDLE transition.
        self._state_machine.transition_to(LumiState.IDLE)
        self._llm_cancel_flag.clear()

    def _drain_event_types(self, type_names: set[str]) -> None:
        """Remove events of the given type names from the queue.

        This is best-effort: events may arrive after the drain. The
        orchestrator re-checks state before dispatching anyway.

        Args:
            type_names: Set of event class names to discard.
        """
        retained: list[Any] = []
        try:
            while True:
                item = self._event_queue.get_nowait()
                if type(item).__name__ not in type_names:
                    retained.append(item)
        except queue.Empty:
            pass

        for item in retained:
            self._event_queue.put(item)
