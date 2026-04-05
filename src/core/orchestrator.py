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
from src.core.events import InterruptEvent, ShutdownEvent
from src.core.state_machine import LumiState, StateMachine

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

        # Register built-in handlers.
        self.register_handler(ShutdownEvent, self._handle_shutdown)
        self.register_handler(InterruptEvent, self._handle_interrupt)

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
            # Signal LLM workers to abort.
            self._llm_cancel_flag.set()
            self._drain_event_types(
                {"LLMResponseReadyEvent", "TranscriptReadyEvent"}
            )

        if current == LumiState.SPEAKING:
            self._drain_event_types({"TTSChunkReadyEvent"})

        # Transition back to IDLE.
        self._state_machine.transition_to(LumiState.IDLE)

        # Clear the cancel flag so future LLM work proceeds normally.
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
