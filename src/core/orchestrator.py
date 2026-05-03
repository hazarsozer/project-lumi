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

import dataclasses
import logging
import queue
import threading
import uuid
from collections.abc import Callable
from typing import Any

from src.audio.ears import Ears
from src.audio.mouth import KokoroTTS
from src.audio.scribe import Scribe
from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig
from src.core.config_runtime import ConfigManager, ConfigUpdateResult
from src.core.config_schema import FIELD_META
from src.core.event_bridge import EventBridge
from src.core.events import (
    ConfigSchemaRequestEvent,
    ConfigUpdateEvent,
    EarsErrorCode,
    EarsErrorEvent,
    InterruptEvent,
    InterruptSource,
    LLMResponseReadyEvent,
    LLMTokenEvent,
    RAGRetrievalEvent,
    RAGSetEnabledEvent,
    RAGStatusEvent,
    RAGStatusRequestEvent,
    RecordingCompleteEvent,
    ShutdownEvent,
    SpeechCompletedEvent,
    SystemStatusEvent,
    SystemStatusSource,
    TimerExpiredEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    VisemeEvent,
    WakeDetectedEvent,
)
from src.core.state_machine import LumiState, StateMachine
from src.llm.memory import ConversationMemory
from src.llm.inference_dispatcher import LLMInferenceDispatcher
from src.llm.model_loader import ModelLoader
from src.llm.prompt_engine import PromptEngine
from src.llm.reasoning_router import ReasoningRouter
from src.llm.reflex_router import ReflexRouter
from src.tools import ToolExecutor, ToolRegistry
from src.tools.datetime_tool import DateTimeTool
from src.tools.os_actions import (
    AppLaunchTool,
    ClipboardTool,
    FileInfoTool,
    WindowListTool,
)
from src.tools.rag_ingest import RagIngestTool
from src.tools.timer_tool import TimerTool
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


def _flatten_config(config: LumiConfig) -> dict[str, Any]:
    """Flatten LumiConfig into a dotted-path key → value dict.

    Produces a dict whose keys match those in ``FIELD_META`` (from
    ``src.core.config_schema``).  Sub-config section fields are prefixed with
    ``"<section>."``; top-level scalars use their bare field name.

    Tuple-valued fields (e.g. ``tools.allowed_tools``) are converted to
    ``list`` so the result is JSON-serialisable.

    Args:
        config: The ``LumiConfig`` instance to flatten.

    Returns:
        A flat ``dict[str, Any]`` of dotted-path keys to current values.
    """
    result: dict[str, Any] = {
        "edition": config.edition,
        "log_level": config.log_level,
        "json_logs": config.json_logs,
    }
    sections: dict[str, Any] = {
        "audio": config.audio,
        "scribe": config.scribe,
        "llm": config.llm,
        "tts": config.tts,
        "ipc": config.ipc,
        "tools": config.tools,
        "vision": config.vision,
        "rag": config.rag,
        "persona": config.persona,
    }
    for section_name, section_obj in sections.items():
        for f in dataclasses.fields(section_obj):
            val = getattr(section_obj, f.name)
            # Convert tuple → list for JSON serialisability.
            if isinstance(val, tuple):
                val = list(val)
            result[f"{section_name}.{f.name}"] = val
    return result


class Orchestrator:
    """Central coordinator that consumes events and dispatches to handlers.

    Args:
        config: The application configuration.
    """

    def __init__(
        self,
        config: LumiConfig,
        *,
        speaker: SpeakerThread | None = None,
        tts: KokoroTTS | None = None,
        event_bridge: EventBridge | None = None,
        ears: Ears | None = None,
        scribe: Scribe | None = None,
        missing_setup_items: list[str] | None = None,
    ) -> None:
        self._config: LumiConfig = config
        self._config_manager: ConfigManager = ConfigManager(config)
        self._event_queue: queue.Queue[Any] = queue.Queue()
        self._state_machine: StateMachine = StateMachine()
        self._shutdown: bool = False
        self._handlers: dict[type, list[Callable[..., None]]] = {}

        # LLM subsystem — components are created here; the model itself is
        # loaded on first use (ModelLoader.load() is deferred until inference).
        self._reflex_router: ReflexRouter = ReflexRouter()
        self._model_loader: ModelLoader = ModelLoader()
        self._prompt_engine: PromptEngine = PromptEngine(config=config)
        self._memory: ConversationMemory = ConversationMemory(config.llm.memory_dir)
        self._memory.load()
        # RAG subsystem — built only when enabled in config.
        self._rag_runtime_enabled: bool = config.rag.enabled
        self._rag_store = None
        self._rag_retriever = None
        if config.rag.enabled:
            try:
                from src.rag.retriever import RAGRetriever  # noqa: PLC0415
                from src.rag.store import DocumentStore  # noqa: PLC0415

                self._rag_store = DocumentStore(config.rag)
                self._rag_retriever = RAGRetriever(self._rag_store, config.rag)
                logger.info("RAG subsystem initialised (db=%s)", config.rag.db_path)
            except Exception:
                logger.exception("RAG subsystem failed to initialise; disabling RAG")
                self._rag_runtime_enabled = False

        self._reasoning_router: ReasoningRouter = ReasoningRouter(
            model_loader=self._model_loader,
            prompt_engine=self._prompt_engine,
            memory=self._memory,
            config=config.llm,
            event_queue=self._event_queue,
            retriever=self._rag_retriever,
        )
        self._config_manager.register_observer("prompt_engine", self._prompt_engine)
        self._config_manager.register_observer("reasoning_router", self._reasoning_router)

        # Tool registry and executor — wired when tools are enabled.
        self._tool_registry: ToolRegistry = ToolRegistry()
        if config.tools.enabled:
            self._tool_registry.register(AppLaunchTool())
            self._tool_registry.register(ClipboardTool())
            self._tool_registry.register(FileInfoTool())
            self._tool_registry.register(WindowListTool())
            self._tool_registry.register(RagIngestTool(rag_config=config.rag))
            self._tool_registry.register(WebSearchTool())
            self._tool_registry.register(DateTimeTool())
            self._tool_registry.register(TimerTool(post_event=self.post_event))

        # Register ScreenshotTool if vision is enabled.
        if config.vision.enabled:
            from src.tools.vision import ScreenshotTool  # noqa: PLC0415

            self._tool_registry.register(
                ScreenshotTool(
                    config=config.vision,
                    llm_loader=self._model_loader,
                )
            )

        self._tool_executor: ToolExecutor = ToolExecutor(
            self._tool_registry, config.tools
        )

        # Inference dispatcher — owns the inference thread, watchdog, and tool pass.
        self._inference_dispatcher = LLMInferenceDispatcher(
            model_loader=self._model_loader,
            reflex_router=self._reflex_router,
            reasoning_router=self._reasoning_router,
            memory=self._memory,
            tool_executor=self._tool_executor,
            state_machine=self._state_machine,
            event_queue=self._event_queue,
            llm_config=config.llm,
        )
        # Aliases kept for _handle_interrupt and _handle_llm_response compatibility.
        self._llm_cancel_flag: threading.Event = self._inference_dispatcher.cancel_flag
        self._llm_state_lock: threading.Lock = self._inference_dispatcher.llm_state_lock

        # Speaker output thread — injectable for testing; created here otherwise.
        self._speaker: SpeakerThread = (
            speaker if speaker is not None else SpeakerThread(self._event_queue)
        )
        self._speaker.start()

        # TTS engine — injectable for testing; None means no TTS (state machine
        # still transitions correctly via a synthetic SpeechCompletedEvent).
        self._tts: KokoroTTS | None = tts

        # Guards _current_utterance_id and _tts_pending_count so that interrupt
        # and multi-sentence completion can atomically read and modify TTS state.
        self._tts_state_lock: threading.Lock = threading.Lock()
        self._current_utterance_id: str | None = None
        # Counts in-flight TTS sentences.  SpeechCompletedEvent only triggers
        # SPEAKING→IDLE when this reaches zero, preventing state-machine flicker
        # during multi-sentence streaming (C4 Trap 3 fix).
        self._tts_pending_count: int = 0

        # Audio-in pipeline — both are optional (None = text-only mode / testing).
        self._ears: Ears | None = ears
        self._scribe: Scribe | None = scribe
        self._missing_setup_items: list[str] = missing_setup_items or []

        # Push-to-talk listener — optional; created when config.audio.ptt_enabled.
        self._ptt_listener = None
        if config.audio.ptt_enabled:
            from src.audio.hotkey import PTTListener  # noqa: PLC0415

            self._ptt_listener = PTTListener(
                event_queue=self._event_queue,
                hotkey=config.audio.ptt_hotkey,
            )

        # Register built-in handlers.
        # NOTE: TranscriptReadyEvent and SpeechCompletedEvent each receive a
        # second handler below (on_transcript / on_tts_stop) when an EventBridge
        # is present.  Both registrations are intentional: the internal handler
        # runs first (state transitions), then the IPC forwarder sends the event
        # to Godot.  This ordering is guaranteed by registration order in _dispatch.
        self.register_handler(ShutdownEvent, self._handle_shutdown)
        self.register_handler(EarsErrorEvent, self._handle_ears_error)
        self.register_handler(InterruptEvent, self._handle_interrupt)
        self.register_handler(WakeDetectedEvent, self._handle_wake_detected)
        self.register_handler(RecordingCompleteEvent, self._handle_recording_complete)
        self.register_handler(TranscriptReadyEvent, self._handle_transcript)
        self.register_handler(LLMResponseReadyEvent, self._handle_llm_response)
        self.register_handler(SpeechCompletedEvent, self._handle_speech_completed)
        self.register_handler(UserTextEvent, self._handle_user_text)
        self.register_handler(RAGSetEnabledEvent, self._handle_rag_set_enabled)
        self.register_handler(RAGStatusRequestEvent, self._handle_rag_status_request)
        self.register_handler(TimerExpiredEvent, self._handle_timer_expired)
        self.register_handler(
            ConfigSchemaRequestEvent, self._handle_config_schema_request
        )
        self.register_handler(ConfigUpdateEvent, self._handle_config_update)

        # EventBridge wiring — optional; injected for testing or when IPC is
        # enabled.  If not injected but config.ipc.enabled is True, create it
        # here using the orchestrator's own queue and state machine so that
        # inbound events from the Godot frontend are posted to this event loop.
        # When created internally, EventBridge.__init__ registers on_state_change
        # as a state observer.  When injected (e.g. in tests), the caller is
        # responsible for ensuring the state machine is shared — the Orchestrator
        # registers on_state_change explicitly so injected instances also receive
        # state transition forwarding.
        self._event_bridge: EventBridge | None = event_bridge
        if self._event_bridge is None and config.ipc.enabled:
            self._event_bridge = EventBridge(
                config.ipc, self._event_queue, self._state_machine
            )
            self._event_bridge.start()

        if self._event_bridge is not None:
            # Injected instances have not had on_state_change registered against
            # this orchestrator's state machine; do it here.  For auto-created
            # instances EventBridge.__init__ already registered, so we avoid
            # double registration by only registering for the injected path.
            if event_bridge is not None:
                self._state_machine.register_observer(self._event_bridge.on_state_change)
            self.register_handler(VisemeEvent, self._event_bridge.on_tts_viseme)
            self.register_handler(SpeechCompletedEvent, self._event_bridge.on_tts_stop)
            self.register_handler(TranscriptReadyEvent, self._event_bridge.on_transcript)
            self.register_handler(LLMResponseReadyEvent, self._event_bridge.on_tts_start)
            self.register_handler(LLMTokenEvent, self._event_bridge.on_llm_token)
            self.register_handler(RAGRetrievalEvent, self._event_bridge.on_rag_retrieval)
            self.register_handler(RAGStatusEvent, self._event_bridge.on_rag_status)
            self.register_handler(SystemStatusEvent, self._event_bridge.on_system_status)

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

    def register_handler(self, event_type: type, handler: Callable[..., None]) -> None:
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
        Starts the Ears audio pipeline (if configured) before entering the loop.
        """
        logger.info("Orchestrator starting event loop")
        if self._ears is not None:
            self._ears.start(self._event_queue)
        if self._ptt_listener is not None:
            self._ptt_listener.start()

        self._event_queue.put(
            SystemStatusEvent(
                tts_available=self._tts is not None,
                rag_available=self._rag_runtime_enabled
                and self._rag_retriever is not None,
                mic_available=self._ears is not None,
                llm_available=True,
                source=SystemStatusSource.STARTUP,
                setup_required=bool(self._missing_setup_items),
                missing_items=tuple(self._missing_setup_items),
            )
        )

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
            logger.debug("No handler registered for %s", event_type.__name__)
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
        # _handle_recording_complete already transitioned to PROCESSING when it
        # started the Scribe thread; only advance if we somehow arrive from LISTENING.
        if self._state_machine.current_state != LumiState.PROCESSING:
            self._state_machine.transition_to(LumiState.PROCESSING)
        self._dispatch_user_turn(event.text, source="transcript")

    def _handle_user_text(self, event: UserTextEvent) -> None:
        """Handle UserTextEvent: route typed text to reflex or LLM.

        Mirrors _handle_transcript but the text arrives pre-formed from the
        Godot frontend — no STT step required.

        State guard: accepts events from IDLE or LISTENING only.  Events
        arriving in PROCESSING or SPEAKING are dropped; the caller (Godot)
        should wait for state_change→idle before sending new text.  When the
        machine is IDLE, this handler performs the IDLE→LISTENING→PROCESSING
        double-step because the wake-word pipeline that normally drives that
        transition is not involved in the text-input path.

        Args:
            event: The user-text event containing text from the frontend.
        """
        current = self._state_machine.current_state
        if current not in (LumiState.IDLE, LumiState.LISTENING):
            logger.debug("UserTextEvent received in state %s; dropping.", current.value)
            return

        # Text input bypasses the wake-word pipeline.  If we are IDLE, step
        # through LISTENING first so the LISTENING→PROCESSING transition below
        # is always valid.
        if current == LumiState.IDLE:
            self._state_machine.transition_to(LumiState.LISTENING)

        self._state_machine.transition_to(LumiState.PROCESSING)
        self._dispatch_user_turn(event.text, source="user_text")

    def _dispatch_user_turn(self, text: str, source: str) -> None:
        """Delegate inference dispatch to LLMInferenceDispatcher."""
        self._inference_dispatcher.dispatch(
            text=text,
            source=source,
            rag_runtime_enabled=self._rag_runtime_enabled,
            post_event=self.post_event,
        )

    def _handle_llm_response(self, event: LLMResponseReadyEvent) -> None:
        """Handle LLMResponseReadyEvent: pass text to TTS for synthesis.

        Generates a fresh utterance_id, records it for cancel targeting,
        then launches synthesis in a daemon thread so the event loop
        stays responsive.  When TTS is unavailable, posts SpeechCompletedEvent
        directly so the state machine can transition back to IDLE.

        Args:
            event: The LLM response event containing the text to speak.
        """
        utterance_id = str(uuid.uuid4())

        tts = self._tts

        # Increment pending count and prepare KokoroTTS BEFORE starting the
        # thread.  Both must happen under the same lock so a cancel() call
        # arriving between thread start and synthesize() can correctly target
        # the utterance.  The pending count ensures SpeechCompletedEvent only
        # triggers SPEAKING→IDLE when all streamed sentences have finished.
        with self._tts_state_lock:
            self._tts_pending_count += 1
            self._current_utterance_id = utterance_id
            if tts is not None:
                tts.prepare(utterance_id)

        if tts is None:
            # No TTS engine configured — fire completion immediately so the
            # pending count is decremented and state returns to IDLE.
            logger.debug(
                "No TTS engine; posting SpeechCompletedEvent for utterance_id=%s",
                utterance_id,
            )
            self.post_event(SpeechCompletedEvent(utterance_id=utterance_id))
            return

        def _run_tts() -> None:
            tts.synthesize(event.text, utterance_id)

        thread = threading.Thread(
            target=_run_tts, daemon=True, name="TTSSynthesisThread"
        )
        thread.start()
        logger.debug(
            "TTS synthesis started for utterance_id=%s (%.40r…)",
            utterance_id,
            event.text,
        )

    def _handle_speech_completed(self, event: SpeechCompletedEvent) -> None:
        """Handle SpeechCompletedEvent: transition from SPEAKING to IDLE.

        With multi-sentence streaming (C4), multiple LLMResponseReadyEvents fire
        per turn.  Each spawns a TTS thread that posts SpeechCompletedEvent when
        done.  The _tts_pending_count guard ensures SPEAKING→IDLE only fires when
        the last sentence has finished, preventing state-machine flicker between
        consecutive sentences.

        Args:
            event: The speech-completion event carrying the utterance identifier.
        """
        with self._tts_state_lock:
            self._tts_pending_count = max(0, self._tts_pending_count - 1)
            pending = self._tts_pending_count

        current = self._state_machine.current_state
        if current == LumiState.SPEAKING:
            if pending == 0:
                with self._tts_state_lock:
                    self._current_utterance_id = None
                self._state_machine.transition_to(LumiState.IDLE)
                logger.info(
                    "All sentences complete (utterance_id=%s), returning to IDLE",
                    event.utterance_id,
                )
            else:
                logger.debug(
                    "Sentence complete (utterance_id=%s), %d still pending",
                    event.utterance_id,
                    pending,
                )
        else:
            logger.debug(
                "SpeechCompletedEvent received in state %s, ignoring",
                current.value,
            )

    def _handle_timer_expired(self, event: TimerExpiredEvent) -> None:
        """Handle TimerExpiredEvent: speak a verbal alarm when a user timer fires.

        If Lumi is IDLE, she transitions to SPEAKING and announces the alarm.
        If Lumi is busy (PROCESSING or SPEAKING), the alarm is logged and skipped
        to avoid interrupting ongoing work.  This is an MVP limitation — a future
        version could queue the alarm for the next IDLE transition.

        Args:
            event: The expired timer, containing label and original seconds.
        """
        logger.info(
            "Timer expired: '%s' (%ds)", event.label, event.seconds
        )
        current = self._state_machine.current_state
        if current in (LumiState.IDLE, LumiState.LISTENING):
            alarm_text = f"Your {event.label} timer just went off!"
            if current == LumiState.IDLE:
                self._state_machine.transition_to(LumiState.LISTENING)
            self._state_machine.transition_to(LumiState.PROCESSING)
            self._state_machine.transition_to(LumiState.SPEAKING)
            self.post_event(LLMResponseReadyEvent(text=alarm_text))
        else:
            logger.warning(
                "Timer '%s' fired but Lumi is busy (state=%s) — alarm skipped",
                event.label,
                current.value,
            )

    def _handle_rag_set_enabled(self, event: RAGSetEnabledEvent) -> None:
        """Handle RAGSetEnabledEvent: toggle RAG retrieval at runtime.

        Only effective when a RAGRetriever was successfully constructed at
        startup (i.e. config.rag.enabled was True).  Silently ignored when
        no retriever is available so the event is always safe to send.

        Args:
            event: The enable/disable event from the ZMQ layer.
        """
        if self._rag_retriever is None:
            logger.warning(
                "RAGSetEnabledEvent(%s) ignored — no RAG retriever available",
                event.enabled,
            )
            return
        self._rag_runtime_enabled = event.enabled
        logger.info("RAG runtime enabled set to %s", event.enabled)

    def _handle_rag_status_request(self, event: RAGStatusRequestEvent) -> None:
        """Handle RAGStatusRequestEvent: build and post current RAG runtime state.

        Queries the store for live doc/chunk counts and posts a RAGStatusEvent,
        which the EventBridge outbound handler forwards to the frontend.
        """
        import datetime

        rag_active = self._rag_runtime_enabled and self._rag_retriever is not None
        doc_count = 0
        chunk_count = 0
        last_indexed = ""

        if rag_active and self._rag_store is not None:
            try:
                stats = self._rag_store.stats()
                doc_count = stats.doc_count
                chunk_count = stats.chunk_count
                if stats.last_indexed is not None:
                    last_indexed = datetime.datetime.fromtimestamp(
                        stats.last_indexed, tz=datetime.timezone.utc
                    ).isoformat()
            except Exception:
                logger.exception("RAG stats() failed; returning zeroes")

        self._event_queue.put(
            RAGStatusEvent(
                enabled=rag_active,
                doc_count=doc_count,
                chunk_count=chunk_count,
                last_indexed=last_indexed,
            )
        )

    def _handle_config_schema_request(self, event: ConfigSchemaRequestEvent) -> None:
        """Handle ConfigSchemaRequestEvent: send the full config schema + current values.

        Builds a flat dotted-path dict from the current ``LumiConfig`` and
        forwards it — along with ``FIELD_META`` — to the Godot frontend via
        ``EventBridge.send_config_schema()``.  If no ZMQ server is connected
        the handler returns silently.

        Args:
            event: The schema-request event (carries no payload).
        """
        if self._event_bridge is None:
            return
        current = self._config_manager.current
        current_values = _flatten_config(current)
        self._event_bridge.send_config_schema(FIELD_META, current_values)

    def _handle_config_update(self, event: ConfigUpdateEvent) -> None:
        """Handle ConfigUpdateEvent: apply config changes and report results.

        Delegates to ``ConfigManager.apply()`` for validation, live-apply, and
        optional persistence.  The resulting ``ConfigUpdateResult`` is forwarded
        to the Godot frontend via ``EventBridge.send_config_update_result()``.
        If no ZMQ server is connected the config is still applied but the
        result is not forwarded (silent success).

        Args:
            event: The config-update event carrying ``changes`` and ``persist``.
        """
        result: ConfigUpdateResult = self._config_manager.apply(
            event.changes, persist=event.persist
        )
        if self._event_bridge is not None:
            self._event_bridge.send_config_update_result(
                applied_live=result.applied_live,
                pending_restart=result.pending_restart,
                errors=result.errors,
            )

    def _handle_wake_detected(self, event: WakeDetectedEvent) -> None:
        """Handle WakeDetectedEvent: transition to LISTENING.

        Normal path (IDLE): IDLE → LISTENING.

        Interrupt path (SPEAKING): posts an InterruptEvent so that any
        in-flight TTS playback is cancelled, then transitions
        SPEAKING → IDLE → LISTENING so recording can begin immediately.
        This prevents a double-listen deadlock when the user says the wake
        word while Lumi is mid-response.

        Events arriving in PROCESSING or LISTENING are silently dropped to
        avoid corrupting the state machine during active inference.

        Args:
            event: The wake-word detection event posted by Ears.
        """
        current = self._state_machine.current_state

        if current == LumiState.IDLE:
            self._state_machine.transition_to(LumiState.LISTENING)
            logger.info("Wake word detected — transitioning to LISTENING")
            return

        if current == LumiState.SPEAKING:
            logger.info(
                "Wake word detected while SPEAKING — posting interrupt and entering LISTENING"
            )
            # Post InterruptEvent so _handle_interrupt can drain TTS queues
            # and cancel in-flight synthesis when it is processed by the loop.
            self._event_queue.put(InterruptEvent(source=InterruptSource.WAKE_WORD))
            # Transition synchronously so Ears can start recording without
            # waiting for the event loop to process the InterruptEvent first.
            self._state_machine.transition_to(LumiState.IDLE)
            self._state_machine.transition_to(LumiState.LISTENING)
            return

        logger.debug("WakeDetectedEvent received in state %s; ignoring.", current.value)

    def _handle_recording_complete(self, event: RecordingCompleteEvent) -> None:
        """Handle RecordingCompleteEvent: invoke Scribe in a daemon thread.

        Only processes the event when in LISTENING state; any other state means
        the recording was stale or arrived out of order.

        Scribe.transcribe() runs in a dedicated daemon thread so it cannot
        block the event-dispatch loop during long CPU-bound transcription.
        On success, TranscriptReadyEvent is posted back to the queue.
        On failure, the state machine falls back to IDLE.

        Args:
            event: The recording-complete event carrying the audio array.
        """
        current = self._state_machine.current_state
        if current != LumiState.LISTENING:
            logger.debug(
                "RecordingCompleteEvent received in state %s; ignoring.", current.value
            )
            return

        self._state_machine.transition_to(LumiState.PROCESSING)

        if self._scribe is None:
            logger.warning(
                "RecordingCompleteEvent received but no Scribe configured; returning to IDLE"
            )
            self._state_machine.transition_to(LumiState.IDLE)
            return

        thread = threading.Thread(
            target=self._run_scribe,
            args=(event.audio,),
            daemon=True,
            name="ScribeTranscribeThread",
        )
        thread.start()

    def _run_scribe(self, audio: Any) -> None:
        """Run Scribe.transcribe() and post the result to the event queue.

        Executed in a daemon thread started by _handle_recording_complete.
        Falls back to IDLE on any transcription error.

        Args:
            audio: The numpy audio array from RecordingCompleteEvent.
        """
        try:
            transcript = self._scribe.transcribe(audio)  # type: ignore[union-attr]
            self._event_queue.put(TranscriptReadyEvent(text=transcript))
        except Exception:
            logger.exception("Scribe transcription failed; returning to IDLE")
            self._state_machine.transition_to(LumiState.IDLE)

    def _handle_ears_error(self, event: EarsErrorEvent) -> None:
        """Handle EarsErrorEvent: log, notify frontend of mic degradation, return to IDLE.

        Args:
            event: The error event posted by the Ears thread on exhausting retries.
        """
        logger.error(
            "Ears unrecoverable error (code=%s): %s — forcing IDLE",
            event.code,
            event.detail,
        )
        current = self._state_machine.current_state
        if current != LumiState.IDLE:
            self._state_machine.transition_to(LumiState.IDLE)
        self._event_queue.put(
            SystemStatusEvent(
                tts_available=self._tts is not None,
                rag_available=self._rag_runtime_enabled
                and self._rag_retriever is not None,
                mic_available=False,
                llm_available=True,
                source=SystemStatusSource.DEGRADATION,
            )
        )

    def _handle_shutdown(self, event: ShutdownEvent) -> None:
        """Handle ShutdownEvent: stop the event loop.

        Args:
            event: The shutdown event.
        """
        logger.info("Shutdown requested")
        self._speaker.stop()
        if self._ears is not None:
            self._ears.stop()
        if self._ptt_listener is not None:
            self._ptt_listener.stop()
        if self._event_bridge is not None:
            self._event_bridge.stop()
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

        if current == LumiState.LISTENING:
            # Wake-while-speaking path: the state machine was already advanced
            # to LISTENING by _handle_wake_detected before this InterruptEvent
            # was enqueued.  Nothing further to cancel — recording is starting.
            logger.debug(
                "Interrupt received while LISTENING (wake-while-speaking); no-op"
            )
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
            with self._tts_state_lock:
                if self._tts is not None and self._current_utterance_id is not None:
                    self._tts.cancel(self._current_utterance_id)
            self._speaker.flush()
            self._drain_event_types({"TTSChunkReadyEvent", "SpeechCompletedEvent"})

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
