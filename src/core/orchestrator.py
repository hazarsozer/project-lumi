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
import uuid
from typing import Any, Callable

from src.audio.ears import Ears
from src.audio.mouth import KokoroTTS
from src.audio.scribe import Scribe
from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig
from src.core.events import (
    EarsErrorEvent,
    InterruptEvent,
    LLMResponseReadyEvent,
    LLMTokenEvent,
    RAGRetrievalEvent,
    RAGSetEnabledEvent,
    RecordingCompleteEvent,
    ShutdownEvent,
    SpeechCompletedEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    VisemeEvent,
    WakeDetectedEvent,
)
from src.llm.tool_call_parser import parse_tool_calls
from src.tools import ToolExecutor, ToolRegistry
from src.tools.os_actions import AppLaunchTool, ClipboardTool, FileInfoTool, WindowListTool
from src.core.state_machine import LumiState, StateMachine
from src.core.zmq_server import ZMQServer
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

    def __init__(
        self,
        config: LumiConfig,
        *,
        speaker: SpeakerThread | None = None,
        tts: KokoroTTS | None = None,
        zmq_server: ZMQServer | None = None,
        ears: Ears | None = None,
        scribe: Scribe | None = None,
    ) -> None:
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
        # RAG subsystem — built only when enabled in config.
        self._rag_runtime_enabled: bool = config.rag.enabled
        self._rag_retriever = None
        if config.rag.enabled:
            try:
                from src.rag.store import DocumentStore  # noqa: PLC0415
                from src.rag.retriever import RAGRetriever  # noqa: PLC0415
                _rag_store = DocumentStore(config.rag)
                self._rag_retriever = RAGRetriever(_rag_store, config.rag)
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

        # Tool registry and executor — wired when tools are enabled.
        self._tool_registry: ToolRegistry = ToolRegistry()
        if config.tools.enabled:
            self._tool_registry.register(AppLaunchTool())
            self._tool_registry.register(ClipboardTool())
            self._tool_registry.register(FileInfoTool())
            self._tool_registry.register(WindowListTool())

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

        # Speaker output thread — injectable for testing; created here otherwise.
        self._speaker: SpeakerThread = (
            speaker if speaker is not None else SpeakerThread(self._event_queue)
        )
        self._speaker.start()

        # TTS engine — injectable for testing; None means no TTS (state machine
        # still transitions correctly via a synthetic SpeechCompletedEvent).
        self._tts: KokoroTTS | None = tts

        # Guards _current_utterance_id so _handle_interrupt can atomically
        # read and cancel the active utterance.
        self._tts_state_lock: threading.Lock = threading.Lock()
        self._current_utterance_id: str | None = None

        # Audio-in pipeline — both are optional (None = text-only mode / testing).
        self._ears: Ears | None = ears
        self._scribe: Scribe | None = scribe

        # Register built-in handlers.
        # NOTE: TranscriptReadyEvent and SpeechCompletedEvent each receive a
        # second handler below (on_transcript / on_tts_stop) when a ZMQServer
        # is present.  Both registrations are intentional: the internal handler
        # runs first (state transitions), then the ZMQ forwarder sends the event
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

        # ZMQServer wiring — optional; injected for testing or when IPC is
        # enabled.  If not injected but config.ipc.enabled is True, create it
        # here using the orchestrator's own queue and state machine so that
        # inbound events from the Godot frontend are posted to this event loop.
        # When created internally, ZMQServer.__init__ registers on_state_change
        # as a state observer.  When injected (e.g. in tests), the caller is
        # responsible for ensuring the state machine is shared — the Orchestrator
        # registers on_state_change explicitly so injected instances also receive
        # state transition forwarding.
        self._zmq_server: ZMQServer | None = zmq_server
        if self._zmq_server is None and config.ipc.enabled:
            self._zmq_server = ZMQServer(
                config.ipc, self._event_queue, self._state_machine
            )
            self._zmq_server.start()

        if self._zmq_server is not None:
            # Injected instances have not had on_state_change registered against
            # this orchestrator's state machine; do it here.  For auto-created
            # instances ZMQServer.__init__ already registered, so we avoid
            # double registration by only registering for the injected path.
            if zmq_server is not None:
                self._state_machine.register_observer(
                    self._zmq_server.on_state_change
                )
            self.register_handler(VisemeEvent, self._zmq_server.on_tts_viseme)
            self.register_handler(SpeechCompletedEvent, self._zmq_server.on_tts_stop)
            self.register_handler(TranscriptReadyEvent, self._zmq_server.on_transcript)
            self.register_handler(LLMResponseReadyEvent, self._zmq_server.on_tts_start)
            self.register_handler(LLMTokenEvent, self._zmq_server.on_llm_token)
            self.register_handler(RAGRetrievalEvent, self._zmq_server.on_rag_retrieval)

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
        Starts the Ears audio pipeline (if configured) before entering the loop.
        """
        logger.info("Orchestrator starting event loop")
        if self._ears is not None:
            self._ears.start(self._event_queue)

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

        # Generate utterance_id here so the ZMQ token handler can correlate events.
        utterance_id = str(uuid.uuid4())

        # RAG intent check — only when RAG is runtime-enabled.
        use_rag = (
            self._rag_runtime_enabled
            and self._reflex_router.route_rag_intent(event.text)
        )

        # Reasoning slow-path — run in a daemon thread so the event loop
        # remains responsive to InterruptEvents during long inference.
        def _run_inference() -> None:
            try:
                response = self._reasoning_router.generate(
                    event.text, self._llm_cancel_flag,
                    utterance_id=utterance_id, use_rag=use_rag,
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

            # Tool-call pass: if the LLM emitted tool-call blocks, execute them
            # and feed the results back for a second inference pass (single-shot only).
            tool_calls = parse_tool_calls(response)
            if tool_calls:
                results = self._tool_executor.execute(tool_calls, self._llm_cancel_flag)
                result_lines = []
                for tc, tr in zip(tool_calls, results):
                    result_lines.append(
                        f"Tool {tc['tool']!r}: {'OK' if tr.success else 'FAIL'} — {tr.output}"
                    )
                tool_context = "\n".join(result_lines)
                followup_prompt = f"{event.text}\n\n[Tool results]\n{tool_context}"
                try:
                    response = self._reasoning_router.generate(
                        followup_prompt, self._llm_cancel_flag, utterance_id=utterance_id
                    )
                except InterruptedError:
                    logger.info("LLM tool-followup cancelled for %r", event.text)
                    return
                except Exception:
                    logger.exception("LLM tool-followup failed for %r", event.text)
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

                try:
                    self._memory.save()
                    self.post_event(LLMResponseReadyEvent(text=response))
                    self._state_machine.transition_to(LumiState.SPEAKING)
                except Exception:
                    logger.exception(
                        "Post-inference save/dispatch failed for %r; returning to IDLE",
                        event.text,
                    )
                    self._state_machine.transition_to(LumiState.IDLE)

        thread = threading.Thread(target=_run_inference, daemon=True)
        thread.start()

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

        Fast-path: try ReflexRouter first (regex, no model needed).
        Slow-path: dispatch ReasoningRouter in a daemon thread so the
        event loop remains unblocked during inference.

        Args:
            event: The user-text event containing text from the frontend.
        """
        current = self._state_machine.current_state
        if current not in (LumiState.IDLE, LumiState.LISTENING):
            logger.debug(
                "UserTextEvent received in state %s; dropping.", current.value
            )
            return

        # Text input bypasses the wake-word pipeline.  If we are IDLE, step
        # through LISTENING first so the LISTENING→PROCESSING transition below
        # is always valid.
        if current == LumiState.IDLE:
            self._state_machine.transition_to(LumiState.LISTENING)

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

        # Generate utterance_id here so the ZMQ token handler can correlate events.
        utterance_id = str(uuid.uuid4())

        # RAG intent check — only when RAG is runtime-enabled.
        use_rag = (
            self._rag_runtime_enabled
            and self._reflex_router.route_rag_intent(event.text)
        )

        # Reasoning slow-path — run in a daemon thread so the event loop
        # remains responsive to InterruptEvents during long inference.
        def _run_inference() -> None:
            try:
                response = self._reasoning_router.generate(
                    event.text, self._llm_cancel_flag,
                    utterance_id=utterance_id, use_rag=use_rag,
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

            # Tool-call pass: if the LLM emitted tool-call blocks, execute them
            # and feed the results back for a second inference pass (single-shot only).
            tool_calls = parse_tool_calls(response)
            if tool_calls:
                results = self._tool_executor.execute(tool_calls, self._llm_cancel_flag)
                result_lines = []
                for tc, tr in zip(tool_calls, results):
                    result_lines.append(
                        f"Tool {tc['tool']!r}: {'OK' if tr.success else 'FAIL'} — {tr.output}"
                    )
                tool_context = "\n".join(result_lines)
                followup_prompt = f"{event.text}\n\n[Tool results]\n{tool_context}"
                try:
                    response = self._reasoning_router.generate(
                        followup_prompt, self._llm_cancel_flag, utterance_id=utterance_id
                    )
                except InterruptedError:
                    logger.info("LLM tool-followup cancelled for %r", event.text)
                    return
                except Exception:
                    logger.exception("LLM tool-followup failed for %r", event.text)
                    with self._llm_state_lock:
                        if self._state_machine.current_state == LumiState.PROCESSING:
                            self._state_machine.transition_to(LumiState.IDLE)
                    return

            with self._llm_state_lock:
                if self._state_machine.current_state != LumiState.PROCESSING:
                    logger.debug(
                        "State changed during inference, discarding response for %r",
                        event.text,
                    )
                    return

                try:
                    self._memory.save()
                    self.post_event(LLMResponseReadyEvent(text=response))
                    self._state_machine.transition_to(LumiState.SPEAKING)
                except Exception:
                    logger.exception(
                        "Post-inference save/dispatch failed for %r; returning to IDLE",
                        event.text,
                    )
                    self._state_machine.transition_to(LumiState.IDLE)

        thread = threading.Thread(target=_run_inference, daemon=True)
        thread.start()

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

        # Prepare KokoroTTS BEFORE starting the thread so that a cancel() call
        # arriving between thread start and synthesize() executing its first lock
        # acquisition can correctly target the utterance (see KokoroTTS.prepare()).
        with self._tts_state_lock:
            self._current_utterance_id = utterance_id
            if tts is not None:
                tts.prepare(utterance_id)

        if tts is None:
            # No TTS engine configured — fire completion immediately.
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

        Args:
            event: The speech-completion event carrying the utterance identifier.
        """
        current = self._state_machine.current_state
        if current == LumiState.SPEAKING:
            with self._tts_state_lock:
                self._current_utterance_id = None
            self._state_machine.transition_to(LumiState.IDLE)
            logger.info(
                "Speech completed (utterance_id=%s), returning to IDLE",
                event.utterance_id,
            )
        else:
            logger.debug(
                "SpeechCompletedEvent received in state %s, ignoring",
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

    def _handle_wake_detected(self, event: WakeDetectedEvent) -> None:
        """Handle WakeDetectedEvent: transition IDLE → LISTENING.

        Events arriving in any state other than IDLE are silently dropped so
        that spurious detections during active speech or processing do not
        corrupt the state machine.

        Args:
            event: The wake-word detection event posted by Ears.
        """
        current = self._state_machine.current_state
        if current != LumiState.IDLE:
            logger.debug(
                "WakeDetectedEvent received in state %s; ignoring.", current.value
            )
            return
        self._state_machine.transition_to(LumiState.LISTENING)
        logger.info("Wake word detected — transitioning to LISTENING")

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
            logger.warning("RecordingCompleteEvent received but no Scribe configured; returning to IDLE")
            self._state_machine.transition_to(LumiState.IDLE)
            return

        thread = threading.Thread(
            target=self._run_scribe,
            args=(event.audio,),
            daemon=True,
            name="ScribeTranscribeThread",
        )
        thread.start()

    def _run_scribe(self, audio: object) -> None:
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
        """Handle EarsErrorEvent: log, surface to Godot, and return to IDLE.

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

    def _handle_shutdown(self, event: ShutdownEvent) -> None:
        """Handle ShutdownEvent: stop the event loop.

        Args:
            event: The shutdown event.
        """
        logger.info("Shutdown requested")
        self._speaker.stop()
        if self._ears is not None:
            self._ears.stop()
        if self._zmq_server is not None:
            self._zmq_server.stop()
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
