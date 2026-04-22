"""
EventBridge: event translation bridge between Lumi's Python Brain and the
Godot 4 Body.

This module is NOT a ZeroMQ server — it sits on top of ``IPCTransport``
(a raw TCP length-prefixed server) and provides the event-semantic layer:
it translates outbound internal events into JSON wire frames and translates
inbound JSON frames into internal event dataclasses posted to the
orchestrator's queue.

Wire format (JSON envelope):
    {
        "event":     "<event_name>",
        "payload":   { ... },
        "timestamp": 1234567890.0,
        "version":   "1.0"
    }

Threading model:
- ``publish``-side methods (on_state_change, on_tts_start, …) are called
  from the orchestrator thread.  They call ``_transport.send()`` which is
  thread-safe.
- ``_on_raw_message`` is called from IPCTransport's recv daemon thread.
  It decodes the frame, validates it, and posts events to the event queue.
  ``queue.Queue`` is thread-safe; no additional locking is required here.

Constraints:
- No asyncio — stdlib threads + queue.Queue only.
- No pyzmq — transport is IPCTransport (raw TCP).
- No print() — all output via logging.getLogger(__name__).
"""

from __future__ import annotations

import json
import logging
import queue
import time
from typing import Any

from src.core.config import IPCConfig
from src.core.events import (
    InterruptEvent,
    LLMResponseReadyEvent,
    LLMTokenEvent,
    RAGRetrievalEvent,
    RAGSetEnabledEvent,
    RAGStatusEvent,
    SpeechCompletedEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    VisemeEvent,
    ZMQMessage,
)
from src.core.ipc_transport import IPCTransport
from src.core.state_machine import LumiState, StateMachine

logger = logging.getLogger(__name__)

# The "tcp://" scheme prefix used in IPCConfig.address; stripped before
# passing the host string to IPCTransport.
_TCP_SCHEME: str = "tcp://"

__all__ = ["EventBridge"]


class EventBridge:
    """Bridges internal Lumi events to/from the Godot frontend over IPC.

    Translates outbound internal events → JSON frames sent via IPCTransport.
    Translates inbound JSON frames → internal events posted to the event queue.
    Registers itself as a StateMachine observer for state_change forwarding.

    All public on_*() methods are safe to call from any thread.

    Args:
        config:        IPCConfig carrying address (with scheme prefix) and port.
        event_queue:   The orchestrator's event queue. Inbound events are
                       posted here so the orchestrator's dispatch loop
                       processes them on the main thread.
        state_machine: The shared state machine; EventBridge registers itself
                       as an observer so every state transition is forwarded
                       to the Godot frontend automatically.
    """

    def __init__(
        self,
        config: IPCConfig,
        event_queue: queue.Queue[Any],
        state_machine: StateMachine,
    ) -> None:
        self._event_queue = event_queue
        self._state_machine = state_machine

        # Strip the "tcp://" scheme prefix that IPCConfig carries — IPCTransport
        # takes a plain hostname/IP string, not a ZMQ-style URI.
        host = config.address
        if host.startswith(_TCP_SCHEME):
            host = host[len(_TCP_SCHEME):]

        self._transport: IPCTransport = IPCTransport(host=host, port=config.port)
        self._transport.set_on_message(self._on_raw_message)

        # Register as a state observer so every transition is forwarded to Body.
        state_machine.register_observer(self.on_state_change)

        logger.debug(
            "EventBridge initialised — transport %s:%d", host, config.port
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the TCP socket and start the recv daemon thread. Idempotent."""
        logger.info("EventBridge starting transport.")
        self._transport.start()

    @property
    def bound_port(self) -> int | None:
        """Return the actual port the transport is bound to, or None if not started.

        Useful when IPCConfig.port is 0 (OS-assigned): start() the server,
        then read bound_port to discover the assigned port.
        """
        return self._transport.bound_port

    def stop(self) -> None:
        """Stop the transport and unregister the state observer.

        Safe to call before start() or after a previous stop().
        """
        logger.info("EventBridge stopping transport.")
        self._transport.stop()

        self._state_machine.unregister_observer(self.on_state_change)

    # ------------------------------------------------------------------
    # Outbound handlers (Brain → Body)
    # ------------------------------------------------------------------

    def on_state_change(self, old_state: LumiState, new_state: LumiState) -> None:
        """Forward a state transition to the Godot frontend.

        Called by StateMachine after every successful transition.
        Runs on whichever thread called ``transition_to()``.

        Args:
            old_state: The state before the transition (unused on the wire).
            new_state: The new state to report.
        """
        payload: dict[str, Any] = {"state": new_state.value}
        self._send("state_change", payload)

    def on_tts_start(self, event: LLMResponseReadyEvent) -> None:
        """Notify the Body that TTS synthesis is starting.

        Args:
            event: The LLM response event whose text is about to be spoken.
                   ``duration_ms`` is sent as 0 because the actual audio
                   duration is not known until synthesis completes.
        """
        payload: dict[str, Any] = {"text": event.text, "duration_ms": 0}
        self._send("tts_start", payload)

    def on_tts_viseme(self, event: VisemeEvent) -> None:
        """Forward a viseme/phoneme timing event to the Body for lip-sync.

        Args:
            event: The viseme event from the TTS engine.  ``event.phoneme``
                   is mapped to the wire-protocol field ``viseme``.
        """
        payload: dict[str, Any] = {
            "viseme": event.phoneme,
            "duration_ms": event.duration_ms,
        }
        self._send("tts_viseme", payload)

    def on_tts_stop(self, event: SpeechCompletedEvent) -> None:
        """Notify the Body that TTS playback has finished.

        Args:
            event: The speech-completion event (utterance_id not forwarded
                   on the wire; the Body only needs the signal).
        """
        self._send("tts_stop", {})

    def on_transcript(self, event: TranscriptReadyEvent) -> None:
        """Send the STT transcript to the Body for display.

        Args:
            event: The transcript event with the recognised text.
        """
        payload: dict[str, Any] = {"text": event.text}
        self._send("transcript", payload)

    def on_llm_token(self, event: LLMTokenEvent) -> None:
        """Forward a streaming LLM token to the Body for live display.

        Args:
            event: The token event with token string and utterance_id.
        """
        payload: dict[str, Any] = {
            "token": event.token,
            "utterance_id": event.utterance_id,
        }
        self._send("llm_token", payload)

    def on_rag_retrieval(self, event: RAGRetrievalEvent) -> None:
        """Forward a RAG retrieval result to the Body for display.

        Args:
            event: The retrieval event with query, hit count, latency, and paths.
        """
        payload: dict[str, Any] = {
            "query": event.query,
            "hit_count": event.hit_count,
            "latency_ms": event.latency_ms,
            "top_doc_paths": list(event.top_doc_paths),
        }
        self._send("rag_retrieval", payload)

    def on_rag_status(self, event: RAGStatusEvent) -> None:
        """Forward the RAG status response to the Body.

        Args:
            event: The status event describing current RAG state.
        """
        payload: dict[str, Any] = {
            "enabled": event.enabled,
            "doc_count": event.doc_count,
            "chunk_count": event.chunk_count,
            "last_indexed": event.last_indexed,
        }
        self._send("rag_status", payload)

    def on_error(self, code: str, message: str) -> None:
        """Forward an error notification to the Body.

        Args:
            code:    Short error identifier (e.g. ``"LLM_TIMEOUT"``).
            message: Human-readable description.
        """
        payload: dict[str, Any] = {"code": code, "message": message}
        self._send("error", payload)

    # ------------------------------------------------------------------
    # Inbound handling (Body → Brain)
    # ------------------------------------------------------------------

    def _on_raw_message(self, raw: bytes) -> None:
        """Decode an inbound frame and dispatch to the appropriate handler.

        Called from IPCTransport's recv daemon thread.  Must not raise —
        any parse or validation error is logged and silently dropped.

        Args:
            raw: Raw UTF-8 bytes of the JSON frame (length prefix stripped).
        """
        msg = self._decode(raw)
        if msg is None:
            return

        event_name = msg.event

        if event_name == "interrupt":
            self._handle_interrupt(msg.payload)
        elif event_name == "user_text":
            self._handle_user_text(msg.payload)
        elif event_name == "rag_set_enabled":
            self._handle_rag_set_enabled(msg.payload)
        else:
            logger.warning(
                "EventBridge: received unknown inbound event %r; dropping.", event_name
            )

    def _handle_interrupt(self, payload: dict[str, Any]) -> None:
        """Post an InterruptEvent to the orchestrator queue.

        Args:
            payload: Wire payload (empty dict for ``interrupt``).
        """
        self._event_queue.put(InterruptEvent(source="zmq"))
        logger.debug("EventBridge: posted InterruptEvent(source='zmq') to queue.")

    def _handle_rag_set_enabled(self, payload: dict[str, Any]) -> None:
        """Post a RAGSetEnabledEvent to the orchestrator queue.

        Args:
            payload: Wire payload; must contain a boolean ``"enabled"`` key.
        """
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            logger.warning(
                "EventBridge: rag_set_enabled payload missing or invalid 'enabled' field; "
                "dropping. payload=%r",
                payload,
            )
            return

        self._event_queue.put(RAGSetEnabledEvent(enabled=enabled))
        logger.debug(
            "EventBridge: posted RAGSetEnabledEvent(enabled=%s) to queue.", enabled
        )

    def _handle_user_text(self, payload: dict[str, Any]) -> None:
        """Validate and post a UserTextEvent to the orchestrator queue.

        Args:
            payload: Wire payload; must contain a non-empty ``"text"`` key.
        """
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            logger.warning(
                "EventBridge: user_text payload missing or empty 'text' field; "
                "dropping. payload=%r",
                payload,
            )
            return

        self._event_queue.put(UserTextEvent(text=text))
        logger.debug("EventBridge: posted UserTextEvent(text=%r) to queue.", text)

    # ------------------------------------------------------------------
    # Wire encoding / decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(event: str, payload: dict[str, Any]) -> bytes:
        """Serialise an event + payload into a UTF-8 JSON wire frame.

        Constructs a ``ZMQMessage`` dataclass to enforce field presence, then
        serialises it with ``json.dumps``.  ``ensure_ascii=False`` preserves
        non-ASCII characters in user text.

        Args:
            event:   Event name string (e.g. ``"state_change"``).
            payload: Arbitrary JSON-serialisable dict.

        Returns:
            UTF-8 encoded JSON bytes (no length prefix — that is added by
            ``IPCTransport.send()``).
        """
        msg = ZMQMessage(
            event=event,
            payload=payload,
            timestamp=time.time(),
            version="1.0",
        )
        data = {
            "event": msg.event,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "version": msg.version,
        }
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _decode(raw: bytes) -> ZMQMessage | None:
        """Parse and validate a raw JSON frame from the wire.

        Validation rules:
        - Must be valid UTF-8 JSON.
        - Top-level value must be a ``dict``.
        - ``event`` must be a ``str``.
        - ``payload`` must be a ``dict``.
        - ``timestamp`` must be an ``int`` or ``float``.
        - ``version`` must be a ``str``.

        Args:
            raw: Raw bytes received from IPCTransport.

        Returns:
            A ``ZMQMessage`` on success, or ``None`` if parsing or
            validation fails.
        """
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.warning("EventBridge: inbound frame is not valid UTF-8: %s", exc)
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("EventBridge: inbound frame is not valid JSON: %s", exc)
            return None

        if not isinstance(data, dict):
            logger.warning(
                "EventBridge: inbound message is not a JSON object (got %s); dropping.",
                type(data).__name__,
            )
            return None

        event = data.get("event")
        payload = data.get("payload")
        timestamp = data.get("timestamp")
        version = data.get("version")

        if not isinstance(event, str):
            logger.warning(
                "EventBridge: inbound message missing or invalid 'event' field; dropping."
            )
            return None

        if not isinstance(payload, dict):
            logger.warning(
                "EventBridge: inbound message missing or invalid 'payload' field; dropping."
            )
            return None

        if not isinstance(timestamp, (int, float)):
            logger.warning(
                "EventBridge: inbound message missing or invalid 'timestamp' field; dropping."
            )
            return None

        if not isinstance(version, str):
            logger.warning(
                "EventBridge: inbound message missing or invalid 'version' field; dropping."
            )
            return None

        return ZMQMessage(
            event=event,
            payload=payload,
            timestamp=float(timestamp),
            version=version,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _send(self, event: str, payload: dict[str, Any]) -> None:
        """Encode and dispatch a frame via the transport.

        Delegates to ``IPCTransport.send()`` which is thread-safe and drops
        the message silently (logs DEBUG) when no client is connected.

        Args:
            event:   Wire event name.
            payload: JSON-serialisable payload dict.
        """
        frame = self._encode(event, payload)
        self._transport.send(frame)
