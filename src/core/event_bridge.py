"""
EventBridge: event translation bridge between Lumi's Python Brain and the
Tauri/React frontend.

This module is NOT a ZeroMQ server — it sits on top of ``WSTransport``
(a WebSocket server) and provides the event-semantic layer: it translates
outbound internal events into JSON wire frames and translates inbound JSON
frames into internal event dataclasses posted to the orchestrator's queue.

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
- ``_on_raw_message`` is called from WSTransport's asyncio thread.
  It decodes the frame, validates it, and posts events to the event queue.
  ``queue.Queue`` is thread-safe; no additional locking is required here.

Constraints:
- No pyzmq — transport is WSTransport (WebSocket).
- No print() — all output via logging.getLogger(__name__).
"""

from __future__ import annotations

import json
import logging
import queue
import time
from collections.abc import Callable
from typing import Any

from src.core.config import IPCConfig
from src.core.events import (
    ConfigSchemaRequestEvent,
    ConfigUpdateEvent,
    InterruptEvent,
    InterruptSource,
    LLMResponseReadyEvent,
    LLMTokenEvent,
    RAGRetrievalEvent,
    RAGSetEnabledEvent,
    RAGStatusEvent,
    RAGStatusRequestEvent,
    SpeechCompletedEvent,
    SystemStatusEvent,
    TranscriptReadyEvent,
    UserTextEvent,
    VisemeEvent,
    ZMQMessage,
)
from src.core.handshake import HandshakeHandler
from src.core.ws_transport import WSTransport
from src.core.state_machine import LumiState, StateMachine

logger = logging.getLogger(__name__)

__all__ = ["EventBridge"]

# Maps each outbound event type to its wire event name.
# on_state_change, on_error, send_config_schema, and send_config_update_result
# have custom signatures and are intentionally excluded.
_OUTBOUND: dict[type, str] = {
    LLMResponseReadyEvent: "tts_start",
    VisemeEvent:           "tts_viseme",
    SpeechCompletedEvent:  "tts_stop",
    TranscriptReadyEvent:  "transcript",
    LLMTokenEvent:         "llm_token",
    RAGRetrievalEvent:     "rag_retrieval",
    RAGStatusEvent:        "rag_status",
    SystemStatusEvent:     "system_status",
}


class EventBridge:
    """Bridges internal Lumi events to/from the Tauri/React frontend over IPC.

    Translates outbound internal events → JSON frames sent via WSTransport.
    Translates inbound JSON frames → internal events posted to the event queue.
    Registers itself as a StateMachine observer for state_change forwarding.

    All public on_*() methods are safe to call from any thread.

    Args:
        config:        IPCConfig carrying address (plain host/IP) and port.
        event_queue:   The orchestrator's event queue. Inbound events are
                       posted here so the orchestrator's dispatch loop
                       processes them on the main thread.
        state_machine: The shared state machine; EventBridge registers itself
                       as an observer so every state transition is forwarded
                       to the frontend automatically.
    """

    def __init__(
        self,
        config: IPCConfig,
        event_queue: queue.Queue[Any],
        state_machine: StateMachine,
    ) -> None:
        self._event_queue = event_queue
        self._state_machine = state_machine

        host = config.address
        self._transport: WSTransport = WSTransport(host=host, port=config.port)

        # Wire capability handshake: on client connect, send hello and wait for
        # hello_ack before forwarding normal messages downstream.
        self._handshake: HandshakeHandler = HandshakeHandler(self._transport)
        self._handshake.set_downstream_callback(self._on_raw_message)
        self._transport.set_on_connect(self._handshake.on_client_connected)
        self._transport.set_on_message(self._handshake.on_message_received)

        # Register as a state observer so every transition is forwarded to Body.
        state_machine.register_observer(self.on_state_change)

        # Inbound dispatch: wire event name → handler method.
        # Add entries here when new inbound message types are introduced.
        self._inbound: dict[str, Callable[[dict[str, Any]], None]] = {
            "interrupt":             self._handle_interrupt,
            "user_text":             self._handle_user_text,
            "rag_set_enabled":       self._handle_rag_set_enabled,
            "rag_status_request":    self._handle_rag_status_request,
            "config_schema_request": self._handle_config_schema_request,
            "config_update":         self._handle_config_update,
        }

        logger.debug("EventBridge initialised — transport %s:%d", host, config.port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the WebSocket server and start the recv daemon thread. Idempotent."""
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
        """Forward a state transition to the frontend.

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

    def on_system_status(self, event: SystemStatusEvent) -> None:
        """Forward system capability flags to the frontend.

        Args:
            event: The status event describing which subsystems are available.
        """
        payload: dict[str, Any] = {
            "tts_available": event.tts_available,
            "rag_available": event.rag_available,
            "mic_available": event.mic_available,
            "llm_available": event.llm_available,
            "source": event.source,
            "setup_required": event.setup_required,
            "missing_items": list(event.missing_items),
        }
        self._send("system_status", payload)

    def on_error(self, code: str, message: str) -> None:
        """Forward an error notification to the Body.

        Args:
            code:    Short error identifier (e.g. ``"LLM_TIMEOUT"``).
            message: Human-readable description.
        """
        payload: dict[str, Any] = {"code": code, "message": message}
        self._send("error", payload)

    def send_config_schema(
        self,
        fields: dict[str, Any],
        current_values: dict[str, Any],
    ) -> None:
        """Send the full config schema and current values to the frontend.

        Called by the orchestrator (or a dedicated handler) after it receives
        a ``ConfigSchemaRequestEvent``.  The ``fields`` dict mirrors the
        structure of ``FIELD_META`` from ``config_schema.py``; ``current_values``
        maps the same dotted-path keys to their live values.

        Args:
            fields:         Schema metadata keyed by dotted config path.
            current_values: Current runtime values keyed by dotted config path.
        """
        payload: dict[str, Any] = {
            "fields": fields,
            "current_values": current_values,
        }
        self._send("config_schema", payload)

    def send_config_update_result(
        self,
        applied_live: list[str],
        pending_restart: list[str],
        errors: dict[str, str],
    ) -> None:
        """Send the result of a ``config_update`` back to the frontend.

        Called by the orchestrator after it processes a ``ConfigUpdateEvent``
        through ``ConfigManager.apply()``.

        Args:
            applied_live:    Keys applied immediately without a restart.
            pending_restart: Keys whose changes take effect after a restart.
            errors:          Dotted-path key → human-readable error for any
                             rejected change.
        """
        payload: dict[str, Any] = {
            "applied_live": applied_live,
            "pending_restart": pending_restart,
            "errors": errors,
        }
        self._send("config_update_result", payload)

    # ------------------------------------------------------------------
    # Inbound handling (Body → Brain)
    # ------------------------------------------------------------------

    def _on_raw_message(self, raw: bytes) -> None:
        """Decode an inbound frame and dispatch to the appropriate handler.

        Called from WSTransport's asyncio thread.  Must not raise —
        any parse or validation error is logged and silently dropped.

        Args:
            raw: Raw UTF-8 bytes of the JSON frame (length prefix stripped).
        """
        msg = self._decode(raw)
        if msg is None:
            return

        handler = self._inbound.get(msg.event)
        if handler is None:
            logger.warning(
                "EventBridge: received unknown inbound event %r; dropping.", msg.event
            )
            return
        handler(msg.payload)

    def _handle_interrupt(self, payload: dict[str, Any]) -> None:
        """Post an InterruptEvent to the orchestrator queue.

        Args:
            payload: Wire payload (empty dict for ``interrupt``).
        """
        self._event_queue.put(InterruptEvent(source=InterruptSource.ZMQ))
        logger.debug("EventBridge: posted InterruptEvent(source=InterruptSource.ZMQ) to queue.")

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

    def _handle_rag_status_request(self, payload: dict[str, Any]) -> None:  # noqa: ARG002
        """Post a RAGStatusRequestEvent to the orchestrator queue."""
        self._event_queue.put(RAGStatusRequestEvent())
        logger.debug("EventBridge: posted RAGStatusRequestEvent to queue.")

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

    def _handle_config_schema_request(self, payload: dict[str, Any]) -> None:
        """Post a ConfigSchemaRequestEvent to the orchestrator queue.

        The payload for ``config_schema_request`` is always an empty dict;
        no validation is required — the event carries no data.

        Args:
            payload: Wire payload (ignored; event carries no parameters).
        """
        self._event_queue.put(ConfigSchemaRequestEvent())
        logger.debug("EventBridge: posted ConfigSchemaRequestEvent to queue.")

    # Fields that must never be changed over the IPC wire.  Allowing a
    # compromised frontend client to mutate these could redirect the IPC socket
    # to listen on 0.0.0.0 (network-exposed) or change the port in ways that
    # break the local-only security boundary.  These fields require a deliberate
    # edit of config.yaml by the user, not a remote wire command.
    _WIRE_BLOCKED_KEYS: frozenset[str] = frozenset(
        [
            "ipc.address",
            "ipc.port",
            "ipc.enabled",
        ]
    )

    def _handle_config_update(self, payload: dict[str, Any]) -> None:
        """Validate and post a ConfigUpdateEvent to the orchestrator queue.

        Validation rules:
        - ``changes`` must be present and must be a ``dict``.
        - ``persist`` must be present and must be a ``bool``.
        - ``changes`` must not contain IPC binding keys (``ipc.address``,
          ``ipc.port``, ``ipc.enabled``).  Allowing those over the wire would
          let a compromised client open the IPC socket to the network.

        Args:
            payload: Wire payload; must contain ``"changes"`` (dict) and
                     ``"persist"`` (bool).
        """
        changes = payload.get("changes")
        persist = payload.get("persist", False)

        if not isinstance(changes, dict):
            logger.warning(
                "EventBridge: config_update payload missing or invalid 'changes'; "
                "dropping. payload=%r",
                payload,
            )
            return

        if not isinstance(persist, bool):
            logger.warning(
                "EventBridge: config_update 'persist' must be bool; "
                "dropping. payload=%r",
                payload,
            )
            return

        # Reject any attempt to mutate IPC binding fields over the wire.
        blocked = self._WIRE_BLOCKED_KEYS.intersection(changes.keys())
        if blocked:
            logger.warning(
                "EventBridge: config_update attempted to mutate restricted IPC "
                "field(s) %r over the wire; dropping entire request.",
                sorted(blocked),
            )
            return

        self._event_queue.put(ConfigUpdateEvent(changes=changes, persist=persist))
        logger.debug(
            "EventBridge: posted ConfigUpdateEvent(persist=%s) to queue.", persist
        )

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
            UTF-8 encoded JSON bytes delivered as a single WebSocket message
            via ``WSTransport.send()``.
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
            raw: Raw bytes received from WSTransport.

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

        Delegates to ``WSTransport.send()`` which is thread-safe and drops
        the message silently (logs DEBUG) when no client is connected.

        Args:
            event:   Wire event name.
            payload: JSON-serialisable payload dict.
        """
        frame = self._encode(event, payload)
        self._transport.send(frame)
