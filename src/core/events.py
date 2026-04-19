"""
Typed event definitions for Project Lumi's event-driven architecture.

All events are frozen dataclasses — immutable after construction.
This module imports ONLY from stdlib and numpy to prevent circular imports.

Usage:
    from src.core.events import WakeDetectedEvent, ShutdownEvent
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WakeDetectedEvent:
    """Fired by Ears when the wake word is detected."""

    timestamp: float  # time.monotonic() at detection


@dataclass(frozen=True)
class RecordingCompleteEvent:
    """Fired by Ears when VAD recording finishes.

    The audio field holds a numpy ndarray (16 kHz int16 mono).
    Typed as ``object`` to avoid hash issues with frozen dataclasses
    containing numpy arrays.
    """

    audio: object  # np.ndarray — 16 kHz int16 mono

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordingCompleteEvent):
            return NotImplemented
        if isinstance(self.audio, np.ndarray) and isinstance(other.audio, np.ndarray):
            return bool(np.array_equal(self.audio, other.audio))
        return self.audio is other.audio

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TranscriptReadyEvent:
    """Fired by Scribe after STT transcription completes."""

    text: str


@dataclass(frozen=True)
class CommandResultEvent:
    """Fired when a local command is parsed from the transcript."""

    command_type: str  # "interrupt" | "volume_control" | etc.


@dataclass(frozen=True)
class LLMResponseReadyEvent:
    """Fired by the LLM engine when a response is generated."""

    text: str


@dataclass(frozen=True)
class TTSChunkReadyEvent:
    """Fired by the TTS engine for each audio chunk to play.

    audio holds a numpy ndarray (float32, mono, at TTS sample rate).
    Typed as ``object`` for the same frozen-dataclass hash reason as
    RecordingCompleteEvent.
    """

    audio: object        # np.ndarray float32 mono
    sample_rate: int     # Hz — e.g. 24000 for Kokoro
    chunk_id: int        # monotonically increasing within one utterance
    is_final: bool       # True on the last chunk of an utterance
    utterance_id: str    # identifies the utterance; used for interrupt-triggered drains

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TTSChunkReadyEvent):
            return NotImplemented
        if isinstance(self.audio, np.ndarray) and isinstance(other.audio, np.ndarray):
            return bool(
                np.array_equal(self.audio, other.audio)
                and self.sample_rate == other.sample_rate
                and self.chunk_id == other.chunk_id
                and self.is_final == other.is_final
                and self.utterance_id == other.utterance_id
            )
        return (
            self.audio is other.audio
            and self.sample_rate == other.sample_rate
            and self.chunk_id == other.chunk_id
            and self.is_final == other.is_final
            and self.utterance_id == other.utterance_id
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class VisemeEvent:
    """Fired by the TTS engine for each phoneme/viseme in an utterance.

    Used by the Godot frontend to animate avatar lip-sync.
    """

    utterance_id: str  # binds this viseme to its utterance
    phoneme: str       # IPA or ARPAbet phoneme string
    start_ms: int      # offset from utterance start, milliseconds
    duration_ms: int   # phoneme duration, milliseconds


@dataclass(frozen=True)
class SpeechCompletedEvent:
    """Fired by the speaker thread after the last TTS chunk finishes playing.

    Signals the Orchestrator to transition from SPEAKING back to IDLE.
    """

    utterance_id: str


@dataclass(frozen=True)
class LLMTokenEvent:
    """Fired by ReasoningRouter for each generated token (streaming display).

    Used by the Godot frontend to show a typing indicator or live transcript.
    Not consumed by the Orchestrator's main dispatch loop.
    """

    token: str
    utterance_id: str


@dataclass(frozen=True)
class InterruptEvent:
    """Fired to cancel in-flight work and return to IDLE."""

    source: str  # "zmq" | "wake_word" | "keyboard" | "user_stop"


@dataclass(frozen=True)
class ShutdownEvent:
    """Fired to terminate the orchestrator and all worker threads."""

    pass


@dataclass(frozen=True)
class UserTextEvent:
    """Typed text input from the Body (Godot frontend) via ZMQ."""

    text: str


@dataclass(frozen=True)
class ZMQMessage:
    """Wire-format message for ZMQ IPC communication."""

    event: str
    payload: dict[str, object]
    timestamp: float
    version: str = "1.0"


@dataclass(frozen=True)
class RAGRetrievalEvent:
    """Fired after RAG retrieval completes (for ZMQ forwarding and logging)."""

    query: str
    hit_count: int
    latency_ms: int
    top_doc_paths: tuple[str, ...]  # up to retrieval_top_k paths, for display


@dataclass(frozen=True)
class RAGStatusEvent:
    """Fired in response to a status request; describes RAG runtime state."""

    enabled: bool
    doc_count: int
    chunk_count: int
    last_indexed: str  # ISO-8601 timestamp or "" if never indexed


@dataclass(frozen=True)
class RAGSetEnabledEvent:
    """Fired from ZMQ layer to toggle RAG on/off at runtime without restart."""

    enabled: bool


@dataclass(frozen=True)
class EarsErrorEvent:
    """Fired by Ears when the audio capture thread fails unrecoverably.

    code follows the namespace pattern ``ears.<reason>``:
      - ``ears.unrecoverable`` — InputStream failed after all retries
    """

    code: str    # e.g. "ears.unrecoverable"
    detail: str  # human-readable description for logs / UI toast
