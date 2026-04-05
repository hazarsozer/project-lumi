"""
Typed event definitions for Project Lumi's event-driven architecture.

All events are frozen dataclasses — immutable after construction.
This module imports ONLY from stdlib and numpy to prevent circular imports.

Usage:
    from src.core.events import WakeDetectedEvent, ShutdownEvent
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        import numpy as np

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

    The audio field holds a numpy ndarray. Typed as ``object`` for the
    same frozen-dataclass hash reason as RecordingCompleteEvent.
    """

    audio: object  # np.ndarray
    viseme: str
    duration_ms: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TTSChunkReadyEvent):
            return NotImplemented
        import numpy as np

        if isinstance(self.audio, np.ndarray) and isinstance(other.audio, np.ndarray):
            return bool(
                np.array_equal(self.audio, other.audio)
                and self.viseme == other.viseme
                and self.duration_ms == other.duration_ms
            )
        return (
            self.audio is other.audio
            and self.viseme == other.viseme
            and self.duration_ms == other.duration_ms
        )

    __hash__ = None  # type: ignore[assignment]


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
    payload: dict  # type: ignore[type-arg]
    timestamp: float
    version: str = "1.0"
