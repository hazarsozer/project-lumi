"""
Speech-to-text transcription for Project Lumi using faster-whisper.

This module sits in the middle of the audio pipeline:

    RecordingCompleteEvent → Scribe.transcribe() → TranscriptReadyEvent

Key public class:
    Scribe — wraps a faster-whisper WhisperModel instance. Accepts raw
    int16 numpy audio arrays from the Ears pipeline, normalises them to
    float32, and returns a plain text transcription string.

The Orchestrator is responsible for posting the resulting TranscriptReadyEvent
to the event queue after calling Scribe.transcribe().

Usage:
    from src.audio.scribe import Scribe
    scribe = Scribe(model_size="tiny.en")
    text = scribe.transcribe(audio_array)
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    """Result of a recognized voice command."""

    type: str  # "interrupt" | "volume_control"
    action: str | None = None  # "up" | "down" | "mute" | None


_INTERRUPT_RE = re.compile(r"\b(stop|cancel|never\s+mind)\b", re.IGNORECASE)
_VOL_UP_RE = re.compile(r"\bvolume\s+up\b", re.IGNORECASE)
_VOL_DOWN_RE = re.compile(r"\bvolume\s+down\b", re.IGNORECASE)
_MUTE_RE = re.compile(r"\bmute\b", re.IGNORECASE)


def parse_command(text: str) -> CommandResult | None:
    """Match text against known voice command patterns.

    Returns a CommandResult for recognized commands, or None for unrecognized input.
    Priority: interrupt is checked before volume controls.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if _INTERRUPT_RE.search(stripped):
        return CommandResult(type="interrupt")
    if _VOL_UP_RE.search(stripped):
        return CommandResult(type="volume_control", action="up")
    if _VOL_DOWN_RE.search(stripped):
        return CommandResult(type="volume_control", action="down")
    if _MUTE_RE.search(stripped):
        return CommandResult(type="volume_control", action="mute")
    return None


class Scribe:
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        initial_prompt: str = "Lumi, Firefox, browser, desktop assistant.",
    ):
        """
        Scribe converts audio to text.
        Args:
            model_size: The size of the model to use.
            device: The device to use for inference.
            initial_prompt: Context injection for the transcription model.
        """
        self.initial_prompt = initial_prompt
        logger.info("Loading Whisper model: %s on %s...", model_size, device)

        # Load the model on int8 quantization
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        logger.info("Whisper model loaded successfully on %s.", device)

    def transcribe(self, audio_data, initial_prompt: str = None):
        """
        Transcribe the audio to text.
        Args:
            audio_data: The audio array to transcribe.
            initial_prompt: Optional override for the initial prompt context.
        Returns:
            The text transcription.
        """
        # faster-whisper expects float32, but mic gives us int16
        # we normalize it to -1.0 to 1.0
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Use provided prompt or default
        prompt = initial_prompt if initial_prompt is not None else self.initial_prompt

        # Transcribe the audio
        segments, info = self.model.transcribe(
            audio_data, beam_size=5, initial_prompt=prompt
        )

        # Combine segments into a single text string
        text = " ".join([segment.text for segment in segments])
        return text.strip()
