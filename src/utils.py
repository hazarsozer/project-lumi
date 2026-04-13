"""
Shared utility functions for Project Lumi.

Key public functions:
    play_ready_sound(speaker) — generates a short 880 Hz sine-wave "ping"
    and enqueues it onto the SpeakerThread for non-blocking playback.
"""

from __future__ import annotations

import logging

import numpy as np

from src.audio.speaker import SpeakerThread

logger = logging.getLogger(__name__)

# Canonical sample rate used throughout the audio pipeline.
_SAMPLE_RATE: int = 24_000


def play_ready_sound(speaker: SpeakerThread) -> None:
    """Enqueue a short 880 Hz ping onto the speaker thread (non-blocking).

    Generates a 0.2-second sine wave at 880 Hz, applies a linear fade-out to
    avoid a click at the end, and enqueues the chunk on *speaker*.

    Args:
        speaker: The active :class:`~src.audio.speaker.SpeakerThread` instance.
    """
    duration = 0.2  # seconds
    frequency = 880  # Hz — high-pitch "ping"

    t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), endpoint=False)
    # Sine wave with linear fade-out to avoid a DC-offset click at the tail.
    fade = np.linspace(1.0, 0.0, len(t), dtype=np.float32)
    audio = (np.sin(frequency * t * 2.0 * np.pi) * 0.5 * fade).astype(np.float32)

    try:
        speaker.enqueue(audio, utterance_id="ready-sound", is_final=True)
    except Exception:
        logger.warning("Could not enqueue ready sound", exc_info=True)
