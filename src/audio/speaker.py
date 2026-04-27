"""
Speaker output thread for Project Lumi.

Owns the audio output stream and consumes audio chunks from an internal queue,
writing them through sounddevice.OutputStream.  When the last chunk of an
utterance plays, SpeechCompletedEvent is posted to the orchestrator event queue.

Usage:
    speaker = SpeakerThread(event_queue=orchestrator_queue)
    speaker.start()
    speaker.enqueue(audio_array, utterance_id="utt-1", is_final=True)
    speaker.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
from math import gcd
from typing import Any

import numpy as np
import sounddevice as sd

from src.core.events import SpeechCompletedEvent

logger = logging.getLogger(__name__)

# All audio is resampled to this rate before playback.
# Matches Kokoro-82M ONNX TTS output rate.
_CANONICAL_RATE: int = 24000

# Sentinel value enqueued to unblock and terminate the consumer loop.
_STOP_SENTINEL: None = None


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample *audio* from *source_rate* to *target_rate*.

    Prefers ``scipy.signal.resample_poly`` for high-quality audio resampling.
    Falls back to numpy linear interpolation when scipy is unavailable
    (acceptable for short startup sounds; not recommended for TTS audio).

    Args:
        audio: Input audio array (any dtype; converted to float32 internally).
        source_rate: Input sample rate in Hz.
        target_rate: Desired output sample rate in Hz.

    Returns:
        float32 numpy array resampled to *target_rate*.
    """
    audio_f32 = audio.astype(np.float32)
    if source_rate == target_rate:
        return audio_f32

    try:
        import scipy.signal  # optional heavy dependency

        g = gcd(source_rate, target_rate)
        return scipy.signal.resample_poly(  # type: ignore[no-any-return]
            audio_f32, target_rate // g, source_rate // g
        ).astype(np.float32)
    except ImportError:
        logger.debug(
            "scipy not available; using linear interpolation for %d→%d Hz resample",
            source_rate,
            target_rate,
        )
        new_length = int(round(len(audio_f32) * target_rate / source_rate))
        return np.interp(
            np.linspace(0, len(audio_f32) - 1, new_length),
            np.arange(len(audio_f32)),
            audio_f32.astype(np.float64),
        ).astype(np.float32)


class SpeakerThread:
    """Daemon thread that writes audio chunks to a sounddevice OutputStream.

    Enqueued chunks are played in arrival order.  When *is_final=True* on a
    chunk, a :class:`~src.core.events.SpeechCompletedEvent` is posted to the
    orchestrator event queue after that chunk finishes playing.

    If the audio device cannot be opened, the thread continues in *silent
    mode* — chunks are consumed and completion events are still fired, but
    no audio plays.  This prevents a missing audio device from crashing the
    entire assistant.

    Args:
        event_queue: Orchestrator event queue for posting
            :class:`~src.core.events.SpeechCompletedEvent`.
        sample_rate: Canonical output sample rate in Hz.  All input audio is
            resampled to this rate.  Defaults to 24 000 (Kokoro TTS rate).
    """

    def __init__(
        self,
        event_queue: queue.Queue[Any],
        sample_rate: int = _CANONICAL_RATE,
    ) -> None:
        self._event_queue = event_queue
        self._sample_rate = sample_rate
        # Queue items: (audio, utterance_id, is_final) | None (sentinel)
        self._queue: queue.Queue[tuple[np.ndarray, str, bool] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._playing = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SpeakerThread"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the consumer thread."""
        self._thread.start()
        logger.debug("SpeakerThread started (sample_rate=%d Hz)", self._sample_rate)

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 2 s for it to join."""
        self._stop_event.set()
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("SpeakerThread did not join within 2 s timeout")

    def enqueue(
        self,
        audio: np.ndarray,
        utterance_id: str,
        is_final: bool,
        source_rate: int = _CANONICAL_RATE,
    ) -> None:
        """Enqueue an audio chunk for playback.

        Resamples to the canonical rate when *source_rate* differs.

        Args:
            audio: float32 mono numpy array.
            utterance_id: Identifies the utterance this chunk belongs to.
            is_final: ``True`` on the last chunk of the utterance.
            source_rate: Sample rate of *audio* in Hz.
        """
        resampled = _resample(audio, source_rate, self._sample_rate)
        self._queue.put((resampled, utterance_id, is_final))

    def flush(self, utterance_id: str | None = None) -> None:
        """Drain pending chunks from the queue.

        Args:
            utterance_id: When given, discard only chunks whose
                ``utterance_id`` matches.  When ``None``, discard all
                pending chunks (the shutdown sentinel is always preserved).
        """
        retained: list[tuple[np.ndarray, str, bool] | None] = []
        try:
            while True:
                item = self._queue.get_nowait()
                if item is _STOP_SENTINEL:
                    # Always preserve the sentinel so the thread can shut down.
                    retained.append(item)
                elif utterance_id is not None and item[1] != utterance_id:
                    retained.append(item)
                # else: discard
        except queue.Empty:
            pass
        for item in retained:
            self._queue.put(item)

    @property
    def is_speaking(self) -> bool:
        """``True`` when audio is queued or actively playing."""
        return self._playing.is_set() or not self._queue.empty()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Consumer loop — runs in the daemon thread."""
        try:
            with sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                self._consume(stream)
        except Exception:
            logger.exception(
                "SpeakerThread: could not open audio output stream; "
                "continuing in silent mode"
            )
            self._consume(None)

    def _consume(self, stream: sd.OutputStream | None) -> None:
        """Drain the queue and write each chunk to *stream*."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP_SENTINEL:
                break

            audio, utterance_id, is_final = item
            self._playing.set()

            try:
                if stream is not None:
                    stream.write(audio)
            except Exception:
                logger.exception(
                    "SpeakerThread: error writing chunk (utterance_id=%s)",
                    utterance_id,
                )

            if is_final:
                self._event_queue.put(SpeechCompletedEvent(utterance_id=utterance_id))
                logger.debug(
                    "SpeakerThread: utterance complete (utterance_id=%s)",
                    utterance_id,
                )

            if self._queue.empty():
                self._playing.clear()
