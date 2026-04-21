"""
KokoroTTS — Text-to-Speech engine for Project Lumi.

Wraps the kokoro-onnx ONNX inference engine to stream audio chunks to the
SpeakerThread and post TTSChunkReadyEvent to the orchestrator event bus.

Text is split into sentences so synthesis can begin streaming to the speaker
while subsequent sentences are still being synthesized.

Usage:
    tts = KokoroTTS(
        model_path="models/kokoro.onnx",
        voices_path="models/voices.bin",
        speaker=speaker_thread,
        event_queue=orchestrator_queue,
    )
    # Call from a daemon thread — synthesize() blocks until done or cancelled.
    tts.synthesize("Hello world.", utterance_id="utt-1")
    tts.cancel("utt-1")  # from another thread to abort mid-stream
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from typing import Any

import numpy as np

from src.audio.speaker import SpeakerThread
from src.core.events import SpeechCompletedEvent, TTSChunkReadyEvent, VisemeEvent

logger = logging.getLogger(__name__)

# Kokoro-82M ONNX outputs at 24 kHz — matches SpeakerThread canonical rate.
_TTS_SAMPLE_RATE: int = 24_000

# KPipeline.forward yields pred_dur as integer hop-frame counts (not ms).
# Each frame = hop_length / sample_rate * 1000 ms (from istftnet.py defaults).
_KOKORO_HOP_LENGTH: int = 256
_KOKORO_FRAMES_TO_MS: float = _KOKORO_HOP_LENGTH / _TTS_SAMPLE_RATE * 1000.0

# Sentence boundary: split after .  !  ?  … followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentence-level chunks for incremental synthesis.

    Returns a list of non-empty stripped strings.  Falls back to a
    single-element list containing the *stripped* text if no sentence
    boundary is found.
    """
    stripped = text.strip()
    parts = _SENTENCE_RE.split(stripped)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [stripped]


class KokoroTTS:
    """Streams Kokoro ONNX TTS audio to SpeakerThread and the event bus.

    Synthesis is intended to be called from a daemon thread (not the
    orchestrator event loop) because ONNX inference is blocking.

    If *model_path* does not exist or the model fails to load, the engine
    falls back to **silent mode**: synthesize() fires SpeechCompletedEvent
    directly without producing any audio, keeping the state machine healthy.

    Args:
        model_path: Path to the Kokoro ONNX model file.
        voices_path: Path to the Kokoro voices file.
        voice: Kokoro voice identifier (e.g. "af_heart").
        speaker: SpeakerThread to enqueue audio chunks to.
            When None, chunks are generated but not played.
        event_queue: Orchestrator event queue for TTSChunkReadyEvent
            (and SpeechCompletedEvent in silent mode).
    """

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        voice: str = "af_heart",
        speaker: SpeakerThread | None = None,
        event_queue: queue.Queue[Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice
        self._speaker = speaker
        self._event_queue = event_queue

        # Thread-safe cancel flag; cleared at the start of each synthesize()
        # *unless* the utterance was pre-cancelled (see prepare() + synthesize()).
        self._cancel_flag: threading.Event = threading.Event()

        # Guards _busy and _current_utterance_id for cross-thread access.
        self._busy_lock: threading.Lock = threading.Lock()
        self._busy: bool = False
        self._current_utterance_id: str | None = None

        # Kokoro model handle; None in silent mode.
        self._kokoro: Any = None
        self._silent: bool = False

        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """True when synthesis is in progress."""
        with self._busy_lock:
            return self._busy

    def prepare(self, utterance_id: str) -> None:
        """Pre-register *utterance_id* so cancel() can target it before synthesize() starts.

        The orchestrator calls this immediately before launching the synthesis
        daemon thread.  Without this call there is a window between the thread
        being started and synthesize() executing its first lock acquisition where
        a concurrent cancel() would see _current_utterance_id=None and be a no-op.

        Args:
            utterance_id: The utterance that is about to be synthesized.
        """
        with self._busy_lock:
            self._current_utterance_id = utterance_id
        logger.debug("KokoroTTS: prepared utterance_id=%s", utterance_id)

    def synthesize(self, text: str, utterance_id: str) -> None:
        """Synthesize *text* and stream audio to the speaker and event bus.

        Blocks until synthesis completes, is cancelled, or an error occurs.
        Intended to be called from a daemon thread — never from the event loop.

        Always arranges for a SpeechCompletedEvent to reach the orchestrator:
        - Normal completion: speaker fires it after the final chunk.
        - Silent mode / empty: _emit_silence() posts it directly.
        - Cancelled mid-stream: the finally block posts it directly so the
          state machine can exit SPEAKING even without an interrupt.

        Args:
            text: The text to synthesize.
            utterance_id: Unique identifier for this utterance; used for
                cancel targeting and event correlation.
        """
        # Atomically set busy + id, snapshot the cancel flag, then always clear it.
        # Reading the flag BEFORE clearing detects the pre-cancel case (cancel()
        # called after prepare() but before this thread reached the lock).
        # Clearing the flag even when pre_cancelled=True prevents a stale-flag
        # cascade where the next synthesize() call also sees a set flag.
        with self._busy_lock:
            self._busy = True
            self._current_utterance_id = utterance_id
            pre_cancelled = self._cancel_flag.is_set()
            self._cancel_flag.clear()

        completed = False
        try:
            if pre_cancelled:
                logger.debug(
                    "KokoroTTS: pre-cancelled utterance_id=%s — aborting", utterance_id
                )
                return  # finally will post SpeechCompletedEvent

            if not text.strip():
                logger.debug(
                    "KokoroTTS: empty text for utterance_id=%s — emitting silence",
                    utterance_id,
                )
                self._emit_silence(utterance_id)
                completed = True
                return

            if self._silent:
                logger.debug(
                    "KokoroTTS: silent mode — emitting silence for utterance_id=%s",
                    utterance_id,
                )
                self._emit_silence(utterance_id)
                completed = True
                return

            completed = self._stream_text(text, utterance_id)
        finally:
            if not completed:
                # Synthesis was cancelled or pre-cancelled — no is_final chunk
                # was sent to the speaker, so SpeechCompletedEvent must come from
                # here.  If _handle_interrupt is also running, it drains the queue
                # and transitions to IDLE directly; a redundant SpeechCompletedEvent
                # arriving later is harmless (guarded by _handle_speech_completed).
                if self._event_queue is not None:
                    self._event_queue.put(SpeechCompletedEvent(utterance_id=utterance_id))
                logger.debug(
                    "KokoroTTS: synthesis incomplete — posted SpeechCompletedEvent "
                    "for utterance_id=%s",
                    utterance_id,
                )
            with self._busy_lock:
                self._busy = False
                self._current_utterance_id = None

    def cancel(self, utterance_id: str) -> None:
        """Signal cancellation for *utterance_id*.

        Thread-safe.  Sets the cancel flag only if the given utterance is
        the one currently being synthesized (or prepared via prepare()).
        The running synthesize() call will notice the flag at the next
        sentence or emit boundary and return; the finally block ensures
        SpeechCompletedEvent is posted.

        Args:
            utterance_id: The utterance to cancel.
        """
        # Hold the lock across the entire check-and-set to prevent a concurrent
        # synthesize() from clearing the flag between the check and the set.
        with self._busy_lock:
            if self._current_utterance_id == utterance_id:
                self._cancel_flag.set()
                logger.debug(
                    "KokoroTTS: cancel signalled for utterance_id=%s", utterance_id
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the Kokoro ONNX model; fall back to silent mode."""
        if not os.path.exists(self._model_path):
            logger.warning(
                "KokoroTTS: model not found at %r — running in silent mode",
                self._model_path,
            )
            self._silent = True
            return

        try:
            import kokoro_onnx  # type: ignore[import]

            self._kokoro = kokoro_onnx.Kokoro(self._model_path, self._voices_path)
            logger.info("KokoroTTS: loaded model from %r", self._model_path)
        except Exception:
            logger.exception(
                "KokoroTTS: failed to load model from %r — silent mode",
                self._model_path,
            )
            self._silent = True

    def _stream_text(self, text: str, utterance_id: str) -> bool:
        """Synthesize *text* sentence-by-sentence and emit audio chunks.

        Returns:
            True if synthesis completed normally (all chunks emitted including
            the is_final=True chunk — speaker will fire SpeechCompletedEvent).
            False if synthesis was cancelled before the final chunk was emitted
            (caller must arrange SpeechCompletedEvent itself).
        """
        sentences = _split_sentences(text)
        chunks: list[np.ndarray] = []

        # Inference pass — collect all sentence audio, checking cancel each time.
        for sentence in sentences:
            if self._cancel_flag.is_set():
                logger.debug(
                    "KokoroTTS: cancelled before synthesising %r (utterance_id=%s)",
                    sentence,
                    utterance_id,
                )
                return False

            try:
                samples, phonemes = self._kokoro.create(
                    sentence, voice=self._voice, speed=1.0, lang="en-us"
                )
                self._post_visemes(phonemes, utterance_id)
            except Exception:
                logger.exception(
                    "KokoroTTS: inference failed for sentence %r (utterance_id=%s)",
                    sentence,
                    utterance_id,
                )
                continue

            if isinstance(samples, np.ndarray) and samples.size > 0:
                chunks.append(samples.astype(np.float32))

        if not chunks:
            logger.warning(
                "KokoroTTS: no audio produced for utterance_id=%s — emitting silence",
                utterance_id,
            )
            self._emit_silence(utterance_id)
            return True  # SpeechCompletedEvent posted by _emit_silence

        # Emit pass — send each chunk, checking cancel between chunks.
        # NOTE: a cancel that fires *during* _emit_chunk (not between chunks) will
        # not be caught until the next loop iteration.  The orchestrator's interrupt
        # path calls speaker.flush() after cancel(), so any already-enqueued chunks
        # from the current iteration will be discarded at the speaker level.
        for chunk_id, chunk in enumerate(chunks):
            if self._cancel_flag.is_set():
                logger.debug(
                    "KokoroTTS: cancelled at emit %d/%d for utterance_id=%s",
                    chunk_id,
                    len(chunks),
                    utterance_id,
                )
                return False

            is_final = chunk_id == len(chunks) - 1
            self._emit_chunk(
                chunk,
                chunk_id=chunk_id,
                is_final=is_final,
                utterance_id=utterance_id,
            )

        return True  # all chunks emitted; speaker will fire SpeechCompletedEvent

    def _emit_silence(self, utterance_id: str) -> None:
        """Post SpeechCompletedEvent without audio (silent mode / empty synthesis).

        Bypasses the speaker entirely; posts the completion event directly
        so the orchestrator can transition back to IDLE.
        """
        if self._event_queue is not None:
            self._event_queue.put(SpeechCompletedEvent(utterance_id=utterance_id))
        logger.debug(
            "KokoroTTS: silence emitted for utterance_id=%s", utterance_id
        )

    def _post_visemes(self, phonemes: Any, utterance_id: str) -> None:
        """Post VisemeEvent instances for each phoneme in *phonemes*.

        Handles the case where phonemes is None or an unrecognised type by
        logging a debug message and returning without posting any events.

        Args:
            phonemes: KPipeline.forward phoneme list:
                      [(phoneme_str: str, start_frames: int, duration_frames: torch.Tensor), ...]
                      start_frames comes from itertools.accumulate (Python int);
                      duration_frames is a 0-dim torch.Tensor of hop-frame counts.
                      Both are converted to float ms via _KOKORO_FRAMES_TO_MS.
            utterance_id: The utterance this viseme sequence belongs to.
        """
        if phonemes is None:
            return
        if not isinstance(phonemes, (list, tuple)):
            logger.debug(
                "KokoroTTS: unexpected phonemes type %s — skipping visemes",
                type(phonemes).__name__,
            )
            return

        from src.audio.viseme_map import map_phoneme  # noqa: F811 — deferred import

        for item in phonemes:
            try:
                phoneme_str, start_frames, duration_frames = item
            except (TypeError, ValueError):
                logger.debug(
                    "KokoroTTS: cannot unpack phoneme item %r — skipping", item
                )
                continue

            # float() extracts scalar from a 0-dim torch.Tensor via __float__
            start_ms = float(start_frames) * _KOKORO_FRAMES_TO_MS
            duration_ms = float(duration_frames) * _KOKORO_FRAMES_TO_MS

            if self._event_queue is not None:
                self._event_queue.put(
                    VisemeEvent(
                        utterance_id=utterance_id,
                        phoneme=str(phoneme_str),
                        start_ms=start_ms,
                        duration_ms=duration_ms,
                    )
                )

    def _emit_chunk(
        self,
        audio: np.ndarray,
        chunk_id: int,
        is_final: bool,
        utterance_id: str,
    ) -> None:
        """Post TTSChunkReadyEvent and enqueue audio to speaker.

        Args:
            audio: float32 mono audio samples at _TTS_SAMPLE_RATE.
            chunk_id: Monotonically increasing index within the utterance.
            is_final: True on the last chunk; speaker posts SpeechCompletedEvent.
            utterance_id: Utterance identifier for event correlation.
        """
        if self._event_queue is not None:
            self._event_queue.put(
                TTSChunkReadyEvent(
                    audio=audio,
                    sample_rate=_TTS_SAMPLE_RATE,
                    chunk_id=chunk_id,
                    is_final=is_final,
                    utterance_id=utterance_id,
                )
            )

        if self._speaker is not None:
            self._speaker.enqueue(
                audio,
                utterance_id=utterance_id,
                is_final=is_final,
                source_rate=_TTS_SAMPLE_RATE,
            )
