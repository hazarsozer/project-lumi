"""
Microphone capture, wake word detection, and VAD-based recording for Project Lumi.

This module sits at the start of the audio pipeline:

    Microphone → [audio_queue] → Ears thread → [event_queue]

Key public class:
    Ears — starts a background consumer thread that continuously reads from the
    microphone via sounddevice, runs openwakeword inference on each 80 ms chunk,
    and posts a WakeDetectedEvent (then RecordingCompleteEvent after VAD recording)
    to the central event queue.

Constants:
    SAMPLE_RATE = 16000   (Hz — required by openwakeword and faster-whisper)
    CHUNK_SIZE  = 1280    (frames — 80 ms per chunk at 16 kHz)

Usage:
    from src.audio.ears import Ears
    ears = Ears(sensitivity=0.8, model_paths=["models/hey_lumi.onnx"])
    ears.start(event_queue)   # non-blocking; posts events to queue
    ears.stop()
"""

import logging
import queue
import threading
import time as _time
from typing import Any

import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.vad import VAD

from src.core.events import EarsErrorCode, EarsErrorEvent, WakeDetectedEvent

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

_MAX_RETRIES = 3  # InputStream open/run failures before giving up
_RETRY_DELAY_S = 0.25  # seconds to wait between retries


class Ears:
    def __init__(self, sensitivity: float = 0.5, model_paths: list[str] | None = None):
        """
        Implementation of the threaded microphone listener.
        Args:
            sensitivity: The sensitivity of the wake word detector.
            model_paths: Optional list of paths to wake word models.
        """

        self.sensitivity = sensitivity
        self.listening = False
        self._event_queue: queue.Queue[Any] | None = None

        # The buffer
        self.audio_queue: queue.Queue[Any] = queue.Queue()

        # The model
        logger.info("Loading the model...")

        # Work around the installed openwakeword version where AudioFeatures.__init__
        # does not accept the `inference_framework` keyword that Model passes.
        try:
            import openwakeword.utils as oww_utils

            AudioFeatures = oww_utils.AudioFeatures
            original_init = AudioFeatures.__init__

            # Only patch if the current signature doesn't already accept the kwarg
            if (
                hasattr(original_init, "__code__")
                and "inference_framework" not in original_init.__code__.co_varnames
            ):

                def _patched_audiofeatures_init(
                    self: Any, *args: Any, **kwargs: Any
                ) -> None:
                    # Drop unsupported kwarg and delegate to original initializer
                    kwargs.pop("inference_framework", None)
                    original_init(self, *args, **kwargs)

                AudioFeatures.__init__ = _patched_audiofeatures_init
        except Exception as e:
            logger.warning(
                "Could not apply openwakeword AudioFeatures compatibility patch: %s", e
            )

        # If no model paths are provided, try to use the custom hey_lumi model if it exists
        if model_paths is None:
            import os

            custom_model = "models/hey_lumi.onnx"
            if os.path.exists(custom_model):
                model_paths = [custom_model]
            else:
                model_paths = []  # Empty list will load all default models

        # Instantiate the wake word model (will lazily download resources if needed)
        self.model = Model(wakeword_model_paths=model_paths, inference_framework="onnx")
        logger.info(
            "Model loaded successfully. Active models: %s",
            list(self.model.models.keys()),
        )

        # Initialize VAD
        self.vad = VAD()

        # Cooldown timestamp (monotonic seconds) to ignore audio after a wake event
        self._cooldown_until = 0.0

    def _mic_callback(
        self, indata: np.ndarray, frames: int, time: Any, status: Any
    ) -> None:
        """
        Callback function for the microphone.
        """

        if status:
            logger.warning("Microphone status: %s", status)

        self.audio_queue.put(indata.copy())

    def record_command_with_vad(
        self, timeout: float = 10.0, silence_limit: float = 1.5
    ) -> np.ndarray:
        """
        Records audio from the queue until VAD detects silence or timeout is reached.
        Args:
            timeout: Maximum recording time in seconds.
            silence_limit: How many seconds of silence to wait before stopping.
        Returns:
            The recorded audio as a numpy array.
        """
        recorded_chunks = []

        start_time = _time.monotonic()
        last_voice_time = _time.monotonic()
        speech_detected = False

        while True:
            now = _time.monotonic()
            if now - start_time > timeout:
                break
            # Silence/no-speech checks run every iteration, even when queue is empty
            if speech_detected and (now - last_voice_time > silence_limit):
                break
            if not speech_detected and (now - start_time > 3.0):
                break

            try:
                chunk = self.audio_queue.get(timeout=0.1)

                # Ensure audio is 1D int16
                if isinstance(chunk, np.ndarray):
                    if chunk.ndim == 2:
                        chunk_flat = chunk[:, 0]
                    else:
                        chunk_flat = chunk
                    chunk_flat = chunk_flat.astype(np.int16, copy=False)

                recorded_chunks.append(chunk_flat)

                # VAD expects 16kHz 16-bit PCM; CHUNK_SIZE=1280 (80ms) is fine.
                vad_score = self.vad.predict(chunk_flat)

                if vad_score > 0.5:
                    speech_detected = True
                    last_voice_time = now

            except queue.Empty:
                continue

        if not recorded_chunks:
            return np.array([], dtype=np.int16)

        return np.concatenate(recorded_chunks)

    def _consumer_loop(self) -> None:
        """
        This runs in the background thread and processes the audio data.
        Posts WakeDetectedEvent to the event queue on wake word detection.

        Transient InputStream failures (PortAudioError, USB hiccups) are
        retried up to _MAX_RETRIES times with a short delay between attempts.
        On exhaustion, EarsErrorEvent is posted and the thread exits cleanly.
        """

        logger.info("Ears: starting listening...")
        _time.sleep(0)  # yield GIL so caller's post-start assertions run first

        retries = 0
        while self.listening and retries <= _MAX_RETRIES:
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_SIZE,
                    dtype="int16",
                    channels=1,
                    latency="high",
                    callback=self._mic_callback,
                ):
                    retries = 0  # stream opened successfully — reset retry counter
                    while self.listening:
                        try:
                            chunk = self.audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        # Ensure audio is 1D int16 as expected by openwakeword's AudioFeatures
                        if isinstance(chunk, np.ndarray):
                            if chunk.ndim == 2:
                                chunk = chunk[:, 0]
                            elif chunk.ndim > 2:
                                chunk = chunk.reshape(-1)
                            chunk = chunk.astype(np.int16, copy=False)

                        # Respect cooldown window: keep draining queue but skip inference
                        now = _time.monotonic()
                        if now < self._cooldown_until:
                            continue

                        try:
                            predictions = self.model.predict(chunk)
                        except Exception:
                            logger.warning(
                                "Ears: model.predict() failed on chunk; skipping",
                                exc_info=True,
                            )
                            continue

                        for model_name, score in predictions.items():
                            if score > self.sensitivity:
                                logger.info(
                                    "Wake word detected: %s with score %s",
                                    model_name,
                                    score,
                                )

                                if self._event_queue is not None:
                                    self._event_queue.put(
                                        WakeDetectedEvent(timestamp=_time.monotonic())
                                    )

                                try:
                                    while True:
                                        self.audio_queue.get_nowait()
                                except queue.Empty:
                                    pass

                                self.model.reset()
                                self._cooldown_until = _time.monotonic() + 2.0
                                break

            except sd.PortAudioError:
                retries += 1
                logger.warning(
                    "Ears: PortAudioError (attempt %d/%d); retrying in %.2fs",
                    retries,
                    _MAX_RETRIES,
                    _RETRY_DELAY_S,
                )
                _time.sleep(_RETRY_DELAY_S)
            except Exception:
                retries += 1
                logger.exception(
                    "Ears: unexpected error in capture loop (attempt %d/%d); retrying in %.2fs",
                    retries,
                    _MAX_RETRIES,
                    _RETRY_DELAY_S,
                )
                _time.sleep(_RETRY_DELAY_S)

        if self.listening and retries > _MAX_RETRIES:
            logger.error(
                "Ears: audio capture failed after %d retries; posting EarsErrorEvent",
                _MAX_RETRIES,
            )
            if self._event_queue is not None:
                self._event_queue.put(
                    EarsErrorEvent(
                        code=EarsErrorCode.UNRECOVERABLE,
                        detail=f"InputStream failed after {_MAX_RETRIES} retries",
                    )
                )

    def start(self, event_queue: queue.Queue[Any]) -> None:
        """
        Start the listener in a separate thread.
        Args:
            event_queue: The central event queue for posting wake/recording events.
        """
        self._event_queue = event_queue
        self.listening = True

        # Creating a new thread for the consumer loop
        self.thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
        )

        # Starting the thread
        self.thread.start()

    def stop(self) -> None:
        """
        Stop the listener.
        """
        self.listening = False
        if hasattr(self, "thread"):
            self.thread.join()

