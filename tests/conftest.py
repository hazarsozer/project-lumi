"""
Project Lumi — shared pytest fixtures and mocking conventions.

## Fixture Conventions

All fixtures in this file are session-safe unless explicitly marked otherwise.
Audio fixtures return numpy arrays with dtype=int16 and sample rate 16000 Hz,
matching the constants in src/audio/ears.py (SAMPLE_RATE=16000, CHUNK_SIZE=1280).

## Mocking Strategy

Hardware and ML model boundaries are mocked at the module-import level, never
at deep internal call sites. This means:

- ``sounddevice`` is patched via ``unittest.mock.patch`` on the ``sounddevice``
  module itself, so both ``ears.py`` and ``utils.py`` (which both ``import
  sounddevice as sd``) receive the mock transparently.
- ``openwakeword.model.Model`` is patched before ``Ears.__init__`` runs, so no
  ONNX runtime is invoked and no model files are required on disk.
- ``faster_whisper.WhisperModel`` is patched before ``Scribe.__init__`` runs,
  so no model weights are downloaded and no GPU/CPU inference occurs.
- ``openwakeword.vad.VAD`` is patched alongside the model to avoid any
  secondary ONNX loading that the VAD class triggers.

## Adding New Test Files

1. Import the fixtures you need by name — pytest auto-discovers them from here.
2. Mark every test with ``@pytest.mark.unit`` or ``@pytest.mark.integration``.
3. Never import from ``src.core`` in unit tests until the infra modules exist.
4. For any new hardware boundary (e.g. a camera, a serial port) follow the same
   pattern: patch at the top-level import, return a ``MagicMock``, and expose
   the mock object from the fixture so test functions can configure return values.
"""

from __future__ import annotations

import queue
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Audio data fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000
CHUNK_SIZE: int = 1280


@pytest.fixture()
def silence_chunk() -> np.ndarray:
    """Return a 1280-sample int16 array of zeros (pure silence)."""
    return np.zeros(CHUNK_SIZE, dtype=np.int16)


@pytest.fixture()
def speech_chunk() -> np.ndarray:
    """Return a 1280-sample 440 Hz sine wave at 16 kHz, dtype int16.

    The amplitude is set to 80 % of int16 max so the array is clearly
    non-silent and triggers VAD-like thresholds in tests.
    """
    t = np.arange(CHUNK_SIZE, dtype=np.float32) / SAMPLE_RATE
    sine = np.sin(2.0 * np.pi * 440.0 * t)
    return (sine * 0.8 * 32767).astype(np.int16)


@pytest.fixture()
def recorded_audio(speech_chunk: np.ndarray) -> np.ndarray:
    """Return ~3 seconds of synthetic speech-like audio (int16, 16 kHz).

    Built by repeating ``speech_chunk`` enough times to fill 3 seconds.
    Used for Scribe transcription tests.
    """
    num_chunks = int(np.ceil(SAMPLE_RATE * 3.0 / CHUNK_SIZE))
    return np.tile(speech_chunk, num_chunks)[: SAMPLE_RATE * 3]


# ---------------------------------------------------------------------------
# sounddevice mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sounddevice() -> Generator[MagicMock, None, None]:
    """Patch ``sounddevice`` so no real audio hardware is accessed.

    Both ``sounddevice.InputStream`` and ``sounddevice.play`` /
    ``sounddevice.wait`` are replaced with ``MagicMock`` objects.

    The ``InputStream`` mock supports the context-manager protocol so code
    using ``with sd.InputStream(...) as stream:`` works without error.

    Yields the top-level ``sounddevice`` mock so individual tests can
    inspect call counts or configure side-effects.
    """
    with patch("sounddevice.InputStream") as mock_stream_cls, \
         patch("sounddevice.play") as mock_play, \
         patch("sounddevice.wait") as mock_wait:

        # Make InputStream usable as a context manager
        mock_stream_instance = MagicMock()
        mock_stream_instance.__enter__ = MagicMock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = MagicMock(return_value=False)
        mock_stream_cls.return_value = mock_stream_instance

        sd_mock = MagicMock()
        sd_mock.InputStream = mock_stream_cls
        sd_mock.play = mock_play
        sd_mock.wait = mock_wait

        yield sd_mock


# ---------------------------------------------------------------------------
# faster-whisper mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_whisper_model() -> Generator[MagicMock, None, None]:
    """Patch ``faster_whisper.WhisperModel`` to return canned transcription.

    The mock ``transcribe`` method returns a single segment with text
    ``"hello lumi"`` and a dummy ``info`` object so callers that unpack
    ``(segments, info)`` work without error.

    Yields the mock *class* (not the instance) so tests can inspect
    constructor arguments if needed.
    """
    fake_segment = MagicMock()
    fake_segment.text = "hello lumi"

    fake_info = MagicMock()
    fake_info.language = "en"
    fake_info.language_probability = 0.99

    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = (iter([fake_segment]), fake_info)

    # Patch both the original module and the name as bound inside scribe.py
    # (``from faster_whisper import WhisperModel``).
    with patch("faster_whisper.WhisperModel", return_value=mock_instance), \
         patch("src.audio.scribe.WhisperModel", return_value=mock_instance) as mock_cls:
        yield mock_cls


# ---------------------------------------------------------------------------
# openwakeword model mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_oww_model() -> Generator[MagicMock, None, None]:
    """Patch ``openwakeword.model.Model`` to prevent ONNX loading.

    Also patches ``openwakeword.vad.VAD`` because the VAD class triggers
    its own ONNX session on construction.

    The mock Model instance has:
    - ``models`` dict keyed by ``"hey_lumi"`` (mirrors real model key)
    - ``predict()`` returning ``{"hey_lumi": 0.0}`` (below any threshold)
    - ``reset()`` as a no-op

    Yields the mock *Model class* so tests can configure ``predict``
    return values to simulate a wake-word detection event.
    """
    mock_model_instance = MagicMock()
    mock_model_instance.models = {"hey_lumi": MagicMock()}
    mock_model_instance.predict.return_value = {"hey_lumi": 0.0}
    mock_model_instance.reset.return_value = None

    mock_vad_instance = MagicMock()
    mock_vad_instance.predict.return_value = 0.0

    # Patch the names as they are bound inside ears.py (imported via
    # ``from openwakeword.model import Model`` and
    # ``from openwakeword.vad import VAD``).
    # Patching the original module paths alone would not intercept calls
    # already resolved in the ears module namespace.
    with patch("src.audio.ears.Model", return_value=mock_model_instance) as mock_cls, \
         patch("src.audio.ears.VAD", return_value=mock_vad_instance), \
         patch("openwakeword.model.Model", return_value=mock_model_instance), \
         patch("openwakeword.vad.VAD", return_value=mock_vad_instance):
        yield mock_cls


# ---------------------------------------------------------------------------
# llama_cpp mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llama_cpp() -> Generator[MagicMock, None, None]:
    """Patch llama_cpp.Llama to prevent any model loading.

    The mock instance supports:
    - __call__(prompt, ...) returning {"choices": [{"text": "mock response"}]}
    - create_completion(...) returning the same structure
    - Configurable via mock_llama_cpp.return_value to set custom responses

    Yields the mock Llama *class* so tests can configure return values.

    Mocking strategy: ``llama_cpp.Llama`` is patched at the top-level
    module boundary as well as at the name bound inside
    ``src.llm.model_loader`` (``from llama_cpp import Llama``).  This
    mirrors the same dual-patch pattern used for faster-whisper and
    openwakeword, ensuring the mock is seen regardless of whether the
    module imports the class directly or via the package namespace.
    """
    mock_instance = MagicMock()
    mock_instance.return_value = {"choices": [{"text": "mock response"}]}
    mock_instance.create_completion.return_value = {"choices": [{"text": "mock response"}]}

    with patch("llama_cpp.Llama", return_value=mock_instance) as mock_cls:
        yield mock_cls
