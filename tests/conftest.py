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
import sys
import types
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# IPC auth — disable token verification during all tests
# ---------------------------------------------------------------------------

# When ~/.lumi/ipc_token exists on a dev machine, EventBridge enables bearer-
# token auth.  Integration tests that use FakeWSClient.do_handshake() send no
# token, so they would be rejected.  Patching read_token at the event_bridge
# import site to always return None keeps token auth disabled for the whole
# test session without touching any test or the production handshake logic.
@pytest.fixture(autouse=True, scope="session")
def _disable_ipc_token_auth() -> Generator[None, None, None]:
    """Patch EventBridge's read_token so no bearer token is required during tests."""
    with patch("src.core.event_bridge.read_token", return_value=None):
        yield

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
    with (
        patch("sounddevice.InputStream") as mock_stream_cls,
        patch("sounddevice.play") as mock_play,
        patch("sounddevice.wait") as mock_wait,
    ):

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
    with (
        patch("faster_whisper.WhisperModel", return_value=mock_instance),
        patch("src.audio.scribe.WhisperModel", return_value=mock_instance) as mock_cls,
    ):
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
    with (
        patch("src.audio.ears.Model", return_value=mock_model_instance) as mock_cls,
        patch("src.audio.ears.VAD", return_value=mock_vad_instance),
        patch("openwakeword.model.Model", return_value=mock_model_instance),
        patch("openwakeword.vad.VAD", return_value=mock_vad_instance),
    ):
        yield mock_cls


# ---------------------------------------------------------------------------
# llama_cpp mock
# ---------------------------------------------------------------------------


def _make_streaming_side_effect(
    return_value: dict,
) -> Any:
    """Return a ``create_completion`` side-effect that honours ``stream=True``.

    When *stream=True* is passed the callable returns a generator that yields
    the single-response dict as one chunk, matching the llama.cpp streaming
    protocol (an iterator of ``{"choices": [{"text": ..., "finish_reason":
    ...}]}`` dicts).

    When *stream* is absent or False the callable returns the plain dict,
    preserving non-streaming usage.

    If a test has already set a custom ``side_effect`` on
    ``mock_instance.create_completion``, that side_effect is used directly —
    this helper only wraps the *default* return value.
    """

    def _side_effect(*args: object, **kwargs: object) -> Any:
        stream = kwargs.get("stream", False)
        if stream:
            # Return a one-shot generator containing the response dict.
            # Tests that need multi-token streaming override side_effect
            # themselves (see test_cr02_empty_first_token.py).
            def _gen() -> Any:
                yield return_value

            return _gen()
        return return_value

    return _side_effect


@pytest.fixture()
def mock_llama_cpp() -> Generator[MagicMock, None, None]:
    """Patch llama_cpp.Llama to prevent any model loading.

    The mock instance supports:
    - __call__(prompt, ...) returning {"choices": [{"text": "mock response"}]}
    - create_completion(..., stream=False) returning the same structure as a
      plain dict (non-streaming, backward-compatible).
    - create_completion(..., stream=True) returning a one-shot generator that
      yields the response dict as a single chunk, matching the llama.cpp
      streaming protocol used by ReasoningRouter.generate().
    - Configurable via mock_llama_cpp.return_value to set custom responses;
      tests may also override create_completion.side_effect directly for
      multi-token sequences.

    Yields the mock Llama *class* so tests can configure return values.

    Mocking strategy: ``llama_cpp.Llama`` is patched at the top-level
    module boundary as well as at the name bound inside
    ``src.llm.model_loader`` (``from llama_cpp import Llama``).  This
    mirrors the same dual-patch pattern used for faster-whisper and
    openwakeword, ensuring the mock is seen regardless of whether the
    module imports the class directly or via the package namespace.
    """
    _default_response: dict = {"choices": [{"text": "mock response", "finish_reason": "stop"}]}

    mock_instance = MagicMock()
    mock_instance.return_value = _default_response
    mock_instance.create_completion.side_effect = _make_streaming_side_effect(
        _default_response
    )
    mock_cls = MagicMock(return_value=mock_instance)

    # llama_cpp is an optional extra not installed on CI. Inject a fake module
    # into sys.modules so the lazy `import llama_cpp` inside ModelLoader.load()
    # picks up the mock without requiring the real package.
    mock_module = types.ModuleType("llama_cpp")
    mock_module.Llama = mock_cls  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"llama_cpp": mock_module}):
        yield mock_cls
