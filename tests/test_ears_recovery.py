"""
Tests for Ears runtime error recovery (Wave A1).

Covers:
- Transient PortAudioError is caught and retried; thread survives.
- model.predict() exception on a single chunk is skipped; loop continues.
- After _MAX_RETRIES exhausted, EarsErrorEvent is posted and thread exits.
- EarsErrorEvent posts correct code and detail strings.
- Orchestrator._handle_ears_error transitions non-IDLE state to IDLE.
- Orchestrator._handle_ears_error is a no-op when already IDLE.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.core.events import EarsErrorEvent
from src.audio.ears import _MAX_RETRIES, _RETRY_DELAY_S


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ears(sensitivity: float = 0.5):
    """Build Ears with all hardware mocked out (mirrors mock_oww_model fixture)."""
    mock_model_instance = MagicMock()
    mock_model_instance.models = {"hey_lumi": MagicMock()}
    mock_model_instance.predict.return_value = {"hey_lumi": 0.0}
    mock_model_instance.reset.return_value = None

    mock_vad_instance = MagicMock()
    mock_vad_instance.predict.return_value = 0.0

    with (
        patch("src.audio.ears.Model", return_value=mock_model_instance),
        patch("src.audio.ears.VAD", return_value=mock_vad_instance),
        patch("openwakeword.model.Model", return_value=mock_model_instance),
        patch("openwakeword.vad.VAD", return_value=mock_vad_instance),
        patch("os.path.exists", return_value=False),
    ):
        from src.audio.ears import Ears
        ears = Ears(sensitivity=sensitivity, model_paths=[])
    return ears


# ---------------------------------------------------------------------------
# model.predict() chunk-level errors are skipped, loop continues
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_predict_exception_skips_chunk_and_continues():
    """A RuntimeError from model.predict() on one chunk is caught; the loop
    processes subsequent chunks without crashing."""
    ears = _make_ears()

    # predict raises on first call, returns empty on second (no wake word),
    # and we flip listening off on the third get() so the thread exits.
    predict_calls = [0]

    def _predict(chunk):
        predict_calls[0] += 1
        if predict_calls[0] == 1:
            raise RuntimeError("ONNX inference failed")
        return {}

    ears.model.predict.side_effect = _predict

    get_calls = [0]

    def _limited_get(**kwargs):
        get_calls[0] += 1
        if get_calls[0] >= 3:
            ears.listening = False
            raise queue.Empty
        return np.zeros(1280, dtype=np.int16)

    ears.audio_queue.get = _limited_get  # type: ignore[method-assign]

    eq = queue.Queue()

    with patch("sounddevice.InputStream") as mock_sd:
        mock_sd.return_value.__enter__ = lambda s: s
        mock_sd.return_value.__exit__ = MagicMock(return_value=False)
        ears.start(eq)
        ears.thread.join(timeout=4.0)

    # Thread must have exited without posting an EarsErrorEvent
    assert not ears.thread.is_alive()
    assert eq.empty()


# ---------------------------------------------------------------------------
# Transient PortAudioError is retried; thread survives
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_port_audio_error_triggers_retry():
    """A PortAudioError on InputStream.__enter__ is caught; the loop retries
    and eventually exits cleanly when listening is set to False."""
    import sounddevice as sd

    ears = _make_ears()
    eq = queue.Queue()

    open_calls = [0]

    class _FakeStream:
        def __enter__(self):
            open_calls[0] += 1
            if open_calls[0] == 1:
                raise sd.PortAudioError("device busy")
            # Second open succeeds; immediately stop listening so thread exits.
            ears.listening = False
            return self

        def __exit__(self, *args):
            return False

    with patch("sounddevice.InputStream", return_value=_FakeStream()):
        ears.start(eq)
        ears.thread.join(timeout=4.0)

    assert not ears.thread.is_alive()
    assert open_calls[0] == 2  # one fail + one success
    assert eq.empty()  # no EarsErrorEvent — recovered before exhaustion


# ---------------------------------------------------------------------------
# After _MAX_RETRIES exhausted, EarsErrorEvent is posted
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(10)
def test_exhausted_retries_posts_ears_error_event():
    """When PortAudioError fires on every attempt, EarsErrorEvent is posted
    to the event queue after _MAX_RETRIES failures."""
    import sounddevice as sd

    ears = _make_ears()
    eq = queue.Queue()

    class _AlwaysFail:
        def __enter__(self):
            raise sd.PortAudioError("no device")

        def __exit__(self, *args):
            return False

    with patch("sounddevice.InputStream", return_value=_AlwaysFail()):
        with patch("src.audio.ears._RETRY_DELAY_S", 0.0):
            ears.start(eq)
            ears.thread.join(timeout=8.0)

    assert not ears.thread.is_alive()

    # Exactly one EarsErrorEvent should be in the queue
    assert not eq.empty()
    event = eq.get_nowait()
    assert isinstance(event, EarsErrorEvent)
    assert event.code == "ears.unrecoverable"
    assert str(_MAX_RETRIES) in event.detail


# ---------------------------------------------------------------------------
# EarsErrorEvent fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(10)
def test_ears_error_event_detail_mentions_retry_count():
    """EarsErrorEvent.detail includes the retry count so operators can triage."""
    import sounddevice as sd

    ears = _make_ears()
    eq = queue.Queue()

    class _AlwaysFail:
        def __enter__(self):
            raise sd.PortAudioError("boom")

        def __exit__(self, *args):
            return False

    with patch("sounddevice.InputStream", return_value=_AlwaysFail()):
        with patch("src.audio.ears._RETRY_DELAY_S", 0.0):
            ears.start(eq)
            ears.thread.join(timeout=8.0)

    event: EarsErrorEvent = eq.get_nowait()
    assert "retries" in event.detail.lower() or str(_MAX_RETRIES) in event.detail


# ---------------------------------------------------------------------------
# Orchestrator handles EarsErrorEvent
# ---------------------------------------------------------------------------


def _make_minimal_orchestrator():
    """Build an Orchestrator with all heavy subsystems mocked."""
    from src.core.config import LumiConfig, IPCConfig, RAGConfig, VisionConfig, ToolsConfig
    from src.core.orchestrator import Orchestrator

    config = LumiConfig(
        ipc=IPCConfig(enabled=False),
        rag=RAGConfig(enabled=False),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
    )

    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()

    with (
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
        patch("src.core.orchestrator.ReasoningRouter"),
    ):
        mock_mem_cls.return_value.load = MagicMock()
        orch = Orchestrator(config, speaker=speaker)
    return orch


@pytest.mark.unit
def test_ears_error_handler_transitions_to_idle_from_listening():
    """_handle_ears_error moves LISTENING → IDLE."""
    from src.core.state_machine import LumiState

    orch = _make_minimal_orchestrator()
    orch._state_machine.transition_to(LumiState.LISTENING)

    orch._handle_ears_error(EarsErrorEvent(code="ears.unrecoverable", detail="test"))

    assert orch._state_machine.current_state == LumiState.IDLE


@pytest.mark.unit
def test_ears_error_handler_noop_when_already_idle():
    """_handle_ears_error is a no-op when already IDLE."""
    from src.core.state_machine import LumiState

    orch = _make_minimal_orchestrator()
    assert orch._state_machine.current_state == LumiState.IDLE

    orch._handle_ears_error(EarsErrorEvent(code="ears.unrecoverable", detail="test"))

    assert orch._state_machine.current_state == LumiState.IDLE
