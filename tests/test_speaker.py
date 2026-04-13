"""
Tests for src/audio/speaker.py — SpeakerThread

Mocking strategy
----------------
- ``sounddevice.OutputStream`` is patched via the ``mock_sd_output_stream``
  fixture so no real audio device is ever opened.
- For pure queue-mechanics tests (enqueue, flush, is_speaking) the thread is
  NOT started; those tests construct SpeakerThread directly and exercise the
  public API without requiring the mock fixture.
- Thread-based tests (SpeechCompletedEvent, stop) use
  ``threading.Event.wait(timeout=2.0)`` or ``queue.Queue.get(timeout=2.0)``
  to avoid hangs.

All tests are marked ``pytest.mark.unit`` and pass without real audio hardware,
GPU, or network access.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.events import SpeechCompletedEvent


# ---------------------------------------------------------------------------
# Fixture: mock sounddevice.OutputStream
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sd_output_stream() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Patch ``sounddevice.OutputStream`` to prevent real audio device access.

    Yields a 2-tuple ``(mock_class, mock_stream_instance)`` so tests can
    inspect constructor arguments or configure ``write`` side-effects.
    """
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    with patch("src.audio.speaker.sd.OutputStream", return_value=mock_stream) as mock_cls:
        yield mock_cls, mock_stream


# ---------------------------------------------------------------------------
# Helper: build a SpeakerThread ready for testing
# ---------------------------------------------------------------------------


def _make_speaker(sample_rate: int = 24000) -> tuple["SpeakerThread", queue.Queue]:  # noqa: F821
    from src.audio.speaker import SpeakerThread

    event_q: queue.Queue = queue.Queue()
    speaker = SpeakerThread(event_queue=event_q, sample_rate=sample_rate)
    return speaker, event_q


def _make_audio(n: int = 480, dtype=np.float32) -> np.ndarray:
    """Return a small sine wave array for test purposes."""
    t = np.arange(n, dtype=np.float32) / 24000
    return (np.sin(2.0 * np.pi * 440.0 * t)).astype(dtype)


# ===========================================================================
# 1. Construction
# ===========================================================================


@pytest.mark.unit
def test_speaker_thread_initial_is_speaking_false():
    """is_speaking must be False immediately after construction."""
    speaker, _ = _make_speaker()
    assert speaker.is_speaking is False


@pytest.mark.unit
def test_speaker_thread_initial_queue_empty():
    """Internal queue must be empty immediately after construction."""
    speaker, _ = _make_speaker()
    assert speaker._queue.empty()


# ===========================================================================
# 2. enqueue
# ===========================================================================


@pytest.mark.unit
def test_enqueue_puts_item_on_queue():
    """A single enqueue call must result in exactly one item on the queue."""
    speaker, _ = _make_speaker()
    audio = _make_audio()
    speaker.enqueue(audio, utterance_id="utt-1", is_final=False)
    assert speaker._queue.qsize() == 1


@pytest.mark.unit
def test_enqueue_no_resample_when_rates_match():
    """When source_rate == sample_rate the audio must be returned unchanged as float32."""
    speaker, _ = _make_speaker(sample_rate=24000)
    audio = _make_audio(n=480, dtype=np.float32)
    speaker.enqueue(audio, utterance_id="utt-1", is_final=False, source_rate=24000)

    item = speaker._queue.get_nowait()
    resampled, uid, final = item
    assert resampled.dtype == np.float32
    np.testing.assert_array_almost_equal(resampled, audio.astype(np.float32))


@pytest.mark.unit
def test_enqueue_resamples_when_rates_differ():
    """When source_rate != sample_rate the output length must approximate the resampled target."""
    speaker, _ = _make_speaker(sample_rate=24000)
    source_rate = 44100
    n_input = 441  # 10 ms at 44100 Hz
    audio = _make_audio(n=n_input, dtype=np.float32)
    speaker.enqueue(audio, utterance_id="utt-1", is_final=False, source_rate=source_rate)

    item = speaker._queue.get_nowait()
    resampled, _, _ = item
    expected_len = round(n_input * 24000 / source_rate)
    # Allow ±2 samples tolerance from resampling rounding
    assert abs(len(resampled) - expected_len) <= 2


@pytest.mark.unit
def test_enqueue_converts_to_float32():
    """int16 audio input must be stored as float32 in the queue."""
    speaker, _ = _make_speaker()
    audio_int16 = _make_audio(n=480, dtype=np.int16)
    speaker.enqueue(audio_int16, utterance_id="utt-1", is_final=False)

    item = speaker._queue.get_nowait()
    resampled, _, _ = item
    assert resampled.dtype == np.float32


@pytest.mark.unit
def test_is_speaking_true_after_enqueue():
    """is_speaking must be True while a chunk is waiting in the queue."""
    speaker, _ = _make_speaker()
    audio = _make_audio()
    speaker.enqueue(audio, utterance_id="utt-1", is_final=False)
    assert speaker.is_speaking is True


# ===========================================================================
# 3. flush
# ===========================================================================


@pytest.mark.unit
def test_flush_all_drains_queue():
    """flush() with no argument must empty the queue completely."""
    speaker, _ = _make_speaker()
    for i in range(3):
        speaker.enqueue(_make_audio(), utterance_id=f"utt-{i}", is_final=False)
    assert speaker._queue.qsize() == 3

    speaker.flush()
    assert speaker._queue.empty()


@pytest.mark.unit
def test_flush_by_utterance_id_removes_matching():
    """flush('utt-1') must remove all items whose utterance_id is 'utt-1'."""
    speaker, _ = _make_speaker()
    speaker.enqueue(_make_audio(), utterance_id="utt-1", is_final=False)
    speaker.enqueue(_make_audio(), utterance_id="utt-1", is_final=True)
    speaker.enqueue(_make_audio(), utterance_id="utt-2", is_final=False)

    speaker.flush("utt-1")

    remaining = []
    while not speaker._queue.empty():
        remaining.append(speaker._queue.get_nowait())

    # No "utt-1" items should remain
    assert all(item[1] != "utt-1" for item in remaining if item is not None)


@pytest.mark.unit
def test_flush_by_utterance_id_keeps_other_utterances():
    """flush('utt-1') must leave items for other utterance_ids intact."""
    speaker, _ = _make_speaker()
    speaker.enqueue(_make_audio(), utterance_id="utt-1", is_final=False)
    speaker.enqueue(_make_audio(), utterance_id="utt-2", is_final=False)
    speaker.enqueue(_make_audio(), utterance_id="utt-2", is_final=True)

    speaker.flush("utt-1")

    remaining = []
    while not speaker._queue.empty():
        remaining.append(speaker._queue.get_nowait())

    utt2_items = [item for item in remaining if item is not None and item[1] == "utt-2"]
    assert len(utt2_items) == 2


@pytest.mark.unit
def test_flush_preserves_sentinel():
    """flush() must put the None sentinel back if it was in the queue."""
    speaker, _ = _make_speaker()
    # Manually place sentinel and a regular item
    speaker._queue.put((_make_audio(), "utt-1", False))
    speaker._queue.put(None)  # sentinel

    speaker.flush()

    # Only the sentinel should remain
    assert speaker._queue.qsize() == 1
    assert speaker._queue.get_nowait() is None


# ===========================================================================
# 4. SpeechCompletedEvent posting (integration — thread must run)
# ===========================================================================


@pytest.mark.unit
def test_speech_completed_event_posted_on_is_final(mock_sd_output_stream):
    """A SpeechCompletedEvent must be posted when is_final=True chunk is consumed."""
    speaker, event_q = _make_speaker()
    speaker.start()

    audio = _make_audio()
    speaker.enqueue(audio, utterance_id="utt-42", is_final=True)

    # Wait up to 2 s for the event to arrive
    try:
        event = event_q.get(timeout=2.0)
    except queue.Empty:
        pytest.fail("SpeechCompletedEvent was not posted within 2 s")
    finally:
        speaker.stop()

    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-42"


@pytest.mark.unit
def test_no_speech_completed_event_when_not_final(mock_sd_output_stream):
    """No SpeechCompletedEvent must be posted when is_final=False."""
    speaker, event_q = _make_speaker()
    speaker.start()

    audio = _make_audio()
    speaker.enqueue(audio, utterance_id="utt-99", is_final=False)

    # Give the thread enough time to consume the chunk
    time.sleep(0.3)
    speaker.stop()

    assert event_q.empty(), "Expected no SpeechCompletedEvent for non-final chunk"


# ===========================================================================
# 5. stop
# ===========================================================================


@pytest.mark.unit
def test_stop_joins_thread(mock_sd_output_stream):
    """After stop(), the internal thread must no longer be alive."""
    speaker, _ = _make_speaker()
    speaker.start()
    assert speaker._thread.is_alive()

    speaker.stop()
    assert not speaker._thread.is_alive()


@pytest.mark.unit
def test_stop_after_stop_is_safe(mock_sd_output_stream):
    """Calling stop() a second time must not raise any exception."""
    speaker, _ = _make_speaker()
    speaker.start()
    speaker.stop()
    # Second stop must be harmless
    speaker.stop()


# ===========================================================================
# 6. Silent fallback
# ===========================================================================


@pytest.mark.unit
def test_run_continues_in_silent_mode_on_stream_error():
    """SpeechCompletedEvent must still be posted when the audio stream fails to open."""
    # Patch OutputStream to raise on __enter__ (simulates missing audio device)
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(side_effect=Exception("no audio device"))
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("src.audio.speaker.sd.OutputStream", return_value=mock_stream):
        speaker, event_q = _make_speaker()
        speaker.start()

        audio = _make_audio()
        speaker.enqueue(audio, utterance_id="silent-utt", is_final=True)

        try:
            event = event_q.get(timeout=2.0)
        except queue.Empty:
            pytest.fail("SpeechCompletedEvent not posted in silent fallback mode within 2 s")
        finally:
            speaker.stop()

    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "silent-utt"
