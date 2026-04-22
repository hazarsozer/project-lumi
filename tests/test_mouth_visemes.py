"""Tests for viseme event posting in KokoroTTS (src/audio/mouth.py).

Validates that _post_visemes correctly extracts phoneme tuples from
kokoro_onnx.Kokoro.create() and posts VisemeEvent instances.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.mouth import KokoroTTS
from src.audio.speaker import SpeakerThread
from src.core.events import SpeechCompletedEvent, TTSChunkReadyEvent, VisemeEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(n: int = 480) -> np.ndarray:
    """Return a small float32 sine-wave array (24 kHz content)."""
    t = np.arange(n, dtype=np.float32) / 24_000
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


def _make_tts(
    mock_kokoro: MagicMock | None = None,
    event_queue: queue.Queue | None = None,
) -> tuple[KokoroTTS, queue.Queue | None, MagicMock]:
    """Construct a KokoroTTS whose model load is fully mocked.

    Returns (tts, event_queue, mock_speaker).
    """
    event_q = event_queue if event_queue is not None else queue.Queue()
    mock_speaker = MagicMock(spec=SpeakerThread)

    with patch("os.path.exists", return_value=True):
        with patch("src.audio.mouth.KokoroTTS._load_model"):
            tts = KokoroTTS(
                model_path="/fake.onnx",
                voices_path="/fake.bin",
                voice="af_heart",
                speaker=mock_speaker,
                event_queue=event_q,
            )
            if mock_kokoro is not None:
                tts._kokoro = mock_kokoro
                tts._silent = False
            else:
                tts._silent = False

    return tts, event_q, mock_speaker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthesize_posts_viseme_events() -> None:
    """When kokoro returns phonemes, VisemeEvents are posted to event_queue."""
    audio = _make_audio(480)
    phonemes = [
        ("AH", 0, 80),
        ("L", 80, 60),
        ("OW", 140, 100),
    ]
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio, phonemes)

    tts, event_q, _ = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello.", utterance_id="utt-vis")

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    viseme_events = [e for e in events if isinstance(e, VisemeEvent)]
    assert len(viseme_events) == 3

    # _KOKORO_FRAMES_TO_MS = 256 / 24000 * 1000 ≈ 10.6667 ms/frame
    _hop_ms = 256 / 24000 * 1000

    assert viseme_events[0].phoneme == "AH"
    assert viseme_events[0].start_ms == pytest.approx(0 * _hop_ms)
    assert viseme_events[0].duration_ms == pytest.approx(80 * _hop_ms)
    assert viseme_events[0].utterance_id == "utt-vis"

    assert viseme_events[1].phoneme == "L"
    assert viseme_events[2].phoneme == "OW"


@pytest.mark.unit
def test_synthesize_none_phonemes_no_viseme_events() -> None:
    """When kokoro returns (samples, None), no VisemeEvents are posted."""
    audio = _make_audio(480)
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio, None)

    tts, event_q, _ = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello.", utterance_id="utt-none")

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    viseme_events = [e for e in events if isinstance(e, VisemeEvent)]
    assert len(viseme_events) == 0


@pytest.mark.unit
def test_synthesize_malformed_phoneme_item_skipped() -> None:
    """A non-tuple phoneme item is skipped; valid items still produce events."""
    audio = _make_audio(480)
    phonemes = [
        ("AH", 0, 80),
        "not-a-tuple",  # malformed — should be skipped
        ("P", 160, 50),
    ]
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio, phonemes)

    tts, event_q, _ = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello.", utterance_id="utt-malformed")

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    viseme_events = [e for e in events if isinstance(e, VisemeEvent)]
    # Only 2 valid phoneme tuples
    assert len(viseme_events) == 2
    assert viseme_events[0].phoneme == "AH"
    assert viseme_events[1].phoneme == "P"


@pytest.mark.unit
def test_synthesize_no_event_queue_no_crash() -> None:
    """event_queue=None with phonemes present must not crash."""
    audio = _make_audio(480)
    phonemes = [("AH", 0, 80)]
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio, phonemes)

    mock_speaker = MagicMock(spec=SpeakerThread)

    with patch("os.path.exists", return_value=True):
        with patch("src.audio.mouth.KokoroTTS._load_model"):
            tts = KokoroTTS(
                model_path="/fake.onnx",
                voices_path="/fake.bin",
                speaker=mock_speaker,
                event_queue=None,
            )
            tts._kokoro = mock_kokoro
            tts._silent = False

    # Must not raise
    tts.synthesize("Hello.", utterance_id="utt-noqueue")
