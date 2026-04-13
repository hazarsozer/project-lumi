"""
Tests for src/utils.py

Mocking strategy
----------------
``play_ready_sound()`` now enqueues audio onto a :class:`SpeakerThread`
instead of calling ``sd.play`` / ``sd.wait`` directly.  We inject a
``MagicMock(spec=SpeakerThread)`` so no real audio hardware is accessed.
We inspect the numpy array passed to ``speaker.enqueue`` to verify the
generated tone is correct.

All tests are marked ``pytest.mark.unit``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.audio.speaker import SpeakerThread


# ---------------------------------------------------------------------------
# play_ready_sound
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_speaker() -> MagicMock:
    """A MagicMock that satisfies the SpeakerThread interface."""
    return MagicMock(spec=SpeakerThread)


@pytest.mark.unit
def test_play_ready_sound_calls_enqueue(mock_speaker: MagicMock) -> None:
    """play_ready_sound() calls speaker.enqueue() exactly once."""
    from src.utils import play_ready_sound

    play_ready_sound(mock_speaker)

    mock_speaker.enqueue.assert_called_once()


@pytest.mark.unit
def test_play_ready_sound_enqueue_utterance_id(mock_speaker: MagicMock) -> None:
    """play_ready_sound() uses utterance_id='ready-sound'."""
    from src.utils import play_ready_sound

    play_ready_sound(mock_speaker)

    _, kwargs = mock_speaker.enqueue.call_args
    args = mock_speaker.enqueue.call_args.args
    # utterance_id may be positional or keyword
    utterance_id = kwargs.get("utterance_id", args[1] if len(args) > 1 else None)
    assert utterance_id == "ready-sound"


@pytest.mark.unit
def test_play_ready_sound_enqueue_is_final(mock_speaker: MagicMock) -> None:
    """play_ready_sound() passes is_final=True (single chunk, no follow-up)."""
    from src.utils import play_ready_sound

    play_ready_sound(mock_speaker)

    _, kwargs = mock_speaker.enqueue.call_args
    args = mock_speaker.enqueue.call_args.args
    is_final = kwargs.get("is_final", args[2] if len(args) > 2 else None)
    assert is_final is True


@pytest.mark.unit
def test_play_ready_sound_passes_float32_array(mock_speaker: MagicMock) -> None:
    """play_ready_sound() passes a float32 numpy array as the first argument."""
    from src.utils import play_ready_sound

    play_ready_sound(mock_speaker)

    audio = mock_speaker.enqueue.call_args.args[0]
    assert isinstance(audio, np.ndarray), "First argument must be a numpy array"
    assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"


@pytest.mark.unit
def test_play_ready_sound_array_amplitude_bounded(mock_speaker: MagicMock) -> None:
    """The audio array values are in [-1.0, 1.0] (normalised float32 range)."""
    from src.utils import play_ready_sound

    play_ready_sound(mock_speaker)

    audio = mock_speaker.enqueue.call_args.args[0]
    assert float(np.max(np.abs(audio))) <= 1.0, (
        "Audio amplitude exceeds normalised float32 range"
    )


@pytest.mark.unit
def test_play_ready_sound_array_is_approximately_0_2_seconds(
    mock_speaker: MagicMock,
) -> None:
    """The audio array is approximately 0.2 seconds at the canonical 24000 Hz rate.

    Expected length = 24000 * 0.2 = 4800 samples.  We allow ±10 samples tolerance.
    """
    from src.utils import play_ready_sound

    play_ready_sound(mock_speaker)

    audio = mock_speaker.enqueue.call_args.args[0]
    expected_length = int(24_000 * 0.2)
    assert abs(len(audio) - expected_length) <= 10, (
        f"Expected ~{expected_length} samples, got {len(audio)}"
    )


@pytest.mark.unit
def test_play_ready_sound_handles_enqueue_exception(mock_speaker: MagicMock) -> None:
    """play_ready_sound() does not propagate exceptions raised by speaker.enqueue()."""
    mock_speaker.enqueue.side_effect = Exception("no audio device")

    from src.utils import play_ready_sound

    # Should not raise — the function catches and logs the error.
    play_ready_sound(mock_speaker)
