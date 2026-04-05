"""
Tests for src/utils.py

Mocking strategy
----------------
``sounddevice.play`` and ``sounddevice.wait`` are patched at the module level
so ``play_ready_sound()`` never touches real audio hardware.  We inspect the
numpy array passed to ``sd.play`` to verify the generated tone is correct.

All tests are marked ``pytest.mark.unit``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# play_ready_sound
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_play_ready_sound_calls_sd_play(mock_sounddevice: MagicMock) -> None:
    """play_ready_sound() invokes sd.play() exactly once."""
    from src.utils import play_ready_sound

    play_ready_sound()

    mock_sounddevice.play.assert_called_once()


@pytest.mark.unit
def test_play_ready_sound_calls_sd_wait(mock_sounddevice: MagicMock) -> None:
    """play_ready_sound() invokes sd.wait() exactly once after sd.play()."""
    from src.utils import play_ready_sound

    play_ready_sound()

    mock_sounddevice.wait.assert_called_once()


@pytest.mark.unit
def test_play_ready_sound_passes_float32_array(mock_sounddevice: MagicMock) -> None:
    """play_ready_sound() passes a float32 numpy array as the first argument to sd.play()."""
    from src.utils import play_ready_sound

    play_ready_sound()

    args, _kwargs = mock_sounddevice.play.call_args
    audio_array = args[0]
    assert isinstance(audio_array, np.ndarray), "First argument must be a numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"


@pytest.mark.unit
def test_play_ready_sound_array_is_approximately_0_2_seconds(
    mock_sounddevice: MagicMock,
) -> None:
    """The audio passed to sd.play() is approximately 0.2 seconds at 44100 Hz.

    Expected length = 44100 * 0.2 = 8820 samples.  We allow ±10 samples
    tolerance for floating-point rounding in linspace.
    """
    from src.utils import play_ready_sound

    play_ready_sound()

    args, kwargs = mock_sounddevice.play.call_args
    audio_array = args[0]
    sample_rate = args[1] if len(args) > 1 else kwargs.get("samplerate", 44100)

    expected_length = int(sample_rate * 0.2)
    assert abs(len(audio_array) - expected_length) <= 10, (
        f"Expected ~{expected_length} samples, got {len(audio_array)}"
    )


@pytest.mark.unit
def test_play_ready_sound_array_amplitude_bounded(mock_sounddevice: MagicMock) -> None:
    """The audio array values are in [-1.0, 1.0] (valid normalised float32 range)."""
    from src.utils import play_ready_sound

    play_ready_sound()

    args, _kwargs = mock_sounddevice.play.call_args
    audio_array = args[0]
    assert float(np.max(np.abs(audio_array))) <= 1.0, (
        "Audio amplitude exceeds normalised float32 range"
    )


@pytest.mark.unit
def test_play_ready_sound_handles_sd_play_exception(
    mock_sounddevice: MagicMock,
) -> None:
    """play_ready_sound() does not propagate exceptions raised by sd.play()."""
    mock_sounddevice.play.side_effect = Exception("no audio device")

    from src.utils import play_ready_sound

    # Should not raise — the function catches and prints the error.
    play_ready_sound()
