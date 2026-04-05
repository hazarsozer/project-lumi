"""
Tests for src/audio/scribe.py

Mocking strategy
----------------
``faster_whisper.WhisperModel`` is patched via the ``mock_whisper_model``
fixture so no model weights are downloaded and no CPU/GPU inference runs.
The mock ``transcribe`` method returns a single canned segment with text
``"hello lumi"``.

All tests are marked ``pytest.mark.unit``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scribe(mock_whisper_model: MagicMock):
    """Construct a Scribe instance with the WhisperModel mock active."""
    from src.audio.scribe import Scribe
    return Scribe(model_size="tiny.en", device="cpu")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scribe_init_stores_initial_prompt(mock_whisper_model: MagicMock) -> None:
    """Scribe stores the initial_prompt passed at construction."""
    from src.audio.scribe import Scribe
    scribe = Scribe(model_size="tiny.en", device="cpu", initial_prompt="test prompt")
    assert scribe.initial_prompt == "test prompt"


@pytest.mark.unit
def test_scribe_init_calls_whisper_model_ctor(mock_whisper_model: MagicMock) -> None:
    """Scribe calls WhisperModel exactly once during construction."""
    _make_scribe(mock_whisper_model)
    mock_whisper_model.assert_called_once()


# ---------------------------------------------------------------------------
# int16 to float32 normalisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scribe_transcribe_normalises_int16(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() converts int16 audio to float32 before calling the model.

    We capture the array passed to model.transcribe and verify its dtype.
    """
    scribe = _make_scribe(mock_whisper_model)
    # recorded_audio is int16 (from conftest)
    assert recorded_audio.dtype == np.int16

    scribe.transcribe(recorded_audio)

    mock_instance = mock_whisper_model.return_value
    call_args = mock_instance.transcribe.call_args
    audio_passed = call_args[0][0]
    assert audio_passed.dtype == np.float32


@pytest.mark.unit
def test_scribe_transcribe_normalises_int16_range(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """Normalised float32 audio is in the range [-1.0, 1.0]."""
    scribe = _make_scribe(mock_whisper_model)
    scribe.transcribe(recorded_audio)

    mock_instance = mock_whisper_model.return_value
    call_args = mock_instance.transcribe.call_args
    audio_passed = call_args[0][0]
    assert float(np.max(np.abs(audio_passed))) <= 1.0 + 1e-6


@pytest.mark.unit
def test_scribe_transcribe_passes_float32_unchanged(
    mock_whisper_model: MagicMock,
) -> None:
    """transcribe() does not alter audio that is already float32."""
    scribe = _make_scribe(mock_whisper_model)
    float_audio = np.zeros(1024, dtype=np.float32)
    scribe.transcribe(float_audio)

    mock_instance = mock_whisper_model.return_value
    call_args = mock_instance.transcribe.call_args
    audio_passed = call_args[0][0]
    assert audio_passed.dtype == np.float32
    np.testing.assert_array_equal(audio_passed, float_audio)


# ---------------------------------------------------------------------------
# Transcription output
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scribe_transcribe_returns_string(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() returns a str."""
    scribe = _make_scribe(mock_whisper_model)
    result = scribe.transcribe(recorded_audio)
    assert isinstance(result, str)


@pytest.mark.unit
def test_scribe_transcribe_returns_canned_text(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() returns the text produced by the mocked model."""
    scribe = _make_scribe(mock_whisper_model)
    result = scribe.transcribe(recorded_audio)
    assert result == "hello lumi"


@pytest.mark.unit
def test_scribe_transcribe_uses_default_prompt(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() passes the constructor prompt when no override is given."""
    from src.audio.scribe import Scribe
    scribe = Scribe(
        model_size="tiny.en",
        device="cpu",
        initial_prompt="custom default prompt",
    )
    scribe.transcribe(recorded_audio)

    mock_instance = mock_whisper_model.return_value
    call_kwargs = mock_instance.transcribe.call_args[1]
    assert call_kwargs.get("initial_prompt") == "custom default prompt"


@pytest.mark.unit
def test_scribe_transcribe_uses_override_prompt(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() uses the override prompt when one is explicitly supplied."""
    scribe = _make_scribe(mock_whisper_model)
    scribe.transcribe(recorded_audio, initial_prompt="override prompt")

    mock_instance = mock_whisper_model.return_value
    call_kwargs = mock_instance.transcribe.call_args[1]
    assert call_kwargs.get("initial_prompt") == "override prompt"


@pytest.mark.unit
def test_scribe_transcribe_empty_audio_returns_string(
    mock_whisper_model: MagicMock,
) -> None:
    """transcribe() handles an empty int16 array without raising."""
    # Configure the mock to return an empty segment list for empty audio.
    mock_instance = mock_whisper_model.return_value
    fake_info = MagicMock()
    mock_instance.transcribe.return_value = (iter([]), fake_info)

    scribe = _make_scribe(mock_whisper_model)
    result = scribe.transcribe(np.array([], dtype=np.int16))
    assert isinstance(result, str)
    assert result == ""


@pytest.mark.unit
def test_scribe_transcribe_strips_whitespace(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() strips leading/trailing whitespace from the joined segments."""
    mock_instance = mock_whisper_model.return_value
    fake_segment = MagicMock()
    fake_segment.text = "  hello lumi  "
    fake_info = MagicMock()
    mock_instance.transcribe.return_value = (iter([fake_segment]), fake_info)

    scribe = _make_scribe(mock_whisper_model)
    result = scribe.transcribe(recorded_audio)
    assert result == "hello lumi"


@pytest.mark.unit
def test_scribe_transcribe_joins_multiple_segments(
    mock_whisper_model: MagicMock,
    recorded_audio: np.ndarray,
) -> None:
    """transcribe() joins multiple segment texts with a space."""
    mock_instance = mock_whisper_model.return_value
    seg1 = MagicMock()
    seg1.text = "hello"
    seg2 = MagicMock()
    seg2.text = "lumi"
    fake_info = MagicMock()
    mock_instance.transcribe.return_value = (iter([seg1, seg2]), fake_info)

    scribe = _make_scribe(mock_whisper_model)
    result = scribe.transcribe(recorded_audio)
    assert result == "hello lumi"


# ---------------------------------------------------------------------------
# parse_command — CommandResult and pattern matching
# ---------------------------------------------------------------------------


from src.audio.scribe import CommandResult, parse_command


@pytest.mark.unit
def test_command_result_is_frozen() -> None:
    result = CommandResult(type="interrupt")
    with pytest.raises((AttributeError, TypeError)):
        result.type = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_command_result_default_action_is_none() -> None:
    assert CommandResult(type="interrupt").action is None


@pytest.mark.unit
@pytest.mark.parametrize("text", ["stop", "Stop", "STOP", "cancel", "never mind", "please stop"])
def test_parse_command_interrupt(text: str) -> None:
    result = parse_command(text)
    assert result is not None
    assert result.type == "interrupt"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["volume up", "Volume Up", "turn volume up"])
def test_parse_command_volume_up(text: str) -> None:
    result = parse_command(text)
    assert result is not None
    assert result.type == "volume_control"
    assert result.action == "up"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["volume down", "Volume Down"])
def test_parse_command_volume_down(text: str) -> None:
    result = parse_command(text)
    assert result is not None
    assert result.action == "down"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["mute", "Mute", "please mute"])
def test_parse_command_mute(text: str) -> None:
    result = parse_command(text)
    assert result is not None
    assert result.action == "mute"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["what time is it", "hello lumi", "", "   ", "stopping", "volumeup"])
def test_parse_command_returns_none_for_unknown(text: str) -> None:
    assert parse_command(text) is None
