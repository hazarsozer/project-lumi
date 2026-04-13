"""Tests for src/core/events.py — typed event dataclasses."""

import pytest
import numpy as np

from src.core.events import (
    WakeDetectedEvent,
    RecordingCompleteEvent,
    TranscriptReadyEvent,
    CommandResultEvent,
    LLMResponseReadyEvent,
    TTSChunkReadyEvent,
    VisemeEvent,
    SpeechCompletedEvent,
    LLMTokenEvent,
    InterruptEvent,
    ShutdownEvent,
    UserTextEvent,
    ZMQMessage,
)


# ---------------------------------------------------------------------------
# WakeDetectedEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_wake_detected_event_stores_timestamp() -> None:
    evt = WakeDetectedEvent(timestamp=1.23)
    assert evt.timestamp == 1.23


@pytest.mark.unit
def test_wake_detected_event_is_frozen() -> None:
    evt = WakeDetectedEvent(timestamp=1.0)
    with pytest.raises((AttributeError, TypeError)):
        evt.timestamp = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RecordingCompleteEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_recording_complete_event_stores_audio() -> None:
    audio = np.zeros(1280, dtype=np.int16)
    evt = RecordingCompleteEvent(audio=audio)
    assert evt.audio is audio


@pytest.mark.unit
def test_recording_complete_event_equality_with_numpy() -> None:
    a = np.array([1, 2, 3], dtype=np.int16)
    b = np.array([1, 2, 3], dtype=np.int16)
    assert RecordingCompleteEvent(audio=a) == RecordingCompleteEvent(audio=b)


@pytest.mark.unit
def test_recording_complete_event_inequality_with_numpy() -> None:
    a = np.array([1, 2, 3], dtype=np.int16)
    b = np.array([4, 5, 6], dtype=np.int16)
    assert RecordingCompleteEvent(audio=a) != RecordingCompleteEvent(audio=b)


@pytest.mark.unit
def test_recording_complete_event_is_unhashable() -> None:
    evt = RecordingCompleteEvent(audio=np.zeros(8, dtype=np.int16))
    with pytest.raises(TypeError):
        hash(evt)


@pytest.mark.unit
def test_recording_complete_event_eq_with_non_ndarray_audio() -> None:
    sentinel = object()
    assert RecordingCompleteEvent(audio=sentinel) == RecordingCompleteEvent(audio=sentinel)
    assert RecordingCompleteEvent(audio=sentinel) != RecordingCompleteEvent(audio=object())


@pytest.mark.unit
def test_recording_complete_event_eq_returns_notimplemented_for_other_type() -> None:
    evt = RecordingCompleteEvent(audio=np.zeros(4, dtype=np.int16))
    assert evt.__eq__("not an event") is NotImplemented


@pytest.mark.unit
def test_recording_complete_event_is_frozen() -> None:
    evt = RecordingCompleteEvent(audio=np.zeros(4, dtype=np.int16))
    with pytest.raises((AttributeError, TypeError)):
        evt.audio = np.zeros(2, dtype=np.int16)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TranscriptReadyEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_transcript_ready_event_stores_text() -> None:
    evt = TranscriptReadyEvent(text="hello lumi")
    assert evt.text == "hello lumi"


@pytest.mark.unit
def test_transcript_ready_event_is_frozen() -> None:
    evt = TranscriptReadyEvent(text="hi")
    with pytest.raises((AttributeError, TypeError)):
        evt.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CommandResultEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_command_result_event_stores_type() -> None:
    evt = CommandResultEvent(command_type="interrupt")
    assert evt.command_type == "interrupt"


@pytest.mark.unit
def test_command_result_event_is_frozen() -> None:
    evt = CommandResultEvent(command_type="interrupt")
    with pytest.raises((AttributeError, TypeError)):
        evt.command_type = "volume_control"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMResponseReadyEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_llm_response_ready_event_stores_text() -> None:
    evt = LLMResponseReadyEvent(text="The answer is 42.")
    assert evt.text == "The answer is 42."


@pytest.mark.unit
def test_llm_response_ready_event_is_frozen() -> None:
    evt = LLMResponseReadyEvent(text="hi")
    with pytest.raises((AttributeError, TypeError)):
        evt.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TTSChunkReadyEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tts_chunk_ready_event_stores_fields() -> None:
    audio = np.zeros(512, dtype=np.float32)
    evt = TTSChunkReadyEvent(
        audio=audio, sample_rate=24000, chunk_id=0, is_final=False, utterance_id="utt-1"
    )
    assert evt.audio is audio
    assert evt.sample_rate == 24000
    assert evt.chunk_id == 0
    assert evt.is_final is False
    assert evt.utterance_id == "utt-1"


@pytest.mark.unit
def test_tts_chunk_ready_event_is_final_flag() -> None:
    audio = np.zeros(256, dtype=np.float32)
    evt = TTSChunkReadyEvent(
        audio=audio, sample_rate=24000, chunk_id=7, is_final=True, utterance_id="utt-1"
    )
    assert evt.is_final is True
    assert evt.chunk_id == 7


@pytest.mark.unit
def test_tts_chunk_ready_event_equality() -> None:
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.1, 0.2], dtype=np.float32)
    assert TTSChunkReadyEvent(
        audio=a, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    ) == TTSChunkReadyEvent(
        audio=b, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    )


@pytest.mark.unit
def test_tts_chunk_ready_event_inequality_audio() -> None:
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.3, 0.4], dtype=np.float32)
    assert TTSChunkReadyEvent(
        audio=a, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    ) != TTSChunkReadyEvent(
        audio=b, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    )


@pytest.mark.unit
def test_tts_chunk_ready_event_inequality_metadata() -> None:
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.1, 0.2], dtype=np.float32)
    # Different chunk_id
    assert TTSChunkReadyEvent(
        audio=a, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    ) != TTSChunkReadyEvent(
        audio=b, sample_rate=24000, chunk_id=2, is_final=False, utterance_id="utt-1"
    )
    # Different sample_rate
    assert TTSChunkReadyEvent(
        audio=a, sample_rate=16000, chunk_id=1, is_final=False, utterance_id="utt-1"
    ) != TTSChunkReadyEvent(
        audio=b, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    )
    # Different is_final
    assert TTSChunkReadyEvent(
        audio=a, sample_rate=24000, chunk_id=1, is_final=True, utterance_id="utt-1"
    ) != TTSChunkReadyEvent(
        audio=b, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    )


@pytest.mark.unit
def test_tts_chunk_ready_event_inequality_utterance_id() -> None:
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.1, 0.2], dtype=np.float32)
    assert TTSChunkReadyEvent(
        audio=a, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-1"
    ) != TTSChunkReadyEvent(
        audio=b, sample_rate=24000, chunk_id=1, is_final=False, utterance_id="utt-2"
    )


@pytest.mark.unit
def test_tts_chunk_ready_event_eq_with_non_ndarray_audio() -> None:
    sentinel = object()
    evt1 = TTSChunkReadyEvent(
        audio=sentinel, sample_rate=24000, chunk_id=0, is_final=False, utterance_id="utt-1"
    )
    evt2 = TTSChunkReadyEvent(
        audio=sentinel, sample_rate=24000, chunk_id=0, is_final=False, utterance_id="utt-1"
    )
    assert evt1 == evt2


@pytest.mark.unit
def test_tts_chunk_ready_event_eq_returns_notimplemented_for_other_type() -> None:
    audio = np.zeros(8, dtype=np.float32)
    evt = TTSChunkReadyEvent(
        audio=audio, sample_rate=24000, chunk_id=0, is_final=False, utterance_id="utt-1"
    )
    assert evt.__eq__("not an event") is NotImplemented


@pytest.mark.unit
def test_tts_chunk_ready_event_is_unhashable() -> None:
    audio = np.zeros(8, dtype=np.float32)
    evt = TTSChunkReadyEvent(
        audio=audio, sample_rate=24000, chunk_id=0, is_final=False, utterance_id="utt-1"
    )
    with pytest.raises(TypeError):
        hash(evt)


@pytest.mark.unit
def test_tts_chunk_ready_event_is_frozen() -> None:
    audio = np.zeros(8, dtype=np.float32)
    evt = TTSChunkReadyEvent(
        audio=audio, sample_rate=24000, chunk_id=0, is_final=False, utterance_id="utt-1"
    )
    with pytest.raises((AttributeError, TypeError)):
        evt.chunk_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VisemeEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_viseme_event_stores_fields() -> None:
    evt = VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=120, duration_ms=80)
    assert evt.utterance_id == "utt-1"
    assert evt.phoneme == "AA"
    assert evt.start_ms == 120
    assert evt.duration_ms == 80


@pytest.mark.unit
def test_viseme_event_is_frozen() -> None:
    evt = VisemeEvent(utterance_id="utt-1", phoneme="IY", start_ms=0, duration_ms=50)
    with pytest.raises((AttributeError, TypeError)):
        evt.phoneme = "EH"  # type: ignore[misc]


@pytest.mark.unit
def test_viseme_event_equality() -> None:
    assert VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=40) == \
           VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=40)


@pytest.mark.unit
def test_viseme_event_inequality() -> None:
    assert VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=40) != \
           VisemeEvent(utterance_id="utt-1", phoneme="EH", start_ms=10, duration_ms=40)
    assert VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=40) != \
           VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=20, duration_ms=40)
    assert VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=40) != \
           VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=80)


@pytest.mark.unit
def test_viseme_event_inequality_utterance_id() -> None:
    assert VisemeEvent(utterance_id="utt-1", phoneme="AA", start_ms=10, duration_ms=40) != \
           VisemeEvent(utterance_id="utt-2", phoneme="AA", start_ms=10, duration_ms=40)


# ---------------------------------------------------------------------------
# SpeechCompletedEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_speech_completed_event_stores_utterance_id() -> None:
    evt = SpeechCompletedEvent(utterance_id="utt-42")
    assert evt.utterance_id == "utt-42"


@pytest.mark.unit
def test_speech_completed_event_is_frozen() -> None:
    evt = SpeechCompletedEvent(utterance_id="utt-1")
    with pytest.raises((AttributeError, TypeError)):
        evt.utterance_id = "utt-2"  # type: ignore[misc]


@pytest.mark.unit
def test_speech_completed_event_equality() -> None:
    assert SpeechCompletedEvent(utterance_id="x") == \
           SpeechCompletedEvent(utterance_id="x")
    assert SpeechCompletedEvent(utterance_id="x") != \
           SpeechCompletedEvent(utterance_id="y")


# ---------------------------------------------------------------------------
# LLMTokenEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_llm_token_event_stores_fields() -> None:
    evt = LLMTokenEvent(token="hello", utterance_id="utt-9")
    assert evt.token == "hello"
    assert evt.utterance_id == "utt-9"


@pytest.mark.unit
def test_llm_token_event_is_frozen() -> None:
    evt = LLMTokenEvent(token="hi", utterance_id="utt-1")
    with pytest.raises((AttributeError, TypeError)):
        evt.token = "bye"  # type: ignore[misc]


@pytest.mark.unit
def test_llm_token_event_equality() -> None:
    assert LLMTokenEvent(token="a", utterance_id="u1") == \
           LLMTokenEvent(token="a", utterance_id="u1")
    assert LLMTokenEvent(token="a", utterance_id="u1") != \
           LLMTokenEvent(token="b", utterance_id="u1")
    assert LLMTokenEvent(token="a", utterance_id="u1") != \
           LLMTokenEvent(token="a", utterance_id="u2")


# ---------------------------------------------------------------------------
# InterruptEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_interrupt_event_stores_source() -> None:
    evt = InterruptEvent(source="zmq")
    assert evt.source == "zmq"


@pytest.mark.unit
def test_interrupt_event_is_frozen() -> None:
    evt = InterruptEvent(source="keyboard")
    with pytest.raises((AttributeError, TypeError)):
        evt.source = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ShutdownEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_shutdown_event_instantiates() -> None:
    evt = ShutdownEvent()
    assert isinstance(evt, ShutdownEvent)


@pytest.mark.unit
def test_shutdown_event_equality() -> None:
    assert ShutdownEvent() == ShutdownEvent()


# ---------------------------------------------------------------------------
# UserTextEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_user_text_event_stores_text() -> None:
    evt = UserTextEvent(text="what time is it")
    assert evt.text == "what time is it"


@pytest.mark.unit
def test_user_text_event_is_frozen() -> None:
    evt = UserTextEvent(text="hi")
    with pytest.raises((AttributeError, TypeError)):
        evt.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ZMQMessage
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_zmq_message_default_version() -> None:
    msg = ZMQMessage(event="state_change", payload={"state": "idle"}, timestamp=1.0)
    assert msg.version == "1.0"


@pytest.mark.unit
def test_zmq_message_stores_fields() -> None:
    msg = ZMQMessage(event="transcript", payload={"text": "hi"}, timestamp=2.5, version="2.0")
    assert msg.event == "transcript"
    assert msg.payload == {"text": "hi"}
    assert msg.timestamp == 2.5
    assert msg.version == "2.0"


@pytest.mark.unit
def test_zmq_message_is_frozen() -> None:
    msg = ZMQMessage(event="e", payload={}, timestamp=0.0)
    with pytest.raises((AttributeError, TypeError)):
        msg.event = "mutated"  # type: ignore[misc]
