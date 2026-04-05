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


# ---------------------------------------------------------------------------
# LLMResponseReadyEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_llm_response_ready_event_stores_text() -> None:
    evt = LLMResponseReadyEvent(text="The answer is 42.")
    assert evt.text == "The answer is 42."


# ---------------------------------------------------------------------------
# TTSChunkReadyEvent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tts_chunk_ready_event_stores_fields() -> None:
    audio = np.zeros(512, dtype=np.float32)
    evt = TTSChunkReadyEvent(audio=audio, viseme="AA", duration_ms=80)
    assert evt.viseme == "AA"
    assert evt.duration_ms == 80
    assert evt.audio is audio


@pytest.mark.unit
def test_tts_chunk_ready_event_equality() -> None:
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.1, 0.2], dtype=np.float32)
    assert TTSChunkReadyEvent(audio=a, viseme="sil", duration_ms=40) == \
           TTSChunkReadyEvent(audio=b, viseme="sil", duration_ms=40)


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
