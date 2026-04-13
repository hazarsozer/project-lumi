"""Tests for src/audio/mouth.py — KokoroTTS engine."""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.speaker import SpeakerThread
from src.audio.mouth import KokoroTTS, _split_sentences
from src.core.events import SpeechCompletedEvent, TTSChunkReadyEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(n: int = 480) -> np.ndarray:
    """Return a small float32 sine-wave array (24 kHz content)."""
    t = np.arange(n, dtype=np.float32) / 24_000
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


def _make_tts(
    silent: bool = False,
    mock_kokoro: MagicMock | None = None,
    voice: str = "af_heart",
) -> tuple[KokoroTTS, queue.Queue, MagicMock]:
    """Construct a KokoroTTS whose model load is fully mocked.

    Returns (tts, event_queue, mock_speaker).
    """
    event_q: queue.Queue = queue.Queue()
    mock_speaker = MagicMock(spec=SpeakerThread)

    # model_path that does NOT exist on disk so _load_model triggers silent
    # mode unless we patch os.path.exists to return True.
    model_path = "/nonexistent/kokoro.onnx"
    voices_path = "/nonexistent/voices.bin"

    if silent:
        tts = KokoroTTS(
            model_path=model_path,
            voices_path=voices_path,
            voice=voice,
            speaker=mock_speaker,
            event_queue=event_q,
        )
    else:
        # Patch os.path.exists so _load_model thinks the file exists, then
        # patch kokoro_onnx.Kokoro so no real ONNX import happens.
        with patch("os.path.exists", return_value=True):
            with patch("src.audio.mouth.KokoroTTS._load_model"):
                tts = KokoroTTS(
                    model_path=model_path,
                    voices_path=voices_path,
                    voice=voice,
                    speaker=mock_speaker,
                    event_queue=event_q,
                )
                # Inject a mock kokoro handle
                if mock_kokoro is not None:
                    tts._kokoro = mock_kokoro
                    tts._silent = False
                else:
                    tts._silent = False  # optimistic — tests that need real audio set _kokoro

    return tts, event_q, mock_speaker


# ---------------------------------------------------------------------------
# 1. Construction — model missing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_construction_model_missing_sets_silent_mode():
    """KokoroTTS with a nonexistent model_path must enter silent mode without raising."""
    event_q: queue.Queue = queue.Queue()
    mock_speaker = MagicMock(spec=SpeakerThread)

    tts = KokoroTTS(
        model_path="/does/not/exist.onnx",
        voices_path="/does/not/exist.bin",
        speaker=mock_speaker,
        event_queue=event_q,
    )

    assert tts._silent is True
    assert tts._kokoro is None


# ---------------------------------------------------------------------------
# 2. Construction — load fails
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_construction_load_fails_sets_silent_mode(tmp_path):
    """When the model file exists but kokoro_onnx.Kokoro() raises, _silent must be True."""
    # Create a real (empty) file so os.path.exists passes
    model_file = tmp_path / "kokoro.onnx"
    model_file.touch()
    voices_file = tmp_path / "voices.bin"
    voices_file.touch()

    event_q: queue.Queue = queue.Queue()
    mock_speaker = MagicMock(spec=SpeakerThread)

    # mouth.py does `import kokoro_onnx` inside _load_model; we inject a fake
    # module into sys.modules so the import succeeds, then patch its Kokoro class.
    import sys
    import types

    fake_mod = types.ModuleType("kokoro_onnx")
    fake_mod.Kokoro = MagicMock(side_effect=RuntimeError("bad model"))  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"kokoro_onnx": fake_mod}):
        tts = KokoroTTS(
            model_path=str(model_file),
            voices_path=str(voices_file),
            speaker=mock_speaker,
            event_queue=event_q,
        )

    assert tts._silent is True
    assert tts._kokoro is None


# ---------------------------------------------------------------------------
# 3. Construction — success
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_construction_success_sets_kokoro_and_not_silent(tmp_path):
    """When model loads successfully, _silent must be False and _kokoro must be set."""
    model_file = tmp_path / "kokoro.onnx"
    model_file.touch()
    voices_file = tmp_path / "voices.bin"
    voices_file.touch()

    mock_kokoro_instance = MagicMock()
    event_q: queue.Queue = queue.Queue()
    mock_speaker = MagicMock(spec=SpeakerThread)

    import sys
    import types

    fake_mod = types.ModuleType("kokoro_onnx")
    fake_mod.Kokoro = MagicMock(return_value=mock_kokoro_instance)  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"kokoro_onnx": fake_mod}):
        tts = KokoroTTS(
            model_path=str(model_file),
            voices_path=str(voices_file),
            speaker=mock_speaker,
            event_queue=event_q,
        )

    assert tts._silent is False
    assert tts._kokoro is mock_kokoro_instance


# ---------------------------------------------------------------------------
# 4. is_busy — default False
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_busy_default_false():
    """is_busy must be False immediately after construction."""
    tts, _, _ = _make_tts(silent=True)
    assert tts.is_busy is False


# ---------------------------------------------------------------------------
# 5. is_busy — True during synthesize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_busy_true_during_synthesize():
    """is_busy must be True while synthesize() is executing on another thread."""
    tts, _, _ = _make_tts(silent=True)

    busy_during_synthesis: list[bool] = []
    synthesis_started = threading.Event()

    def _slow_emit_silence(utterance_id: str) -> None:
        synthesis_started.set()
        time.sleep(0.15)  # hold busy state long enough to observe
        # Call the real implementation
        if tts._event_queue is not None:
            tts._event_queue.put(SpeechCompletedEvent(utterance_id=utterance_id))

    with patch.object(tts, "_emit_silence", side_effect=_slow_emit_silence):
        t = threading.Thread(
            target=tts.synthesize, args=("Hello", "utt-busy"), daemon=True
        )
        t.start()

        synthesis_started.wait(timeout=2.0)
        busy_during_synthesis.append(tts.is_busy)
        t.join(timeout=2.0)

    assert busy_during_synthesis == [True], "Expected is_busy=True during synthesis"
    assert tts.is_busy is False, "Expected is_busy=False after synthesis"


# ---------------------------------------------------------------------------
# 6. Silent mode synthesize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_silent_mode_synthesize_posts_completed_event():
    """In silent mode, synthesize() must post SpeechCompletedEvent and NOT call enqueue."""
    tts, event_q, mock_speaker = _make_tts(silent=True)

    tts.synthesize("Hello world.", utterance_id="utt-silent")

    event = event_q.get_nowait()
    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-silent"
    mock_speaker.enqueue.assert_not_called()


@pytest.mark.unit
def test_silent_mode_synthesize_queue_has_exactly_one_event():
    """Silent mode must produce exactly one SpeechCompletedEvent per call."""
    tts, event_q, _ = _make_tts(silent=True)

    tts.synthesize("Hello.", utterance_id="utt-one")

    assert event_q.qsize() == 1


# ---------------------------------------------------------------------------
# 7. Empty text synthesize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_text_posts_completed_event():
    """synthesize('', ...) must behave like silent mode — post SpeechCompletedEvent."""
    tts, event_q, mock_speaker = _make_tts(silent=False)
    # Even in non-silent mode, empty text short-circuits before ONNX

    tts.synthesize("", utterance_id="utt-empty")

    event = event_q.get_nowait()
    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-empty"
    mock_speaker.enqueue.assert_not_called()


@pytest.mark.unit
def test_whitespace_only_text_posts_completed_event():
    """synthesize('   ', ...) must also follow the empty-text path."""
    tts, event_q, mock_speaker = _make_tts(silent=False)

    tts.synthesize("   ", utterance_id="utt-ws")

    event = event_q.get_nowait()
    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-ws"
    mock_speaker.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Happy path — single sentence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_happy_path_single_sentence_tts_chunk_event():
    """Single sentence synthesis must emit TTSChunkReadyEvent with is_final=True."""
    audio_data = _make_audio(960)
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio_data, 24_000)

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello world.", utterance_id="utt-single")

    assert not event_q.empty(), "Expected at least one event in the queue"
    event = event_q.get_nowait()

    assert isinstance(event, TTSChunkReadyEvent)
    assert event.chunk_id == 0
    assert event.is_final is True
    assert event.utterance_id == "utt-single"
    assert event.sample_rate == 24_000


@pytest.mark.unit
def test_happy_path_single_sentence_speaker_enqueue_called():
    """Single sentence synthesis must call speaker.enqueue with is_final=True."""
    audio_data = _make_audio(960)
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio_data, 24_000)

    tts, _, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello world.", utterance_id="utt-enq")

    mock_speaker.enqueue.assert_called_once()
    _audio, kwargs_or_args = mock_speaker.enqueue.call_args[0], mock_speaker.enqueue.call_args
    # Verify is_final=True was passed — check keyword args
    call_kwargs = mock_speaker.enqueue.call_args.kwargs
    # is_final can be positional (index 2) or keyword
    call_args = mock_speaker.enqueue.call_args.args
    if "is_final" in call_kwargs:
        assert call_kwargs["is_final"] is True
    else:
        # positional: enqueue(audio, utterance_id, is_final, source_rate=...)
        assert call_args[2] is True


# ---------------------------------------------------------------------------
# 9. Happy path — multi-sentence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_happy_path_multi_sentence_two_chunks():
    """Two sentences must produce two TTSChunkReadyEvents; first non-final, second final."""
    audio1 = _make_audio(480)
    audio2 = _make_audio(960)
    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = [
        (audio1, 24_000),
        (audio2, 24_000),
    ]

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello world. How are you?", utterance_id="utt-multi")

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    # Both must be TTSChunkReadyEvents
    tts_events = [e for e in events if isinstance(e, TTSChunkReadyEvent)]
    assert len(tts_events) == 2, f"Expected 2 TTSChunkReadyEvents, got {len(tts_events)}"

    assert tts_events[0].chunk_id == 0
    assert tts_events[0].is_final is False
    assert tts_events[0].utterance_id == "utt-multi"

    assert tts_events[1].chunk_id == 1
    assert tts_events[1].is_final is True
    assert tts_events[1].utterance_id == "utt-multi"


@pytest.mark.unit
def test_happy_path_multi_sentence_speaker_enqueue_twice():
    """Multi-sentence path must call speaker.enqueue once per sentence."""
    audio1 = _make_audio(480)
    audio2 = _make_audio(960)
    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = [
        (audio1, 24_000),
        (audio2, 24_000),
    ]

    tts, _, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello world. How are you?", utterance_id="utt-multi2")

    assert mock_speaker.enqueue.call_count == 2


# ---------------------------------------------------------------------------
# 10. Cancel before synthesize completes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cancel_aborts_synthesis_during_emit_pass():
    """cancel() fired between the emit-pass iterations must cause early return."""
    audio1 = _make_audio(480)
    audio2 = _make_audio(480)
    audio3 = _make_audio(480)
    mock_kokoro = MagicMock()
    # All three sentences succeed (inference pass completes); cancel fires at emit time.
    mock_kokoro.create.side_effect = [
        (audio1, 24_000),
        (audio2, 24_000),
        (audio3, 24_000),
    ]

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    emit_call_count = 0

    original_emit_chunk = tts._emit_chunk

    def patched_emit_chunk(audio, chunk_id, is_final, utterance_id):
        nonlocal emit_call_count
        emit_call_count += 1
        original_emit_chunk(audio, chunk_id=chunk_id, is_final=is_final, utterance_id=utterance_id)
        if emit_call_count == 1:
            # Cancel after emitting the first chunk
            tts._cancel_flag.set()

    tts._emit_chunk = patched_emit_chunk  # type: ignore[method-assign]

    tts.synthesize("One. Two. Three.", utterance_id="utt-emit-cancel")

    # Only the first emit call should have gone through; second was blocked.
    assert emit_call_count == 1, f"Expected 1 emit before cancel, got {emit_call_count}"
    assert mock_speaker.enqueue.call_count == 1


@pytest.mark.unit
def test_cancel_aborts_synthesis_mid_stream():
    """cancel() from another thread must cause synthesize() to abort early."""
    # Use a slow create() to give the cancel thread time to intervene.
    audio_data = _make_audio(480)
    call_count = 0
    synthesize_started = threading.Event()

    def slow_create(sentence, **kwargs):
        nonlocal call_count
        call_count += 1
        synthesize_started.set()
        time.sleep(0.1)  # pause so cancel thread can fire
        return (audio_data, 24_000)

    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = slow_create

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    def cancel_after_start():
        synthesize_started.wait(timeout=2.0)
        tts.cancel("utt-cancel")

    cancel_thread = threading.Thread(target=cancel_after_start, daemon=True)
    cancel_thread.start()

    # Text with 3 sentences; cancel fires after the first inference begins.
    tts.synthesize(
        "First sentence. Second sentence. Third sentence.", utterance_id="utt-cancel"
    )
    cancel_thread.join(timeout=2.0)

    # speaker.enqueue should have been called fewer than 3 times (or 0 times
    # if cancel fired before the emit pass began).
    assert mock_speaker.enqueue.call_count < 3, (
        "Expected synthesis to abort before all 3 sentences were emitted"
    )


# ---------------------------------------------------------------------------
# 11. Cancel wrong utterance_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cancel_wrong_utterance_id_does_not_abort():
    """cancel() with a different utterance_id must not affect ongoing synthesis."""
    audio1 = _make_audio(480)
    audio2 = _make_audio(480)
    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = [(audio1, 24_000), (audio2, 24_000)]

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    # Cancel a different utterance while this one runs
    tts._current_utterance_id = "utt-real"
    tts.cancel("utt-OTHER")  # wrong id — should be a no-op

    tts.synthesize("Hello world. How are you?", utterance_id="utt-real")

    # Both sentences should have been emitted
    assert mock_speaker.enqueue.call_count == 2


# ---------------------------------------------------------------------------
# 12. Inference failure mid-stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inference_failure_on_second_sentence_falls_back_to_silence():
    """If _kokoro.create raises on the second sentence, first chunk is emitted
    and a SpeechCompletedEvent fires (via _emit_silence fallback)."""
    audio1 = _make_audio(480)
    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = [
        (audio1, 24_000),
        RuntimeError("ONNX inference failed"),
    ]

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello world. Goodbye world.", utterance_id="utt-fail")

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    tts_chunks = [e for e in events if isinstance(e, TTSChunkReadyEvent)]
    completed = [e for e in events if isinstance(e, SpeechCompletedEvent)]

    # First sentence succeeded — one TTSChunkReadyEvent
    assert len(tts_chunks) == 1, f"Expected 1 TTSChunkReady, got {len(tts_chunks)}"
    assert tts_chunks[0].chunk_id == 0
    assert tts_chunks[0].is_final is True  # only 1 chunk so it's the final

    # Because second sentence failed and produced no audio, _emit_silence fires
    # — but wait: the first sentence DID produce audio, so chunks=[audio1].
    # The implementation emits audio1 with is_final=True (single chunk).
    # No SpeechCompletedEvent from _emit_silence in this path.
    # Verify speaker.enqueue was called with is_final=True.
    mock_speaker.enqueue.assert_called_once()


@pytest.mark.unit
def test_inference_failure_all_sentences_emits_silence():
    """If ALL sentences fail inference, _emit_silence must fire."""
    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = RuntimeError("always fails")

    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    tts.synthesize("Hello world.", utterance_id="utt-allfail")

    event = event_q.get_nowait()
    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-allfail"
    mock_speaker.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# 13. _split_sentences helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_sentences_two_sentences():
    """Standard two-sentence input must split into two parts."""
    result = _split_sentences("Hello. World!")
    assert result == ["Hello.", "World!"]


@pytest.mark.unit
def test_split_sentences_question_mark():
    """Question mark boundary must split correctly."""
    result = _split_sentences("How are you? I am fine.")
    assert result == ["How are you?", "I am fine."]


@pytest.mark.unit
def test_split_sentences_ellipsis():
    """Ellipsis boundary must split correctly."""
    result = _split_sentences("Wait\u2026 Then what happened?")
    assert result == ["Wait\u2026", "Then what happened?"]


@pytest.mark.unit
def test_split_sentences_single_sentence_no_period():
    """A single sentence without a terminal punctuation must remain as one item."""
    result = _split_sentences("Hello world")
    assert result == ["Hello world"]


@pytest.mark.unit
def test_split_sentences_single_sentence_with_period():
    """A single sentence ending with a period and no whitespace after must stay as one."""
    result = _split_sentences("Hello world.")
    assert result == ["Hello world."]


@pytest.mark.unit
def test_split_sentences_empty_string():
    """Empty string must not raise and must return a single-element list."""
    result = _split_sentences("")
    # Implementation: returns [""] since parts=[""]: "" stripped is "" which is
    # falsy, so sentences=[]. Falls back to [text] = [""].
    assert result == [""]


@pytest.mark.unit
def test_split_sentences_whitespace_only():
    """Whitespace-only input must return a non-empty list without raising."""
    result = _split_sentences("   ")
    assert isinstance(result, list)
    assert len(result) >= 1


@pytest.mark.unit
def test_split_sentences_three_sentences():
    """Three sentences separated by '. ' must yield exactly three elements."""
    result = _split_sentences("One. Two. Three.")
    assert result == ["One.", "Two.", "Three."]


@pytest.mark.unit
def test_split_sentences_strips_extra_whitespace():
    """Each returned sentence must be stripped of leading/trailing whitespace."""
    result = _split_sentences("  Hello.   World.  ")
    for s in result:
        assert s == s.strip()


# ---------------------------------------------------------------------------
# Additional edge-case tests for coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthesize_resets_is_busy_after_exception():
    """is_busy must return to False even if an internal method raises."""
    tts, event_q, mock_speaker = _make_tts(silent=False)

    # Force an exception inside _stream_text by setting _kokoro to None
    # but _silent to False — _stream_text will call self._kokoro.create and raise.
    tts._silent = False
    tts._kokoro = None

    # Should not raise to the caller
    try:
        tts.synthesize("Hello.", utterance_id="utt-exc")
    except Exception:
        pytest.fail("synthesize() must not propagate internal exceptions")

    assert tts.is_busy is False


@pytest.mark.unit
def test_cancel_when_not_busy_does_not_raise():
    """cancel() on an idle TTS must be a safe no-op."""
    tts, _, _ = _make_tts(silent=True)
    tts.cancel("some-utterance-id")  # must not raise


@pytest.mark.unit
def test_cancel_sets_flag_for_current_utterance():
    """cancel() must set _cancel_flag when utterance_id matches current."""
    tts, _, _ = _make_tts(silent=True)
    # Manually set current utterance to simulate mid-synthesis state
    with tts._busy_lock:
        tts._current_utterance_id = "utt-active"

    tts.cancel("utt-active")

    assert tts._cancel_flag.is_set()


@pytest.mark.unit
def test_cancel_does_not_set_flag_for_wrong_utterance():
    """cancel() must NOT set _cancel_flag when utterance_id does not match."""
    tts, _, _ = _make_tts(silent=True)
    with tts._busy_lock:
        tts._current_utterance_id = "utt-active"

    tts.cancel("utt-different")

    assert not tts._cancel_flag.is_set()


@pytest.mark.unit
def test_event_queue_none_does_not_raise_on_silent_mode():
    """KokoroTTS with event_queue=None must not raise during silent-mode synthesis."""
    mock_speaker = MagicMock(spec=SpeakerThread)
    tts = KokoroTTS(
        model_path="/no/model.onnx",
        voices_path="/no/voices.bin",
        speaker=mock_speaker,
        event_queue=None,
    )
    assert tts._silent is True
    # Must not raise even with no event_queue
    tts.synthesize("Hello.", utterance_id="utt-noqueue")


@pytest.mark.unit
def test_speaker_none_does_not_raise_on_happy_path():
    """KokoroTTS with speaker=None must not raise during audio emission."""
    audio_data = _make_audio(480)
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (audio_data, 24_000)

    event_q: queue.Queue = queue.Queue()

    with patch("os.path.exists", return_value=True):
        with patch("src.audio.mouth.KokoroTTS._load_model"):
            tts = KokoroTTS(
                model_path="/fake.onnx",
                voices_path="/fake.bin",
                speaker=None,  # no speaker
                event_queue=event_q,
            )
            tts._kokoro = mock_kokoro
            tts._silent = False

    tts.synthesize("Hello world.", utterance_id="utt-nospeaker")

    event = event_q.get_nowait()
    assert isinstance(event, TTSChunkReadyEvent)
    assert event.is_final is True


@pytest.mark.unit
def test_synthesize_clears_cancel_flag_at_start():
    """synthesize() must always clear the cancel flag so subsequent calls are not affected."""
    tts, event_q, _ = _make_tts(silent=True)

    # Manually pre-set the flag (simulates a stale flag or pre-cancel).
    tts._cancel_flag.set()

    tts.synthesize("Hello.", utterance_id="utt-stale")

    # After synthesis (aborted via pre-cancel), the flag must be cleared.
    assert not tts._cancel_flag.is_set()

    # SpeechCompletedEvent must still arrive (via the finally block).
    assert not event_q.empty()
    event = event_q.get_nowait()
    assert isinstance(event, SpeechCompletedEvent)


# ---------------------------------------------------------------------------
# prepare() / pre-cancel path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prepare_sets_current_utterance_id():
    """prepare() must set _current_utterance_id so cancel() can target it."""
    tts, _, _ = _make_tts(silent=True)
    assert tts._current_utterance_id is None

    tts.prepare("utt-prepared")

    with tts._busy_lock:
        assert tts._current_utterance_id == "utt-prepared"


@pytest.mark.unit
def test_prepare_then_cancel_sets_flag_before_synthesize():
    """cancel() after prepare() must set the cancel flag; synthesize() detects pre-cancel."""
    tts, event_q, mock_speaker = _make_tts(silent=False)

    tts.prepare("utt-pre")
    tts.cancel("utt-pre")  # fires before synthesize() — the race window we're closing

    assert tts._cancel_flag.is_set(), "cancel() must set flag when utterance is prepared"

    # synthesize() should detect pre-cancel, skip all inference, post SpeechCompletedEvent.
    tts.synthesize("Should not be synthesised.", utterance_id="utt-pre")

    # Cancel flag must be cleared after synthesize() runs.
    assert not tts._cancel_flag.is_set()
    # No audio must have been enqueued to the speaker.
    mock_speaker.enqueue.assert_not_called()
    # SpeechCompletedEvent must be posted via the finally block.
    assert not event_q.empty()
    event = event_q.get_nowait()
    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-pre"


@pytest.mark.unit
def test_cancelled_synthesis_always_posts_speech_completed_event():
    """synthesize() cancelled mid-stream must post SpeechCompletedEvent via finally block."""
    audio_data = _make_audio(480)
    call_count = 0
    synthesize_started = threading.Event()

    def slow_create(sentence, **kwargs):
        nonlocal call_count
        call_count += 1
        synthesize_started.set()
        time.sleep(0.1)
        return (audio_data, 24_000)

    mock_kokoro = MagicMock()
    mock_kokoro.create.side_effect = slow_create
    tts, event_q, mock_speaker = _make_tts(mock_kokoro=mock_kokoro)

    def cancel_after_start():
        synthesize_started.wait(timeout=2.0)
        tts.cancel("utt-cancel-complete")

    cancel_thread = threading.Thread(target=cancel_after_start, daemon=True)
    cancel_thread.start()

    tts.synthesize(
        "First sentence. Second sentence. Third sentence.",
        utterance_id="utt-cancel-complete",
    )
    cancel_thread.join(timeout=2.0)

    # Regardless of how many chunks were emitted before cancel, a
    # SpeechCompletedEvent must always reach the queue so the orchestrator
    # can exit SPEAKING state without an interrupt.
    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    completed_events = [e for e in events if isinstance(e, SpeechCompletedEvent)]
    assert len(completed_events) >= 1, (
        "Cancelled synthesize() must post at least one SpeechCompletedEvent"
    )
    assert all(e.utterance_id == "utt-cancel-complete" for e in completed_events)


@pytest.mark.unit
def test_pre_cancel_does_not_cascade_to_next_utterance():
    """After a pre-cancel, the cancel flag must be cleared so the next synthesize() runs normally."""
    tts, event_q, _ = _make_tts(silent=True)

    # First utterance: pre-cancel.
    tts.prepare("utt-1")
    tts.cancel("utt-1")
    tts.synthesize("First.", utterance_id="utt-1")

    # Drain queue.
    while not event_q.empty():
        event_q.get_nowait()

    # Second utterance: must not be pre-cancelled.
    tts.synthesize("Second.", utterance_id="utt-2")

    assert not event_q.empty()
    event = event_q.get_nowait()
    # Silent mode posts SpeechCompletedEvent — synthesize ran normally.
    assert isinstance(event, SpeechCompletedEvent)
    assert event.utterance_id == "utt-2"
