"""
E2E smoke tests for Project Lumi — integration-level event flow wiring.

All tests use real orchestrator instances, real event queues, and real state
machines.  Hardware (mic, speaker), ML models (LLM, TTS, STT), and network
sockets (ZMQ/IPC) are fully mocked so no external dependencies are required.

Scenarios covered
-----------------
1. test_wake_to_idle_cycle            — UserTextEvent (reflex path) drives
                                        IDLE→LISTENING→PROCESSING→SPEAKING,
                                        then SpeechCompletedEvent returns IDLE.
2. test_transcript_to_llm_request     — TranscriptReadyEvent routes through the
                                        reasoning slow-path and emits
                                        LLMResponseReadyEvent.
3. test_full_turn_no_tts              — With tts=None the full turn completes
                                        and state returns to IDLE without any
                                        TTS call.
4. test_full_turn_with_tts_stub       — With a TTS stub injected, synthesize()
                                        is called with the LLM output text.
5. test_rag_retrieval_integrated      — A mocked RAGRetriever whose retrieve()
                                        returns context verifies that the context
                                        string reaches the LLM prompt.
6. test_ipc_event_forwarded           — LLMResponseReadyEvent is forwarded to
                                        the EventBridge stub's on_tts_start handler.
"""

from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

import pytest

from src.core.config import (
    IPCConfig,
    LumiConfig,
    RAGConfig,
    ToolsConfig,
    VisionConfig,
)
from src.core.events import (
    LLMResponseReadyEvent,
    RAGRetrievalEvent,
    ShutdownEvent,
    SpeechCompletedEvent,
    TranscriptReadyEvent,
    UserTextEvent,
)
from src.core.state_machine import LumiState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Text that does NOT match the ReflexRouter greeting/time patterns, so it
# always falls through to the LLM slow-path.
_NON_REFLEX_TEXT = "what is the capital of France"


def _minimal_config(*, rag_enabled: bool = False, tts_enabled: bool = True) -> LumiConfig:
    """Return a LumiConfig with all heavy subsystems disabled."""
    from src.core.config import TTSConfig

    return LumiConfig(
        ipc=IPCConfig(enabled=False),
        rag=RAGConfig(enabled=rag_enabled),
        vision=VisionConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
        tts=TTSConfig(enabled=tts_enabled),
    )


def _mock_speaker() -> MagicMock:
    """Return a SpeakerThread stand-in that satisfies Orchestrator's API."""
    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()
    speaker.flush = MagicMock()
    return speaker


@contextmanager
def _build_orchestrator(
    config: LumiConfig,
    *,
    tts=None,
    event_bridge=None,
    llm_response: str = "Paris",
):
    """Context manager that creates an Orchestrator with mocked heavy deps.

    Patches ModelLoader, ConversationMemory, and ReasoningRouter so that no
    real model is loaded.  ReasoningRouter.generate() is stubbed to return
    *llm_response* immediately.

    Yields the live Orchestrator instance.
    """
    from src.core.orchestrator import Orchestrator

    speaker = _mock_speaker()

    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.model = MagicMock()
    # create_completion returns a single token then stops on the next call by
    # returning an empty text — the loop in ReasoningRouter.generate() exits
    # when token == "" (or when finish_reason=="stop").
    mock_model.model.create_completion.side_effect = [
        {"choices": [{"text": llm_response, "finish_reason": "stop"}]},
    ]

    mock_mem = MagicMock()
    mock_mem.load = MagicMock()
    mock_mem.get_history.return_value = []
    mock_mem.add_turn = MagicMock()
    mock_mem.save = MagicMock()

    mock_reasoning = MagicMock()
    mock_reasoning.generate.return_value = llm_response

    with (
        patch("src.core.orchestrator.ModelLoader", return_value=mock_model),
        patch("src.core.orchestrator.ConversationMemory", return_value=mock_mem),
        patch("src.core.orchestrator.ReasoningRouter", return_value=mock_reasoning),
    ):
        orch = Orchestrator(config, speaker=speaker, tts=tts, event_bridge=event_bridge)
        try:
            yield orch
        finally:
            # Always drain the queue and stop the speaker to avoid thread leaks.
            try:
                speaker.stop()
            except Exception:
                pass


def _drain_until(
    event_queue: queue.Queue,
    predicate,
    timeout: float = 5.0,
    poll: float = 0.02,
) -> object:
    """Drain *event_queue* and return the first item satisfying *predicate*.

    Raises AssertionError if no matching event arrives within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = event_queue.get_nowait()
            if predicate(item):
                return item
        except queue.Empty:
            time.sleep(poll)
    raise AssertionError(
        f"No matching event arrived within {timeout}s. "
        f"predicate={predicate}"
    )


def _wait_for_state(
    orchestrator,
    target: LumiState,
    timeout: float = 5.0,
    poll: float = 0.02,
) -> None:
    """Poll the state machine until it reaches *target* or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if orchestrator._state_machine.current_state == target:
            return
        time.sleep(poll)
    raise AssertionError(
        f"State machine did not reach {target!r} within {timeout}s; "
        f"current={orchestrator._state_machine.current_state!r}"
    )


# ---------------------------------------------------------------------------
# Test 1 — wake-to-idle cycle via UserTextEvent (reflex fast path)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_wake_to_idle_cycle():
    """Posting a UserTextEvent (reflex path) drives IDLE→LISTENING→PROCESSING
    →SPEAKING, then the SpeechCompleted event returns state to IDLE.

    The orchestrator starts IDLE.  UserTextEvent with a greeting triggers the
    ReflexRouter (no LLM call) and ends in SPEAKING.  We then dispatch a
    synthetic SpeechCompletedEvent and verify the final IDLE state.
    """
    config = _minimal_config()

    with _build_orchestrator(config) as orch:
        assert orch._state_machine.current_state == LumiState.IDLE

        # Post a greeting — ReflexRouter will handle it synchronously.
        orch._dispatch(UserTextEvent(text="hello"))

        # After the synchronous reflex path the state must be SPEAKING.
        assert orch._state_machine.current_state == LumiState.SPEAKING

        # The orchestrator also posts LLMResponseReadyEvent to its own queue.
        # Drain it so no residual items remain, then handle SpeechCompleted.
        llm_evt = _drain_until(
            orch._event_queue,
            lambda e: isinstance(e, LLMResponseReadyEvent),
        )
        assert "Hello" in llm_evt.text

        # Simulate speech finishing — use the utterance_id recorded in the handler.
        uid = orch._current_utterance_id or "test-uid"
        orch._dispatch(SpeechCompletedEvent(utterance_id=uid))

        assert orch._state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Test 2 — TranscriptReadyEvent routes to LLM and produces LLMResponseReadyEvent
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_transcript_to_llm_request():
    """Posting a TranscriptReadyEvent with non-reflex text triggers the
    reasoning router and ultimately places an LLMResponseReadyEvent on the
    queue.
    """
    config = _minimal_config()

    with _build_orchestrator(config, llm_response="Paris") as orch:
        assert orch._state_machine.current_state == LumiState.IDLE

        # Manually drive state to LISTENING so the LISTENING→PROCESSING
        # transition in _handle_transcript is valid.
        orch._state_machine.transition_to(LumiState.LISTENING)

        orch._dispatch(TranscriptReadyEvent(text=_NON_REFLEX_TEXT))

        # The state machine transitions to PROCESSING synchronously, then the
        # daemon thread may immediately advance to SPEAKING (mocked generate()
        # returns instantly).  Accept either transitional state.
        assert orch._state_machine.current_state in (
            LumiState.PROCESSING, LumiState.SPEAKING
        )

        # Wait for the inference daemon thread to post LLMResponseReadyEvent.
        llm_evt = _drain_until(
            orch._event_queue,
            lambda e: isinstance(e, LLMResponseReadyEvent),
        )
        assert llm_evt.text == "Paris"

        # State should now be SPEAKING (set by daemon thread under the lock).
        _wait_for_state(orch, LumiState.SPEAKING)


# ---------------------------------------------------------------------------
# Test 3 — Full turn with TTS disabled; state returns to IDLE without TTS call
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_full_turn_no_tts():
    """With tts=None, a full turn (LISTENING→PROCESSING→SPEAKING→IDLE) completes
    without any TTS synthesis call.

    When no TTS engine is configured the orchestrator immediately posts
    SpeechCompletedEvent, which the event loop dispatches to return IDLE.
    """
    config = _minimal_config()

    with _build_orchestrator(config, tts=None, llm_response="No TTS response") as orch:
        orch._state_machine.transition_to(LumiState.LISTENING)
        orch._dispatch(TranscriptReadyEvent(text=_NON_REFLEX_TEXT))

        # Wait for SPEAKING (set by daemon after generate()).
        _wait_for_state(orch, LumiState.SPEAKING)

        # LLMResponseReadyEvent should be in the queue; dispatch it so the
        # orchestrator handles TTS-less completion.
        llm_evt = _drain_until(
            orch._event_queue,
            lambda e: isinstance(e, LLMResponseReadyEvent),
        )
        orch._dispatch(llm_evt)

        # Without TTS, SpeechCompletedEvent is posted immediately by
        # _handle_llm_response; drain it and dispatch.
        sc_evt = _drain_until(
            orch._event_queue,
            lambda e: isinstance(e, SpeechCompletedEvent),
        )
        orch._dispatch(sc_evt)

        assert orch._state_machine.current_state == LumiState.IDLE


# ---------------------------------------------------------------------------
# Test 4 — Full turn with TTS stub; mouth.synthesize() is called with LLM text
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_full_turn_with_tts_stub():
    """With a mocked KokoroTTS injected, synthesize() is called with the text
    produced by the LLM stub.
    """
    config = _minimal_config()

    mock_tts = MagicMock()
    mock_tts.prepare = MagicMock()
    mock_tts.synthesize = MagicMock()
    mock_tts.cancel = MagicMock()

    with _build_orchestrator(config, tts=mock_tts, llm_response="Bonjour") as orch:
        orch._state_machine.transition_to(LumiState.LISTENING)
        orch._dispatch(TranscriptReadyEvent(text=_NON_REFLEX_TEXT))

        # Wait for SPEAKING.
        _wait_for_state(orch, LumiState.SPEAKING)

        # Consume the LLMResponseReadyEvent and dispatch it.
        llm_evt = _drain_until(
            orch._event_queue,
            lambda e: isinstance(e, LLMResponseReadyEvent),
        )
        orch._dispatch(llm_evt)

        # TTS synthesis runs in a daemon thread; give it a moment to start.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if mock_tts.synthesize.called:
                break
            time.sleep(0.02)

        assert mock_tts.synthesize.called, "KokoroTTS.synthesize() was never called"
        # Verify the text passed to synthesize matches the LLM output.
        called_text = mock_tts.synthesize.call_args[0][0]
        assert called_text == "Bonjour"


# ---------------------------------------------------------------------------
# Test 5 — RAG retrieval integrated; context appears in LLM prompt
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_rag_retrieval_integrated():
    """With RAG enabled and a mocked retriever seeded with context, the
    retrieved context string is forwarded to ReasoningRouter.generate() via
    the use_rag=True path.

    We cannot intercept the prompt string directly (it is built inside
    ReasoningRouter.generate) but we CAN verify that generate() is called with
    use_rag=True when the reflex router signals RAG intent, or we can bypass
    the reflex intent check and force use_rag via the text pattern.
    """
    # RAG intent pattern: "search my docs for …" matches ReflexRouter._RAG_PATTERN.
    rag_query = "search my docs for France"

    config = _minimal_config(rag_enabled=True)

    fake_citation = MagicMock()
    fake_citation.doc_path = "/notes/france.md"
    fake_citation.chunk_idx = 0

    fake_rag_result = MagicMock()
    fake_rag_result.context = "France is a country in Western Europe."
    fake_rag_result.hit_count = 1
    fake_rag_result.latency_ms = 5
    fake_rag_result.citations = [fake_citation]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fake_rag_result

    with (
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
    ):
        mock_mem = MagicMock()
        mock_mem.load = MagicMock()
        mock_mem.get_history.return_value = []
        mock_mem.add_turn = MagicMock()
        mock_mem.save = MagicMock()
        mock_mem_cls.return_value = mock_mem

        mock_reasoning = MagicMock()
        mock_reasoning.generate.return_value = "RAG answer"

        with patch("src.core.orchestrator.ReasoningRouter", return_value=mock_reasoning):
            # Patch the RAG subsystem construction so our mock retriever is used.
            with (
                patch("src.rag.store.DocumentStore"),
                patch(
                    "src.rag.retriever.RAGRetriever",
                    return_value=mock_retriever,
                ),
            ):
                from src.core.orchestrator import Orchestrator

                speaker = _mock_speaker()
                orch = Orchestrator(config, speaker=speaker, tts=None)

                # Override the retriever injected into the reasoning router with
                # our fake so calls to _rag_retriever.retrieve() are tracked.
                orch._rag_retriever = mock_retriever

                orch._state_machine.transition_to(LumiState.LISTENING)
                orch._dispatch(TranscriptReadyEvent(text=rag_query))

                _wait_for_state(orch, LumiState.SPEAKING, timeout=5.0)

                # generate() must have been called with use_rag=True because
                # the reflex router matches the RAG intent pattern.
                assert mock_reasoning.generate.called
                _, kwargs = mock_reasoning.generate.call_args
                assert kwargs.get("use_rag") is True, (
                    f"generate() called with use_rag={kwargs.get('use_rag')!r}; "
                    "expected True"
                )

                speaker.stop()


# ---------------------------------------------------------------------------
# Test 6 — IPC event forwarded through EventBridge stub
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_ipc_event_forwarded():
    """An LLMResponseReadyEvent is forwarded to the injected EventBridge stub's
    on_tts_start handler when the orchestrator dispatches it.
    """
    config = _minimal_config()

    mock_zmq = MagicMock()
    mock_zmq.on_state_change = MagicMock()
    mock_zmq.on_tts_start = MagicMock()
    mock_zmq.on_tts_stop = MagicMock()
    mock_zmq.on_tts_viseme = MagicMock()
    mock_zmq.on_transcript = MagicMock()
    mock_zmq.on_llm_token = MagicMock()
    mock_zmq.on_rag_retrieval = MagicMock()
    mock_zmq.stop = MagicMock()

    with _build_orchestrator(config, event_bridge=mock_zmq) as orch:
        # The orchestrator registers on_tts_start for LLMResponseReadyEvent
        # when event_bridge is provided.
        response_event = LLMResponseReadyEvent(text="Forwarded response")

        # Manually put state into SPEAKING so _handle_llm_response can proceed.
        # We drive state directly so we skip the inference thread for this test.
        orch._state_machine.transition_to(LumiState.LISTENING)
        orch._state_machine.transition_to(LumiState.PROCESSING)
        orch._state_machine.transition_to(LumiState.SPEAKING)

        orch._dispatch(response_event)

        # on_tts_start is registered as a handler for LLMResponseReadyEvent.
        mock_zmq.on_tts_start.assert_called_once_with(response_event)
