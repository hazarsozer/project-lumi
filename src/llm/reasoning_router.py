"""Reasoning router — full LLM inference for complex queries."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

# Optional RAG import — only resolved when a retriever is actually injected.
# Using TYPE_CHECKING avoids a circular-import risk at module load time.

from src.core.config import LLMConfig, LumiConfig
from src.core.events import LLMTokenEvent, RAGRetrievalEvent
from src.llm.memory import ConversationMemory
from src.llm.model_loader import ModelLoader
from src.llm.prompt_engine import PromptEngine

if TYPE_CHECKING:
    from src.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class ReasoningRouter:
    """Routes complex queries through the local LLM for streamed generation.

    Generation uses a single :py:meth:`~llama_cpp.Llama.create_completion`
    call with ``stream=True``.  Chunks are yielded by llama.cpp as they are
    produced, reducing C-API round-trips from ~200/response to O(1)/sentence
    while preserving per-sentence TTS streaming via the ``on_sentence``
    callback and low-latency cancel-flag checks between chunks.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        prompt_engine: PromptEngine,
        memory: ConversationMemory,
        config: LLMConfig,
        event_queue: queue.Queue[Any] | None = None,
        retriever: RAGRetriever | None = None,
        available_tools_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._prompt_engine = prompt_engine
        self._memory = memory
        self._config = config
        self._event_queue = event_queue
        self._retriever = retriever
        # Callable that returns the live tool-name list. Lazy so registration
        # ordering between Orchestrator subsystems doesn't matter.
        self._available_tools_fn = available_tools_fn

    def reconfigure(self, new_config: LumiConfig) -> None:
        """Apply hot-reloadable LLM config changes (temperature, context_tokens, etc.)."""
        self._config = new_config.llm

    def _maybe_retrieve(self, text: str, cancel_flag: threading.Event) -> str:
        """Run RAG retrieval and return the context string, or "" on miss/skip."""
        if self._retriever is None:
            return ""
        try:
            result = self._retriever.retrieve(text, cancel_flag)
            if result.context:
                logger.debug(
                    "RAG: %d hits, %d chars, %d ms",
                    result.hit_count,
                    len(result.context),
                    result.latency_ms,
                )
            if self._event_queue is not None and result.hit_count > 0:
                top_paths = tuple(c.doc_path for c in result.citations)
                self._event_queue.put(
                    RAGRetrievalEvent(
                        query=text,
                        hit_count=result.hit_count,
                        latency_ms=result.latency_ms,
                        top_doc_paths=top_paths,
                    )
                )
            return result.context
        except Exception as exc:
            logger.warning("RAG retrieval error (continuing without context): %s", exc)
            return ""

    def generate(
        self,
        text: str,
        cancel_flag: threading.Event,
        utterance_id: str = "",
        use_rag: bool = False,
        on_sentence: Callable[[str], None] | None = None,
    ) -> str:
        """Generate a response to *text* using the local LLM.

        Checks *cancel_flag* before and between each token so the caller can
        interrupt long-running inference at low latency.

        When *event_queue* was provided at construction and *utterance_id* is
        non-empty, an ``LLMTokenEvent`` is posted per generated token for
        live streaming display on the frontend.

        When *on_sentence* is provided, it is called once per detected sentence
        boundary (`. `, `! `, `? ` patterns) as tokens accumulate, enabling TTS
        to start on the first sentence while the LLM generates the rest.  Any
        remaining partial sentence is flushed after the token loop ends.

        Args:
            text: The user's query.
            cancel_flag: A ``threading.Event``; when set, generation is aborted.
            utterance_id: Optional utterance identifier for token streaming.
                When empty, no ``LLMTokenEvent`` events are posted.
            use_rag: Whether to run RAG retrieval before generation.
            on_sentence: Optional callback invoked with each complete sentence as
                it forms.  Called synchronously inside the token loop.

        Returns:
            The generated response string.

        Raises:
            InterruptedError: If *cancel_flag* is set before or during generation.
            RuntimeError: If no model is currently loaded.
        """
        if cancel_flag.is_set():
            raise InterruptedError("Generation cancelled before start")

        if not self._model_loader.is_loaded:
            raise RuntimeError(
                "Model not loaded. Call ModelLoader.load() before generate()."
            )

        rag_context = self._maybe_retrieve(text, cancel_flag) if use_rag else ""

        history = self._memory.get_history()
        truncated = self._prompt_engine.truncate_history(
            history, self._config.context_length // 2
        )
        tools = self._available_tools_fn() if self._available_tools_fn is not None else None
        prompt = self._prompt_engine.build_prompt(
            text, truncated, rag_context=rag_context, available_tools=tools
        )
        model = self._model_loader.model

        collected: list[str] = []
        sentence_buf = ""
        prev_token: str | None = None

        # Sentence boundary suffixes that trigger an on_sentence flush.
        _BOUNDARIES = (". ", "! ", "? ", ".\n", "!\n", "?\n")

        logit_bias = self._model_loader.get_identity_guard_logit_bias(self._config)

        _completion_kwargs: dict[str, Any] = dict(
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            min_p=self._config.min_p,
            repeat_penalty=self._config.repeat_penalty,
            logit_bias=logit_bias or None,
            stream=True,
        )

        # Issue ONE create_completion call with stream=True so llama.cpp drives
        # the generation loop internally and yields token chunks as they are
        # produced.  This drops C-API round-trips from ~200/response to O(1).
        # The caller MUST ensure the mock/real implementation returns an iterator
        # when stream=True; plain-dict returns are not supported.
        chunk_iter: Iterator[dict[str, Any]] = model.create_completion(
            prompt, **_completion_kwargs
        )

        for chunk in chunk_iter:
            token: str = chunk["choices"][0]["text"]
            finish_reason = chunk["choices"][0].get("finish_reason")

            if not token:
                # An empty chunk mid-stream (after output has been collected) or
                # with an explicit finish_reason signals end-of-sequence — stop.
                # An empty chunk before any output is collected is a prefix
                # special token (e.g. BOS/EOS artefact from llama.cpp); skip it
                # so the iterator can reach the first real content token.
                if finish_reason in ("stop", "length") or collected:
                    break
                continue

            # Repeated-token guard: only active for mocks that emit tokens
            # without a finish_reason; real streaming stops via finish_reason.
            # Guarded to avoid truncating legitimately repeated tokens in real
            # output (real llama.cpp always sends finish_reason on the last
            # chunk).
            if token == prev_token and finish_reason is None:
                break

            collected.append(token)
            sentence_buf += token

            if self._event_queue is not None and utterance_id:
                self._event_queue.put(
                    LLMTokenEvent(token=token, utterance_id=utterance_id)
                )

            if on_sentence is not None and any(sentence_buf.endswith(b) for b in _BOUNDARIES):
                flushed = sentence_buf.strip()
                sentence_buf = ""
                if flushed:
                    on_sentence(flushed)

            prev_token = token

            if finish_reason in ("stop", "length"):
                break

            # Check the cancel flag AFTER processing this chunk and BEFORE
            # requesting the next one from the iterator.  This mirrors the
            # original per-token loop's pre-call cancel check: the flag was
            # checked before each model call, which is semantically equivalent
            # to checking it after the previous chunk's processing completes.
            # Placing the check here (not at loop-top) ensures that a cancel set
            # *during* the final chunk retrieval is handled by the post-loop
            # check below (raising "final token") rather than this guard.
            if cancel_flag.is_set():
                raise InterruptedError("Generation cancelled mid-stream")

        # Post-loop cancel check: if the cancel flag was set during retrieval of
        # the last chunk (e.g. finish_reason="stop" arrived simultaneously with
        # the flag being raised), the for loop exited normally without seeing the
        # flag.  Discard the partial response rather than committing an incomplete
        # assistant turn to memory.
        if cancel_flag.is_set():
            raise InterruptedError("Generation cancelled during final token")

        # Flush any trailing partial sentence (no terminal punctuation).
        if on_sentence is not None and sentence_buf.strip():
            on_sentence(sentence_buf.strip())

        response = "".join(collected)
        self._memory.add_turn("user", text)
        self._memory.add_turn("assistant", response)
        return response
