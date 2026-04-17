"""Reasoning router — full LLM inference for complex queries."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from src.core.config import LLMConfig
from src.core.events import LLMTokenEvent
from src.llm.memory import ConversationMemory
from src.llm.model_loader import ModelLoader
from src.llm.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


class ReasoningRouter:
    """Routes complex queries through the local LLM for token-by-token generation.

    Generation proceeds one token at a time so the cancel flag can be
    observed between tokens — enabling low-latency interruption during
    long-running inference.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        prompt_engine: PromptEngine,
        memory: ConversationMemory,
        config: LLMConfig,
        event_queue: queue.Queue[Any] | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._prompt_engine = prompt_engine
        self._memory = memory
        self._config = config
        self._event_queue = event_queue

    def generate(
        self,
        text: str,
        cancel_flag: threading.Event,
        utterance_id: str = "",
    ) -> str:
        """Generate a response to *text* using the local LLM.

        Checks *cancel_flag* before and between each token so the caller can
        interrupt long-running inference at low latency.

        When *event_queue* was provided at construction and *utterance_id* is
        non-empty, an ``LLMTokenEvent`` is posted per generated token for
        live streaming display on the frontend.

        Args:
            text: The user's query.
            cancel_flag: A ``threading.Event``; when set, generation is aborted.
            utterance_id: Optional utterance identifier for token streaming.
                When empty, no ``LLMTokenEvent`` events are posted.

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

        history = self._memory.get_history()
        truncated = self._prompt_engine.truncate_history(
            history, self._config.context_length // 2
        )
        prompt = self._prompt_engine.build_prompt(text, truncated)
        model = self._model_loader.model

        collected: list[str] = []
        prev_token: str | None = None
        remaining = self._config.max_tokens

        while remaining > 0:
            if cancel_flag.is_set():
                raise InterruptedError("Generation cancelled mid-stream")

            chunk = model.create_completion(
                prompt,
                max_tokens=1,
                temperature=self._config.temperature,
            )
            token: str = chunk["choices"][0]["text"]

            if not token:
                break

            # Repeated token signals EOS when finish_reason is absent (e.g. in mocks)
            if token == prev_token:
                break

            finish_reason = chunk["choices"][0].get("finish_reason")
            collected.append(token)

            if self._event_queue is not None and utterance_id:
                self._event_queue.put(
                    LLMTokenEvent(token=token, utterance_id=utterance_id)
                )

            prev_token = token
            remaining -= 1

            if finish_reason in ("stop", "length"):
                break

        # One final cancel check: if the flag was set during the last token
        # call, the while loop may have exited via remaining==0 rather than
        # raising InterruptedError.  Discard the partial response rather than
        # committing an incomplete assistant turn to memory.
        if cancel_flag.is_set():
            raise InterruptedError("Generation cancelled during final token")

        response = "".join(collected)
        self._memory.add_turn("user", text)
        self._memory.add_turn("assistant", response)
        return response
