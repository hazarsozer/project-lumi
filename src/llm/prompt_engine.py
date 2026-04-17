"""Prompt engine for assembling ChatML-formatted prompts."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are Lumi, a helpful local voice assistant."


class PromptEngine:
    """Builds ChatML prompts with history truncation."""

    def build_prompt(
        self,
        user_text: str,
        history: list[dict[str, str]],
        system_prompt: str | None = None,
        rag_context: str = "",
    ) -> str:
        """Assemble a ChatML prompt from system prompt, history, and user input.

        When *rag_context* is non-empty it is injected between the system
        prompt and conversation history as a ``[Relevant notes]`` block so
        the LLM can cite retrieved passages without confusing them with
        hard facts stated by the system.
        """
        sys = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        sys_block = f"{sys}\n\n[Relevant notes]\n{rag_context}" if rag_context else sys
        parts: list[str] = [f"<|system|>\n{sys_block}<|end|>"]

        for turn in history:
            role = turn["role"]
            content = turn["content"]
            parts.append(f"<|{role}|>\n{content}<|end|>")

        parts.append(f"<|user|>\n{user_text}<|end|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def truncate_history(
        self,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """Remove oldest turns until estimated token count fits within budget.

        Returns a new list; the input is never mutated.
        Token estimate: len(content) // 4 per turn.
        """
        if not history:
            return []

        result = list(history)

        while len(result) > 1:
            total = sum(len(t["content"]) // 4 for t in result)
            if total <= max_tokens:
                break
            result = result[1:]

        return result
