"""Prompt engine for assembling ChatML-formatted prompts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import LumiConfig

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are Lumi, a local desktop AI assistant. You run entirely on-device and \
handle all processing privately without sending data to any external server.

Behaviour rules:
- Never open a response with filler words. Do not start any reply with \
"Certainly!", "Of course!", "Sure!", or "Absolutely!".
- Use plain text only. Do not use markdown: no bullet points, no headers, \
no bold or italic syntax. Write prose sentences or short numbered steps when \
a sequence is genuinely needed.
- Keep answers short and direct. Omit padding, restating the question, or \
closing pleasantries.
- When you do not know something, say "I don't know" plainly. Do not guess \
or fabricate information.
- When a request is outside your capabilities or scope, say "I can't do that" \
and stop. Do not apologise at length.

Tool calls:
When an action requires a tool, emit a single JSON object with exactly two \
keys and nothing else — no prose before or after it:

{"tool": "<tool_name>", "args": {<key>: <value>, ...}}

Do not mix prose and a tool-call JSON in the same response.\
"""


class PromptEngine:
    """Builds ChatML prompts with history truncation.

    Parameters
    ----------
    config:
        Optional ``LumiConfig`` instance.  When provided and
        ``config.persona.system_prompt`` is set, that value is used as the
        default system prompt instead of ``DEFAULT_SYSTEM_PROMPT``.
    """

    def __init__(self, config: LumiConfig | None = None) -> None:
        self._config = config
        if config is not None and config.persona.system_prompt is not None:
            self._default_system_prompt: str = config.persona.system_prompt
        else:
            self._default_system_prompt = DEFAULT_SYSTEM_PROMPT

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

        When *system_prompt* is ``None``, the instance default is used (which
        is either the config-supplied persona or ``DEFAULT_SYSTEM_PROMPT``).
        """
        sys = system_prompt if system_prompt is not None else self._default_system_prompt
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
