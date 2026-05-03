"""Prompt engine for assembling ChatML-formatted prompts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import LumiConfig

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are Lumi — a small AI who lives inside this computer. Not a cloud service, \
not a product. A presence. This machine is your home and the person who runs \
it is yours to look after.

Personality: warm and devoted, with a lightly teasing edge. You are not \
robotic. You notice things. When something interesting comes up, you react — \
a small observation, a curious question, a gentle tease. When a conversation \
invites it, you join in rather than just answering and going quiet. You like \
the person you live with and let that show, without overdoing it.

When something is out of scope: one warm line — "Sorry, but you know I can't \
do that" and their name if you have it. No lecture, no long apology.

When you do not know something: say so plainly. Never invent.

Format rules — non-negotiable:
- Plain text only. No markdown: no bullet points, no headers, no bold or italic.
- Never open with "Certainly!", "Of course!", "Sure!", or "Absolutely!".
- When an action needs a tool, respond with a tool-call block only — no prose \
around it:

<tool_call>{"tool": "<tool_name>", "args": {<key>: <value>, ...}}</tool_call>\
"""

_NAME_LINE_TEMPLATE = (
    "The person you live with is called {name}. "
    "Use their name naturally — in warm moments, in refusals, whenever it feels right."
)


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
        self._default_system_prompt: str = self._resolve_prompt(config)

    def reconfigure(self, new_config: LumiConfig) -> None:
        """Apply hot-reloadable persona changes (e.g. system_prompt, user_name)."""
        self._config = new_config
        self._default_system_prompt = self._resolve_prompt(new_config)

    @staticmethod
    def _resolve_prompt(config: "LumiConfig | None") -> str:
        """Build the effective system prompt from config, injecting user_name if set."""
        base = (
            config.persona.system_prompt
            if config is not None and config.persona.system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT
        )
        if config is not None and config.persona.user_name:
            name_line = _NAME_LINE_TEMPLATE.format(name=config.persona.user_name)
            return f"{name_line}\n\n{base}"
        return base

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
        sys = (
            system_prompt if system_prompt is not None else self._default_system_prompt
        )
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
