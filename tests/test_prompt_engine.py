"""
Tests for src.llm.prompt_engine.PromptEngine.

Mocking strategy
----------------
PromptEngine is pure Python string manipulation — no hardware, no model files,
no external dependencies are required.  All tests are self-contained.

All tests are marked ``unit``.
"""

from __future__ import annotations

import pytest

# RED: these imports will fail until src/llm/prompt_engine.py is written.
from src.llm.prompt_engine import DEFAULT_SYSTEM_PROMPT, PromptEngine  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(turns: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Build a history list from (role, content) tuples."""
    return [{"role": role, "content": content} for role, content in turns]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_prompt_no_history() -> None:
    """build_prompt with empty history must include the system prompt and user text."""
    engine = PromptEngine()
    result = engine.build_prompt(
        user_text="What is 2 + 2?",
        history=[],
        system_prompt="You are a helpful assistant.",
    )
    assert "What is 2 + 2?" in result
    assert "You are a helpful assistant." in result


@pytest.mark.unit
def test_build_prompt_with_history() -> None:
    """build_prompt with prior turns must include those turns in the output."""
    engine = PromptEngine()
    history = _make_history([
        ("user", "Hello"),
        ("assistant", "Hi there!"),
    ])
    result = engine.build_prompt(
        user_text="How are you?",
        history=history,
        system_prompt="You are a helpful assistant.",
    )
    assert "Hello" in result
    assert "Hi there!" in result
    assert "How are you?" in result


@pytest.mark.unit
def test_build_prompt_default_system_prompt() -> None:
    """When system_prompt=None, build_prompt must still produce a non-empty string."""
    engine = PromptEngine()
    result = engine.build_prompt(
        user_text="Tell me a joke.",
        history=[],
        system_prompt=None,
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Tell me a joke." in result


@pytest.mark.unit
def test_truncate_history_empty() -> None:
    """truncate_history on an empty list must return an empty list."""
    engine = PromptEngine()
    result = engine.truncate_history([], max_tokens=512)
    assert result == []


@pytest.mark.unit
def test_truncate_history_removes_oldest() -> None:
    """truncate_history must drop the oldest turns when the token budget is exceeded."""
    engine = PromptEngine()
    # Build a history that exceeds any reasonable small token budget.
    history = _make_history([
        ("user", "A" * 200),
        ("assistant", "B" * 200),
        ("user", "C" * 200),
        ("assistant", "D" * 200),
        ("user", "E" * 200),
    ])
    # max_tokens=50 should force removal of the oldest turns.
    result = engine.truncate_history(history, max_tokens=50)
    # Result must be shorter than the original.
    assert len(result) < len(history)
    # The most-recent turn must be preserved.
    assert result[-1]["content"] == "E" * 200


@pytest.mark.unit
def test_truncate_history_within_budget_unchanged() -> None:
    """truncate_history must not remove turns when history fits within the budget."""
    engine = PromptEngine()
    history = _make_history([
        ("user", "Hi"),
        ("assistant", "Hello!"),
    ])
    result = engine.truncate_history(history, max_tokens=4096)
    assert result == history


@pytest.mark.unit
def test_build_prompt_returns_string() -> None:
    """build_prompt must always return a plain str."""
    engine = PromptEngine()
    result = engine.build_prompt("test", [], None)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Wave F1 — Persona system prompt tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_persona_contains_name() -> None:
    """The default system prompt must mention the assistant name Lumi."""
    assert "Lumi" in DEFAULT_SYSTEM_PROMPT


@pytest.mark.unit
def test_persona_no_markdown_instruction() -> None:
    """The default system prompt must instruct the model to avoid markdown."""
    prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
    # Accept any phrasing that clearly prohibits markdown formatting.
    has_no_markdown = (
        "no markdown" in prompt_lower
        or "do not use markdown" in prompt_lower
        or "avoid markdown" in prompt_lower
        or "plain text" in prompt_lower
    )
    assert has_no_markdown, (
        "DEFAULT_SYSTEM_PROMPT must contain a plain-text / no-markdown instruction"
    )


@pytest.mark.unit
def test_persona_no_filler_instruction() -> None:
    """The default system prompt must prohibit filler openers."""
    prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
    # The prompt should explicitly name at least one banned filler word.
    banned_fillers_mentioned = any(
        filler in prompt_lower
        for filler in ("certainly", "of course", "sure", "absolutely")
    )
    assert banned_fillers_mentioned, (
        "DEFAULT_SYSTEM_PROMPT must enumerate banned filler openers"
    )


@pytest.mark.unit
def test_persona_tool_call_schema_present() -> None:
    """The default system prompt must describe the JSON tool-call schema."""
    prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
    # The prompt must reference both the tool-call format and the key names.
    has_tool_schema = (
        '"tool"' in DEFAULT_SYSTEM_PROMPT or "'tool'" in DEFAULT_SYSTEM_PROMPT
    ) and (
        '"args"' in DEFAULT_SYSTEM_PROMPT or "'args'" in DEFAULT_SYSTEM_PROMPT
    )
    assert has_tool_schema, (
        "DEFAULT_SYSTEM_PROMPT must document the JSON tool-call schema with 'tool' and 'args' keys"
    )


@pytest.mark.unit
def test_persona_overridable_from_config() -> None:
    """A custom persona.system_prompt in config must override the built-in default.

    This test constructs a minimal config dict and passes it through
    load_config to produce a LumiConfig, then verifies that PromptEngine
    honours the override when system_prompt=None is passed to build_prompt.
    """
    import tempfile
    import textwrap
    from pathlib import Path

    from src.core.config import load_config

    custom_prompt = "I am a custom persona for testing."
    yaml_text = textwrap.dedent(f"""\
        edition: light
        persona:
          system_prompt: "{custom_prompt}"
    """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(yaml_text)
        tmp_path = tmp.name

    try:
        cfg = load_config(tmp_path)
        engine = PromptEngine(config=cfg)
        result = engine.build_prompt("hello", [], system_prompt=None)
        assert custom_prompt in result, (
            f"Expected custom persona prompt in output, got: {result[:200]}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
