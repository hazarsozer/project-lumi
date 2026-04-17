"""Tests for PromptEngine RAG context injection."""

import pytest

from src.llm.prompt_engine import PromptEngine


@pytest.fixture()
def engine():
    return PromptEngine()


class TestBuildPromptRAGContext:
    def test_no_rag_context_omits_relevant_notes(self, engine):
        prompt = engine.build_prompt("hi", [])
        assert "[Relevant notes]" not in prompt

    def test_rag_context_injected_into_system_block(self, engine):
        prompt = engine.build_prompt("hi", [], rag_context="Some doc passage.")
        assert "[Relevant notes]" in prompt
        assert "Some doc passage." in prompt

    def test_rag_context_appears_before_user_turn(self, engine):
        prompt = engine.build_prompt("hello", [], rag_context="my note")
        sys_idx = prompt.index("[Relevant notes]")
        user_idx = prompt.index("<|user|>")
        assert sys_idx < user_idx

    def test_rag_context_in_system_block_before_history(self, engine):
        history = [{"role": "user", "content": "prior question"}]
        prompt = engine.build_prompt("follow-up", history, rag_context="ctx")
        sys_idx = prompt.index("[Relevant notes]")
        history_idx = prompt.index("prior question")
        assert sys_idx < history_idx

    def test_empty_rag_context_string_omits_block(self, engine):
        prompt = engine.build_prompt("hi", [], rag_context="")
        assert "[Relevant notes]" not in prompt

    def test_custom_system_prompt_with_rag_context(self, engine):
        prompt = engine.build_prompt(
            "hi", [], system_prompt="Custom sys.", rag_context="extra"
        )
        assert "Custom sys." in prompt
        assert "[Relevant notes]" in prompt
        assert "extra" in prompt

    def test_rag_context_not_confused_with_user_turn(self, engine):
        prompt = engine.build_prompt("my question", [], rag_context="retrieved doc")
        # The user turn should be the last <|user|> block
        last_user = prompt.rfind("<|user|>")
        assert "my question" in prompt[last_user:]
        # The RAG block should be in the system block, before the user turn
        assert prompt.index("[Relevant notes]") < last_user
