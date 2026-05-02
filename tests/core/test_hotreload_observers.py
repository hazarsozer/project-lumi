"""Tests for hot-reload observer wiring on ReasoningRouter and PromptEngine."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.core.config import LLMConfig, LumiConfig, PersonaConfig
from src.llm.prompt_engine import DEFAULT_SYSTEM_PROMPT, PromptEngine
from src.llm.reasoning_router import ReasoningRouter


@pytest.mark.unit
def test_reasoning_router_reconfigure_updates_llm_config() -> None:
    rr = ReasoningRouter(
        model_loader=MagicMock(),
        prompt_engine=MagicMock(),
        memory=MagicMock(),
        config=LumiConfig().llm,
    )
    new_config = LumiConfig(llm=LLMConfig(temperature=0.99))
    rr.reconfigure(new_config)
    assert rr._config.temperature == 0.99


@pytest.mark.unit
def test_prompt_engine_reconfigure_updates_system_prompt() -> None:
    engine = PromptEngine()
    assert engine._default_system_prompt == DEFAULT_SYSTEM_PROMPT
    new_config = LumiConfig(persona=PersonaConfig(system_prompt="Custom prompt"))
    engine.reconfigure(new_config)
    assert engine._default_system_prompt == "Custom prompt"


@pytest.mark.unit
def test_prompt_engine_reconfigure_resets_to_default_when_none() -> None:
    engine = PromptEngine()
    engine.reconfigure(LumiConfig(persona=PersonaConfig(system_prompt="custom")))
    engine.reconfigure(LumiConfig(persona=PersonaConfig(system_prompt=None)))
    assert engine._default_system_prompt == DEFAULT_SYSTEM_PROMPT


@pytest.mark.unit
def test_prompt_engine_reconfigure_propagates_to_next_build() -> None:
    engine = PromptEngine()
    engine.reconfigure(LumiConfig(persona=PersonaConfig(system_prompt="TestBot")))
    result = engine.build_prompt("hi", [], system_prompt=None)
    assert "TestBot" in result
