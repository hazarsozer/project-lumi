"""
src.llm — local LLM integration for Project Lumi.

Modules:
    model_loader.py      — VRAM hibernate/wake lifecycle (wraps llama_cpp.Llama)
    model_registry.py    — named GGUF hot-swap registry (composition over ModelLoader)
    prompt_engine.py     — ChatML prompt assembly and token-budget truncation
    memory.py            — JSON-persisted conversation history
    domain_router.py     — regex domain classifier (<1ms, 6 fine-tune categories)
    reflex_router.py     — regex-based fast-path for greetings and time queries
    reasoning_router.py  — token-by-token LLM inference with cancel support
    tool_call_parser.py  — extract <tool_call> blocks from raw LLM output
"""

from src.llm.domain_router import DomainRouter
from src.llm.memory import ConversationMemory
from src.llm.model_loader import ModelLoader
from src.llm.model_registry import ModelRegistry
from src.llm.prompt_engine import PromptEngine
from src.llm.reasoning_router import ReasoningRouter
from src.llm.reflex_router import ReflexRouter
from src.llm.tool_call_parser import parse_tool_calls

__all__ = [
    "ConversationMemory",
    "DomainRouter",
    "ModelLoader",
    "ModelRegistry",
    "PromptEngine",
    "ReasoningRouter",
    "ReflexRouter",
    "parse_tool_calls",
]
