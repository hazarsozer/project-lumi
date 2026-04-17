"""
src/rag — personal knowledge-base retriever for Project Lumi (Phase 7).

Public surface area (grows as waves are implemented):
    RAGConfig  — frozen dataclass (re-exported from src.core.config)
"""

from src.core.config import RAGConfig

__all__ = ["RAGConfig"]
