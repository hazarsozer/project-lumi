"""
src/rag — personal knowledge-base retriever for Project Lumi (Phase 7).

Public surface area (grows as waves are implemented):
    RAGConfig  — frozen dataclass (re-exported from src.core.config)
"""

from src.core.config import RAGConfig
from src.rag.errors import IngestError, RAGUnavailableError, RetrievalError
from src.rag.store import (
    ChunkRecord,
    DocumentRecord,
    DocumentStore,
    SearchHit,
    StoreStats,
)

__all__ = [
    "RAGConfig",
    "RAGUnavailableError",
    "IngestError",
    "RetrievalError",
    "DocumentStore",
    "DocumentRecord",
    "ChunkRecord",
    "SearchHit",
    "StoreStats",
]
