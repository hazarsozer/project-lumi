"""
src/rag — personal knowledge-base retriever for Project Lumi (Phase 7).

Public surface area (grows as waves are implemented):
    RAGConfig  — frozen dataclass (re-exported from src.core.config)
"""

from src.core.config import RAGConfig
from src.rag.chunker import Chunk, chunk_text
from src.rag.embedder import Embedder, get_embedder
from src.rag.errors import IngestError, RAGUnavailableError, RetrievalError
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.loader import LoadedDocument, is_supported, load
from src.rag.retriever import Citation, RAGResult, RAGRetriever
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
    "Chunk",
    "chunk_text",
    "Embedder",
    "get_embedder",
    "LoadedDocument",
    "is_supported",
    "load",
    "reciprocal_rank_fusion",
    "Citation",
    "RAGResult",
    "RAGRetriever",
]
