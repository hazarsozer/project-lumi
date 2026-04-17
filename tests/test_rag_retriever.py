"""Tests for src/rag/retriever.py — RAGRetriever."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.rag.retriever import Citation, RAGResult, RAGRetriever
from src.rag.store import SearchHit


def _make_config(
    *,
    retrieval_top_k: int = 5,
    context_char_budget: int = 2000,
    min_score: float = 0.15,
    retrieval_timeout_s: float = 2.0,
    embedding_model: str = "all-MiniLM-L6-v2",
):
    cfg = MagicMock()
    cfg.retrieval_top_k = retrieval_top_k
    cfg.context_char_budget = context_char_budget
    cfg.min_score = min_score
    cfg.retrieval_timeout_s = retrieval_timeout_s
    cfg.embedding_model = embedding_model
    return cfg


def _make_store(*, bm25_hits=None, vec_hits=None):
    store = MagicMock()
    store.search_fts.return_value = bm25_hits or []
    store.search_vectors.return_value = vec_hits or []
    store.get_chunk_by_id.return_value = None
    return store


def _make_hit(chunk_id: int, text: str = "chunk text", score: float = 0.9) -> SearchHit:
    h = MagicMock(spec=SearchHit)
    h.chunk_id = chunk_id
    h.text = text
    h.doc_path = f"/doc/{chunk_id}.md"
    h.chunk_idx = chunk_id
    h.score = score
    return h


class TestRAGRetriever:
    def _make_retriever(self, store=None, config=None):
        store = store or _make_store()
        config = config or _make_config()
        with patch("src.rag.retriever.get_embedder") as mock_get:
            embedder = MagicMock()
            embedder.encode.return_value = [[0.1] * 384]
            mock_get.return_value = embedder
            retriever = RAGRetriever(store, config)
            retriever._embedder = embedder
        return retriever

    def test_cancel_before_embed_returns_empty(self):
        retriever = self._make_retriever()
        flag = threading.Event()
        flag.set()
        result = retriever.retrieve("query", flag)
        assert result.context == ""
        assert result.hit_count == 0

    def test_no_hits_returns_empty(self):
        store = _make_store(bm25_hits=[], vec_hits=[])
        retriever = self._make_retriever(store=store)
        result = retriever.retrieve("query", threading.Event())
        assert result.context == ""

    def test_low_score_returns_empty(self):
        hit = _make_hit(1, "some text", score=0.9)
        store = _make_store(bm25_hits=[hit], vec_hits=[])
        config = _make_config(min_score=0.99)  # threshold above RRF score
        retriever = self._make_retriever(store=store, config=config)
        result = retriever.retrieve("query", threading.Event())
        assert result.context == ""

    def test_returns_context_on_hits(self):
        hit = _make_hit(1, "relevant passage", score=0.9)
        store = _make_store(bm25_hits=[hit], vec_hits=[hit])
        config = _make_config(min_score=0.0)
        retriever = self._make_retriever(store=store, config=config)
        result = retriever.retrieve("query", threading.Event())
        assert "relevant passage" in result.context
        assert result.hit_count > 0

    def test_context_trimmed_to_budget(self):
        long_text = "x" * 100
        hits = [_make_hit(i, long_text) for i in range(1, 5)]
        store = _make_store(bm25_hits=hits, vec_hits=hits)
        config = _make_config(context_char_budget=150, min_score=0.0)
        retriever = self._make_retriever(store=store, config=config)
        result = retriever.retrieve("query", threading.Event())
        assert len(result.context) <= 200  # budget + separator overhead

    def test_timeout_returns_empty(self):
        store = _make_store()
        config = _make_config(retrieval_timeout_s=0.001)
        with patch("src.rag.retriever.get_embedder") as mock_get:
            embedder = MagicMock()

            def slow_encode(_):
                import time
                time.sleep(0.5)
                return [[0.1] * 384]

            embedder.encode.side_effect = slow_encode
            mock_get.return_value = embedder
            retriever = RAGRetriever(store, config)
            retriever._embedder = embedder

        result = retriever.retrieve("query", threading.Event())
        assert result.context == ""
        assert result.latency_ms >= 0

    def test_result_latency_ms_populated(self):
        hit = _make_hit(1, "text")
        store = _make_store(bm25_hits=[hit], vec_hits=[])
        config = _make_config(min_score=0.0)
        retriever = self._make_retriever(store=store, config=config)
        result = retriever.retrieve("query", threading.Event())
        assert result.latency_ms >= 0

    def test_citations_populated(self):
        hit = _make_hit(1, "cited text")
        store = _make_store(bm25_hits=[hit], vec_hits=[hit])
        config = _make_config(min_score=0.0)
        retriever = self._make_retriever(store=store, config=config)
        result = retriever.retrieve("query", threading.Event())
        assert len(result.citations) > 0
        citation = result.citations[0]
        assert isinstance(citation, Citation)
        assert citation.chunk_id == 1


class TestRAGResultDataclass:
    def test_frozen(self):
        r = RAGResult(context="ctx", citations=(), latency_ms=10, hit_count=2)
        with pytest.raises((TypeError, AttributeError)):
            r.context = "new"  # type: ignore[misc]

    def test_empty_sentinel_context(self):
        r = RAGResult(context="", citations=(), latency_ms=0, hit_count=0)
        assert r.context == ""


class TestCitationDataclass:
    def test_frozen(self):
        c = Citation(chunk_id=1, doc_path="/x.md", chunk_idx=0, score=0.9)
        with pytest.raises((TypeError, AttributeError)):
            c.score = 0.1  # type: ignore[misc]
