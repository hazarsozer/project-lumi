"""
RAG retriever for Project Lumi — fuses BM25 and vector kNN results.

The retriever is designed to run inside the inference worker thread
(the same thread that calls ReasoningRouter.generate).  This means:

  - It must be synchronous (no asyncio).
  - It checks cancel_flag before embedding and after search so an
    interrupt aborts retrieval and lets the LLM respond without context.
  - A hard timeout (RAGConfig.retrieval_timeout_s) caps the total call
    so a slow embedding or overloaded DB cannot delay the response past
    the 2-second voice UI threshold.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from src.core.config import RAGConfig
from src.rag.embedder import get_embedder
from src.rag.errors import RetrievalError
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.store import DocumentStore, SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Citation:
    """One retrieved chunk cited in the RAG context block."""

    chunk_id: int
    doc_path: str
    chunk_idx: int
    score: float


@dataclass(frozen=True)
class RAGResult:
    """Return value of RAGRetriever.retrieve()."""

    context: str                   # trimmed text block ready for prompt injection
    citations: tuple[Citation, ...]
    latency_ms: int
    hit_count: int


_EMPTY = RAGResult(context="", citations=(), latency_ms=0, hit_count=0)


class RAGRetriever:
    """Hybrid BM25 + vector kNN retriever with RRF fusion.

    Args:
        store:  Initialised :class:`~src.rag.store.DocumentStore` instance.
        config: :class:`~src.core.config.RAGConfig` from the loaded config.
    """

    def __init__(self, store: DocumentStore, config: RAGConfig) -> None:
        self._store = store
        self._config = config
        self._embedder = get_embedder(config.embedding_model)

    def retrieve(
        self,
        query: str,
        cancel_flag: threading.Event,
        max_chars: int | None = None,
    ) -> RAGResult:
        """Retrieve relevant context for *query*.

        Steps:
          1. Check cancel_flag — abort immediately if set.
          2. Embed the query on CPU (~20 ms).
          3. Check cancel_flag again.
          4. Run BM25 (FTS5) + kNN (sqlite-vec) searches.
          5. Fuse rankings with RRF.
          6. Trim fused results to *max_chars* at chunk boundaries.
          7. Return :class:`RAGResult`.

        The entire call is wrapped in a timeout thread so that a slow
        embedding cannot stall the inference pipeline.

        Returns an empty :class:`RAGResult` (``context=""``) if:
          - cancel_flag is set,
          - the timeout fires,
          - the top hit score is below ``config.min_score``, or
          - the store has no chunks yet.

        Raises:
            :class:`~src.rag.errors.RetrievalError`: only for unexpected
            internal failures; cancel and timeout produce empty results, not
            exceptions, so the LLM can still respond.
        """
        import time

        budget_chars = max_chars if max_chars is not None else self._config.context_char_budget
        timeout_s = self._config.retrieval_timeout_s

        result_box: list[RAGResult] = []
        exc_box: list[Exception] = []

        def _work() -> None:
            try:
                result_box.append(self._retrieve_inner(query, cancel_flag, budget_chars))
            except Exception as exc:
                exc_box.append(exc)

        t0 = time.perf_counter()
        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        worker.join(timeout=timeout_s)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if worker.is_alive():
            logger.warning(
                "RAG retrieval timed out after %.0f ms (budget %.0f ms) for query: %.60s",
                elapsed_ms, timeout_s * 1000, query,
            )
            return _EMPTY

        if exc_box:
            raise RetrievalError(str(exc_box[0])) from exc_box[0]

        result = result_box[0]
        return RAGResult(
            context=result.context,
            citations=result.citations,
            latency_ms=elapsed_ms,
            hit_count=result.hit_count,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve_inner(
        self,
        query: str,
        cancel_flag: threading.Event,
        budget_chars: int,
    ) -> RAGResult:
        if cancel_flag.is_set():
            return _EMPTY

        # Embed query.
        try:
            vectors = self._embedder.encode([query])
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        if cancel_flag.is_set():
            return _EMPTY

        embedding = vectors[0]
        top_k = self._config.retrieval_top_k

        bm25_hits: list[SearchHit] = self._store.search_fts(query, top_k)
        vec_hits: list[SearchHit] = self._store.search_vectors(embedding, top_k)

        if not bm25_hits and not vec_hits:
            return _EMPTY

        # Build ranked ID lists for RRF.
        bm25_ids = [h.chunk_id for h in bm25_hits]
        vec_ids = [h.chunk_id for h in vec_hits]
        fused = reciprocal_rank_fusion([bm25_ids, vec_ids])

        # Score threshold: skip if top result is below the floor.
        if not fused or fused[0][1] < self._config.min_score:
            logger.debug("RAG: top fused score %.4f below threshold %.4f — skipping.",
                         fused[0][1] if fused else 0.0, self._config.min_score)
            return _EMPTY

        # Build a lookup from chunk_id → hit for text retrieval.
        hit_by_id: dict[int, SearchHit] = {h.chunk_id: h for h in bm25_hits + vec_hits}

        # Assemble context block, trimming at chunk boundaries to stay within budget.
        context_parts: list[str] = []
        citations: list[Citation] = []
        used_chars = 0

        for chunk_id, score in fused:
            hit = hit_by_id.get(chunk_id)
            if hit is None:
                # Hit came from only one list; fetch from store.
                record = self._store.get_chunk_by_id(chunk_id)
                if record is None:
                    continue
                hit = SearchHit(
                    chunk_id=chunk_id,
                    score=score,
                    text=record.text,
                    doc_path="",
                    chunk_idx=record.chunk_idx,
                )

            chunk_len = len(hit.text)
            if used_chars + chunk_len > budget_chars:
                break

            context_parts.append(hit.text)
            citations.append(Citation(
                chunk_id=chunk_id,
                doc_path=hit.doc_path,
                chunk_idx=hit.chunk_idx,
                score=score,
            ))
            used_chars += chunk_len

        if not context_parts:
            return _EMPTY

        return RAGResult(
            context="\n\n".join(context_parts),
            citations=tuple(citations),
            latency_ms=0,   # filled in by the outer call
            hit_count=len(fused),
        )
