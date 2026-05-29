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

Thread model (CR-30 / issue #31):
  A single-worker ``ThreadPoolExecutor`` is shared across all retrieve()
  calls on a given RAGRetriever instance.  This eliminates the previous
  thread-per-query pattern where each call spawned a new daemon thread
  that was orphaned on timeout and continued to hold the Embedder lock.

  Bounded executor guarantees:
    - At most ONE worker thread exists at a time — no thread accumulation.
    - On timeout the caller receives _EMPTY immediately; the executor's
      worker thread runs to completion (releasing the Embedder lock) and
      is then ready for the next submission.  Because work items are
      serialised through a single worker, the Embedder lock is never
      contested by more than one in-flight encode() call.
    - A per-query abort_flag is set on timeout so _retrieve_inner can
      short-circuit remaining work (post-embed checks) as soon as the
      embed call returns.
"""

from __future__ import annotations

import concurrent.futures
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

    context: str  # trimmed text block ready for prompt injection
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
        # Single-worker executor: bounds thread count to 1, preventing the
        # thread-per-query accumulation that occurred with the old pattern.
        # The executor is intentionally NOT shut down between queries —
        # it lives for the lifetime of the RAGRetriever instance.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rag-worker",
        )

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

        The work is submitted to a shared bounded executor (1 worker) and
        awaited with a hard timeout.  On timeout the caller receives
        ``_EMPTY`` immediately.  The executor's worker continues to
        completion, releasing the Embedder lock naturally, and is then ready
        for the next submission.  A per-query ``abort_flag`` is set on
        timeout so ``_retrieve_inner`` can short-circuit any post-embed work
        as soon as the slow encode() call returns.

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

        budget_chars = (
            max_chars if max_chars is not None else self._config.context_char_budget
        )
        timeout_s = self._config.retrieval_timeout_s

        # Per-query abort flag: set on timeout so _retrieve_inner bails out
        # of remaining work (post-embed checks) once encode() returns.
        abort_flag = threading.Event()

        t0 = time.perf_counter()
        future = self._executor.submit(
            self._retrieve_inner, query, cancel_flag, budget_chars, abort_flag
        )

        try:
            result = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            abort_flag.set()  # signal worker to exit early after encode() returns
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning(
                "RAG retrieval timed out after %.0f ms (budget %.0f ms) for query: %.60s",
                elapsed_ms,
                timeout_s * 1000,
                query,
            )
            return _EMPTY
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(str(exc)) from exc

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
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
        abort_flag: threading.Event | None = None,
    ) -> RAGResult:
        if cancel_flag.is_set():
            return _EMPTY

        # Embed query.  The Embedder._lock is held for the duration of this
        # call; it is released when encode() returns regardless of whether
        # the caller has already timed out.
        try:
            vectors = self._embedder.encode([query])
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        # After encode() the Embedder lock is free.  Check abort (timeout
        # signal from retrieve()) and cancel flags before doing further work.
        if abort_flag is not None and abort_flag.is_set():
            return _EMPTY
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
            logger.debug(
                "RAG: top fused score %.4f below threshold %.4f — skipping.",
                fused[0][1] if fused else 0.0,
                self._config.min_score,
            )
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
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    doc_path=hit.doc_path,
                    chunk_idx=hit.chunk_idx,
                    score=score,
                )
            )
            used_chars += chunk_len

        if not context_parts:
            return _EMPTY

        return RAGResult(
            context="\n\n".join(context_parts),
            citations=tuple(citations),
            latency_ms=0,  # filled in by the outer call
            hit_count=len(fused),
        )
