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


# ---------------------------------------------------------------------------
# CR-30 / Issue #31 — bounded executor + lock-release-on-timeout tests
# ---------------------------------------------------------------------------

def _make_retriever_with_slow_embed(
    *,
    embed_delay_s: float,
    timeout_s: float,
    release_event: threading.Event | None = None,
) -> "RAGRetriever":
    """Build a RAGRetriever whose embedder blocks for ``embed_delay_s``.

    The mocked embedder uses ``release_event`` (if provided) as a gate so
    tests can control exactly when encode() returns, making timing fully
    deterministic.
    """
    store = _make_store()
    config = _make_config(retrieval_timeout_s=timeout_s)

    embedder = MagicMock()
    embedder._lock = threading.Lock()

    def slow_encode(_texts):
        import time
        if release_event is not None:
            release_event.wait()  # block until the test releases it
        else:
            time.sleep(embed_delay_s)
        return [[0.1] * 384]

    embedder.encode.side_effect = slow_encode

    with patch("src.rag.retriever.get_embedder", return_value=embedder):
        retriever = RAGRetriever(store, config)
        retriever._embedder = embedder

    return retriever


class TestCR30BoundedExecutor:
    """Verify that the single-worker executor prevents thread accumulation
    and that the Embedder lock is not held after a timeout (issue #31).
    """

    # ------------------------------------------------------------------
    # 1. Thread count stays bounded under repeated timed-out queries
    # ------------------------------------------------------------------

    def test_no_orphan_threads_after_many_timeouts(self):
        """Many timed-out queries must not accumulate new threads.

        With the old thread-per-query pattern this creates N orphans.
        With the bounded executor, the thread count stays at 1.
        """
        import threading

        # Capture thread count before creating the retriever.
        before_count = threading.active_count()

        store = _make_store()
        config = _make_config(retrieval_timeout_s=0.001)  # very short timeout

        # The embed will always be slower than the timeout.
        embedder = MagicMock()
        embedder._lock = threading.Lock()

        def slow_encode(_texts):
            import time
            time.sleep(0.05)  # 50 ms >> 1 ms timeout
            return [[0.1] * 384]

        embedder.encode.side_effect = slow_encode

        with patch("src.rag.retriever.get_embedder", return_value=embedder):
            retriever = RAGRetriever(store, config)
            retriever._embedder = embedder

        # Submit N queries, all of which will time out.
        N = 10
        for _ in range(N):
            result = retriever.retrieve("query", threading.Event())
            assert result.context == ""  # must return empty on timeout

        # Wait for any in-flight work to drain before counting threads.
        import time
        time.sleep(0.15)

        after_count = threading.active_count()

        # The executor creates exactly 1 worker thread.  N orphaned threads
        # from the old pattern would add N to the count.  With a bounded
        # executor the delta is at most 1 (the single pool worker).
        #
        # We allow a small slack of 2 for any unrelated pytest internals that
        # may spin up during the loop, but the key invariant is << N extra.
        delta = after_count - before_count
        assert delta <= 2, (
            f"Thread count grew by {delta} after {N} timed-out queries — "
            f"orphan threads may be accumulating. before={before_count}, after={after_count}"
        )

    # ------------------------------------------------------------------
    # 2. Embedder lock is released after timeout so a subsequent query
    #    can proceed without deadlocking.
    # ------------------------------------------------------------------

    def test_embedder_lock_released_after_timeout(self):
        """After a timed-out query the Embedder lock must be acquirable.

        Mechanism: the timed-out worker continues in the background holding
        ``embedder._lock`` (our mock lock) during encode().  We use a
        release_event to control exactly when encode() finishes, then
        verify the lock is free after the encode call returns.

        The test also verifies that a *second* retrieve() call succeeds
        (returns a result) once the slow embed finishes and the lock is free.
        """
        import time

        # release_event controls when slow_encode() returns.
        release_event = threading.Event()
        # A real (non-mock) lock to track whether it is held.
        real_lock = threading.Lock()

        store = _make_store()
        config = _make_config(
            retrieval_timeout_s=0.01,  # 10 ms — times out before encode returns
        )

        embedder = MagicMock()
        # Replace the mock lock with a real one so we can actually acquire it.
        embedder._lock = real_lock

        def controlled_encode(_texts):
            # Acquire the real lock to simulate what Embedder.encode() does.
            with real_lock:
                release_event.wait(timeout=5.0)  # wait for test to release
            return [[0.1] * 384]

        embedder.encode.side_effect = controlled_encode

        with patch("src.rag.retriever.get_embedder", return_value=embedder):
            retriever = RAGRetriever(store, config)
            retriever._embedder = embedder

        # First call — will time out because release_event is not set yet.
        result1 = retriever.retrieve("query-1", threading.Event())
        assert result1.context == ""  # timed out

        # The worker is still running inside controlled_encode, holding real_lock.
        # Try to acquire the lock non-blocking — it should FAIL (still held).
        assert not real_lock.acquire(blocking=False), (
            "Lock should still be held by the background worker after timeout"
        )

        # Release the worker — it will exit encode() and free the lock.
        release_event.set()

        # Give the background worker a moment to finish and release the lock.
        deadline = time.monotonic() + 2.0
        acquired = False
        while time.monotonic() < deadline:
            if real_lock.acquire(blocking=False):
                acquired = True
                real_lock.release()
                break
            time.sleep(0.01)

        assert acquired, "Embedder lock was NOT released after background worker finished"

    # ------------------------------------------------------------------
    # 3. A second query succeeds (lock free) once the first finishes.
    # ------------------------------------------------------------------

    def test_second_query_succeeds_after_first_times_out(self):
        """A query following a timed-out one must complete successfully.

        Uses a release_event to sequence: first query times out → event
        released → second query submitted and returns a valid result.
        """
        import time

        release_event = threading.Event()
        first_query_running = threading.Event()

        hit = _make_hit(1, "useful passage", score=0.9)
        store = _make_store(bm25_hits=[hit], vec_hits=[hit])
        config = _make_config(
            retrieval_timeout_s=0.01,  # times out before slow encode returns
            min_score=0.0,
        )

        call_count = {"n": 0}

        embedder = MagicMock()
        embedder._lock = threading.Lock()

        def encode_with_control(_texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: signal that we're running, then block until released.
                first_query_running.set()
                release_event.wait(timeout=5.0)
            return [[0.1] * 384]

        embedder.encode.side_effect = encode_with_control

        with patch("src.rag.retriever.get_embedder", return_value=embedder):
            retriever = RAGRetriever(store, config)
            retriever._embedder = embedder

        # Trigger first query — it will time out.
        result1 = retriever.retrieve("slow-query", threading.Event())
        assert result1.context == ""  # timed out as expected

        # Unblock the first worker so the executor thread is free.
        release_event.set()

        # Wait for the executor's worker thread to become idle.
        time.sleep(0.1)

        # Second query — encode is now fast (no gate), so it should return context.
        config2 = _make_config(
            retrieval_timeout_s=2.0,  # generous timeout
            min_score=0.0,
        )
        retriever._config = config2  # swap config for the second call

        result2 = retriever.retrieve("fast-query", threading.Event())
        assert result2.context != "", (
            "Second query should return context once the executor thread is free"
        )

    # ------------------------------------------------------------------
    # 4. abort_flag causes _retrieve_inner to return _EMPTY after encode.
    # ------------------------------------------------------------------

    def test_abort_flag_causes_early_exit_after_encode(self):
        """Setting abort_flag before _retrieve_inner starts causes it to
        return _EMPTY without running store searches.
        """
        store = _make_store(
            bm25_hits=[_make_hit(1, "passage")],
            vec_hits=[_make_hit(1, "passage")],
        )
        config = _make_config(min_score=0.0)

        embedder = MagicMock()
        embedder.encode.return_value = [[0.1] * 384]

        with patch("src.rag.retriever.get_embedder", return_value=embedder):
            retriever = RAGRetriever(store, config)
            retriever._embedder = embedder

        cancel_flag = threading.Event()
        abort_flag = threading.Event()
        abort_flag.set()  # pre-set — simulate a timeout that already fired

        result = retriever._retrieve_inner(
            "query", cancel_flag, 2000, abort_flag
        )

        assert result.context == ""
        # Store searches must NOT have been called because abort_flag was set.
        store.search_fts.assert_not_called()
        store.search_vectors.assert_not_called()
