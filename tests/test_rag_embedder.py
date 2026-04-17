"""Tests for the Embedder singleton — slow tests skipped in default CI."""

from __future__ import annotations

import threading

import pytest

from src.rag.embedder import Embedder, get_embedder

pytestmark = pytest.mark.slow  # skipped unless -m slow


class TestEmbedderSlow:
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def test_encode_returns_correct_shape(self):
        emb = Embedder(self.MODEL)
        texts = ["hello world", "foo bar baz"]
        vecs = emb.encode(texts)
        assert len(vecs) == 2
        assert len(vecs[0]) == 384
        assert len(vecs[1]) == 384

    def test_encode_empty_list(self):
        emb = Embedder(self.MODEL)
        assert emb.encode([]) == []

    def test_vectors_are_floats(self):
        emb = Embedder(self.MODEL)
        vecs = emb.encode(["test"])
        assert all(isinstance(v, float) for v in vecs[0])

    def test_similar_texts_have_high_cosine_similarity(self):
        import math
        emb = Embedder(self.MODEL)
        v1, v2 = emb.encode(["The cat sat on the mat.", "A cat was sitting on a mat."])
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a ** 2 for a in v1))
        mag2 = math.sqrt(sum(b ** 2 for b in v2))
        cosine = dot / (mag1 * mag2)
        assert cosine > 0.8

    def test_embedding_dim_property(self):
        emb = Embedder(self.MODEL)
        assert emb.embedding_dim == 384

    def test_get_embedder_returns_singleton(self):
        a = get_embedder(self.MODEL)
        b = get_embedder(self.MODEL)
        assert a is b

    def test_thread_safe_concurrent_encode(self):
        emb = Embedder(self.MODEL)
        results: list[list[list[float]]] = []
        errors: list[Exception] = []

        def _encode():
            try:
                results.append(emb.encode(["concurrent test"]))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_encode) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 4
