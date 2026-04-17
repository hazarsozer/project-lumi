"""Tests for the text chunker — boundaries, overlap, edge cases."""

from __future__ import annotations

import pytest

from src.rag.chunker import Chunk, chunk_text


class TestChunkTextBasic:
    def test_empty_string_returns_empty(self):
        assert chunk_text("", size=10, overlap=2) == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n   ", size=10, overlap=2) == []

    def test_short_text_single_chunk(self):
        text = "Hello world. This is a test."
        chunks = chunk_text(text, size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].chunk_idx == 0

    def test_single_chunk_contains_all_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunk_text(text, size=100, overlap=5)
        combined = " ".join(c.text for c in chunks)
        # Every word from the original should appear in the output.
        for word in text.split():
            word_clean = word.strip(".,!?")
            assert word_clean in combined

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("Some text here.", size=5, overlap=5)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("Some text.", size=3, overlap=10)


class TestChunkIndices:
    def test_chunk_idx_sequential(self):
        # Produce enough text to force multiple chunks.
        words = " ".join(["word"] * 200)
        text = ". ".join([words] * 3) + "."
        chunks = chunk_text(text, size=50, overlap=5)
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.chunk_idx == i

    def test_single_chunk_idx_zero(self):
        chunks = chunk_text("Short sentence.", size=50, overlap=5)
        assert chunks[0].chunk_idx == 0


class TestCharOffsets:
    def test_char_start_nonnegative(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, size=3, overlap=1)
        for c in chunks:
            assert c.char_start >= 0

    def test_char_end_within_bounds(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, size=3, overlap=1)
        for c in chunks:
            assert c.char_end <= len(text)

    def test_char_start_before_char_end(self):
        text = "Alpha. Beta. Gamma. Delta. Epsilon."
        chunks = chunk_text(text, size=2, overlap=1)
        for c in chunks:
            assert c.char_start < c.char_end


class TestOverlap:
    def _word_set(self, chunk: Chunk) -> set[str]:
        return set(chunk.text.lower().split())

    def test_overlap_shares_words_across_chunks(self):
        # 10 distinct sentences of 5 words each = 50 words total.
        sentences = [f"sentence {i} has five words." for i in range(10)]
        text = " ".join(sentences)
        chunks = chunk_text(text, size=15, overlap=5)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        # Adjacent chunks should share at least one word from the overlap tail.
        for prev, nxt in zip(chunks, chunks[1:]):
            shared = self._word_set(prev) & self._word_set(nxt)
            assert len(shared) > 0, f"No overlap between chunk {prev.chunk_idx} and {nxt.chunk_idx}"


class TestLargeInputs:
    def test_many_sentences(self):
        sentences = ["This is sentence number {}.".format(i) for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, size=20, overlap=5)
        assert len(chunks) > 1
        # No chunk should exceed size * 2 words (rough sanity check).
        for c in chunks:
            assert len(c.text.split()) <= 40

    def test_oversized_single_sentence_emitted(self):
        # A sentence longer than the chunk size should still be emitted.
        giant = " ".join(["word"] * 600)
        chunks = chunk_text(giant, size=50, overlap=5)
        assert len(chunks) >= 1
        assert any("word" in c.text for c in chunks)

    def test_frozen_dataclass(self):
        chunks = chunk_text("Hello world.", size=10, overlap=2)
        with pytest.raises((AttributeError, TypeError)):
            chunks[0].text = "mutated"  # type: ignore[misc]
