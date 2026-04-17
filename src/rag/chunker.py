"""
Text chunker for the Lumi RAG ingest pipeline.

Splits plain text into overlapping fixed-size chunks while respecting sentence
boundaries.  No ML model is required — splitting is done with regex only, so
chunking is fast, deterministic, and safe to call from any thread.

Token counting uses a word-based approximation (whitespace-split word count).
This is intentional: introducing a real tokeniser at ingest time would add a
model dependency and 50–200 ms per document with negligible accuracy benefit
for retrieval purposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Sentence boundary: end of .  !  ?  (optionally followed by closing
# punctuation) then whitespace or end-of-string.
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')]?\s+|(?<=[.!?])["\')]?$')


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a source document."""

    text: str
    char_start: int   # byte offset of first character in the source text
    char_end: int     # byte offset past the last character (exclusive)
    chunk_idx: int    # 0-based position within the document


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a lightweight regex heuristic."""
    parts = _SENTENCE_END.split(text)
    # Re-attach trailing whitespace that the split consumed.
    sentences: list[str] = []
    pos = 0
    for part in parts:
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)
        pos += len(part)
    return sentences


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, size: int, overlap: int) -> list[Chunk]:
    """Split *text* into overlapping chunks of at most *size* words.

    Args:
        text:    Source document text.
        size:    Maximum chunk size in words (approximates tokens 1:1).
        overlap: Number of words carried over from the previous chunk to
                 preserve context at boundaries.  Must be < *size*.

    Returns:
        A list of :class:`Chunk` objects in document order.  An empty
        document returns an empty list.  A document shorter than *size*
        words returns a single chunk.
    """
    if not text.strip():
        return []
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    # Sliding window over sentences.
    window: list[str] = []
    window_words: int = 0
    chunk_idx: int = 0

    # We track character positions by searching the source text sequentially.
    search_start: int = 0

    def _emit(sents: list[str], idx: int) -> Chunk:
        nonlocal search_start
        joined = " ".join(sents)
        # Find the start of the first sentence in the remaining source text.
        first = sents[0]
        start = text.find(first, search_start)
        if start == -1:
            start = search_start
        end = start + len(joined)
        # Advance search cursor past the start of this chunk so the next
        # chunk's first sentence is found after the current one.
        search_start = start + len(first)
        return Chunk(text=joined, char_start=start, char_end=min(end, len(text)), chunk_idx=idx)

    for sentence in sentences:
        sent_words = _word_count(sentence)

        # If this single sentence already exceeds the size budget, emit it
        # as its own chunk rather than dropping it.
        if sent_words >= size:
            if window:
                chunks.append(_emit(window, chunk_idx))
                chunk_idx += 1
                window, window_words = [], 0
            chunks.append(_emit([sentence], chunk_idx))
            chunk_idx += 1
            continue

        if window_words + sent_words > size and window:
            chunks.append(_emit(window, chunk_idx))
            chunk_idx += 1

            # Seed the next window with the overlap tail of the current window.
            overlap_sents: list[str] = []
            overlap_words = 0
            for s in reversed(window):
                w = _word_count(s)
                if overlap_words + w > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_words += w
            window = overlap_sents
            window_words = overlap_words

        window.append(sentence)
        window_words += sent_words

    if window:
        chunks.append(_emit(window, chunk_idx))

    return chunks
