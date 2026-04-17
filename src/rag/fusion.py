"""Reciprocal Rank Fusion for combining BM25 and vector search rankings."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists of chunk IDs using Reciprocal Rank Fusion.

    RRF score for item i across all rankings:
        score(i) = sum(1 / (k + rank(i, list)))

    Args:
        rankings: Each inner list is a ranked sequence of chunk IDs,
                  best first (index 0 = most relevant).
        k:        Smoothing constant that reduces the impact of high ranks.
                  60 is the standard default from the original RRF paper.

    Returns:
        List of (chunk_id, fused_score) tuples, sorted descending by score.
        chunk_ids that appear in only one list are still included.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
