"""Tests for src/rag/fusion.py — reciprocal rank fusion."""

import pytest

from src.rag.fusion import reciprocal_rank_fusion


def test_empty_rankings_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_single_list_single_item():
    result = reciprocal_rank_fusion([[42]])
    assert len(result) == 1
    chunk_id, score = result[0]
    assert chunk_id == 42
    assert score == pytest.approx(1 / (60 + 1))


def test_single_list_preserves_order():
    result = reciprocal_rank_fusion([[1, 2, 3]])
    ids = [r[0] for r in result]
    scores = [r[1] for r in result]
    # rank 1 should have highest score
    assert ids[0] == 1
    assert scores[0] > scores[1] > scores[2]


def test_two_lists_overlap_boosts_score():
    # chunk_id=1 appears rank-1 in both lists
    result = reciprocal_rank_fusion([[1, 2], [1, 3]])
    by_id = dict(result)
    # 1 should score higher than 2 or 3 (appears once each)
    assert by_id[1] > by_id[2]
    assert by_id[1] > by_id[3]


def test_union_of_ids_included():
    result = reciprocal_rank_fusion([[10, 20], [30, 40]])
    ids = {r[0] for r in result}
    assert ids == {10, 20, 30, 40}


def test_sorted_descending():
    result = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])
    scores = [r[1] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_custom_k():
    r_k60 = reciprocal_rank_fusion([[1]], k=60)
    r_k1 = reciprocal_rank_fusion([[1]], k=1)
    # smaller k → larger score (1/(1+1) > 1/(60+1))
    assert r_k1[0][1] > r_k60[0][1]


def test_returns_list_of_tuples():
    result = reciprocal_rank_fusion([[5, 6]])
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
