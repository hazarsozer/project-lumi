"""Tests for src/audio/viseme_map.py — phoneme to viseme mapping."""

from __future__ import annotations

import pytest

from src.audio.viseme_map import PHONEME_TO_VISEME, map_phoneme


@pytest.mark.unit
def test_known_phoneme_maps_correctly() -> None:
    """Known ARPAbet phonemes must map to the expected viseme group."""
    assert map_phoneme("AH") == "open"
    assert map_phoneme("S") == "teeth"
    assert map_phoneme("P") == "lips"
    assert map_phoneme("T") == "tongue"
    assert map_phoneme("OW") == "round"


@pytest.mark.unit
def test_stress_digit_stripped() -> None:
    """Trailing stress digits (0, 1, 2) must be stripped before lookup."""
    assert map_phoneme("AH1") == "open"
    assert map_phoneme("IY2") == "narrow"
    assert map_phoneme("UW0") == "round"


@pytest.mark.unit
def test_unknown_phoneme_returns_rest() -> None:
    """Unrecognised phonemes must return 'rest'."""
    assert map_phoneme("XYZ") == "rest"
    assert map_phoneme("") == "rest"
    assert map_phoneme("QQ") == "rest"


@pytest.mark.unit
def test_all_viseme_groups_reachable() -> None:
    """All 8 viseme group names must appear at least once in PHONEME_TO_VISEME values."""
    groups = set(PHONEME_TO_VISEME.values())
    expected = {"open", "narrow", "round", "wide", "teeth", "tongue", "lips"}
    # "rest" is the fallback, not mapped directly — all other 7 must appear.
    # (The spec lists 8 groups including "rest".)
    assert expected.issubset(groups)


@pytest.mark.unit
def test_case_insensitive() -> None:
    """map_phoneme must handle lowercase input by uppercasing before lookup."""
    assert map_phoneme("ah") == "open"
    assert map_phoneme("sh") == "teeth"
    assert map_phoneme("p") == "lips"
