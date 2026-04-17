"""Phoneme -> viseme mapping for Project Lumi lip-sync.

Maps ARPAbet and IPA phoneme strings (as returned by kokoro-onnx) to one of
8 viseme group names used by the Godot avatar controller.

Viseme groups:
  rest    -- neutral / silence / unrecognised
  open    -- AH, AA, AE  (open mouth vowels)
  narrow  -- IH, IY, EH, EY  (front narrow vowels)
  round   -- OW, UH, UW, AO  (rounded vowels)
  wide    -- AW, AY, OY  (diphthongs)
  teeth   -- S, Z, SH, ZH, CH, JH  (sibilants)
  tongue  -- T, D, N, L, R, ER  (tongue-tip consonants)
  lips    -- P, B, M, F, V, W  (bilabials + labiodentals)
"""

from __future__ import annotations

PHONEME_TO_VISEME: dict[str, str] = {
    # open
    "AH": "open", "AA": "open", "AE": "open",
    # narrow
    "IH": "narrow", "IY": "narrow", "EH": "narrow", "EY": "narrow",
    # round
    "OW": "round", "UH": "round", "UW": "round", "AO": "round",
    # wide (diphthongs)
    "AW": "wide", "AY": "wide", "OY": "wide",
    # teeth (sibilants)
    "S": "teeth", "Z": "teeth", "SH": "teeth", "ZH": "teeth",
    "CH": "teeth", "JH": "teeth",
    # tongue
    "T": "tongue", "D": "tongue", "N": "tongue", "L": "tongue",
    "R": "tongue", "ER": "tongue",
    # lips
    "P": "lips", "B": "lips", "M": "lips", "F": "lips", "V": "lips", "W": "lips",
}

_VALID_VISEMES: frozenset[str] = frozenset(
    ["rest", "open", "narrow", "round", "wide", "teeth", "tongue", "lips"]
)


def map_phoneme(phoneme: str) -> str:
    """Map a phoneme string to a viseme group name.

    Strips numeric stress markers (e.g. "AH1" -> "AH") before lookup.
    Returns "rest" for any unrecognised phoneme.

    Args:
        phoneme: ARPAbet or IPA phoneme string, possibly with stress digit suffix.

    Returns:
        One of the 8 viseme group names, or "rest".
    """
    stripped = phoneme.rstrip("012")
    return PHONEME_TO_VISEME.get(stripped.upper(), "rest")
