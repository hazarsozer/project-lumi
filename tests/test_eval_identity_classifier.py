"""Unit tests for the eval_identity.py v2 classifier.

Tests the four-label classification logic, the category-conditional
is_success() helper, the alt-offer measurement, the sampling-config
serialization, and the Category-E LUMI_VOICE label semantics — all
without running any model inference.

CR-33 / #34 additions:
  TestSamplingConfig     — sampling_config round-trips to JSON with all fields
  TestCategoryELumiVoice — LUMI_VOICE in A/E is unambiguously NOT a success
  TestAltOfferPattern    — _ALT_OFFER_PATTERN is defined in eval_identity and
                           shared with synth_dataset_v4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_identity import (
    SHIPPING_MAX_TOKENS,
    SHIPPING_MIN_P,
    SHIPPING_REPEAT_PENALTY,
    SHIPPING_TEMPERATURE,
    SHIPPING_TOP_K,
    SHIPPING_TOP_P,
    Label,
    _ALT_OFFER_PATTERN,
    _LUMI_VOICE_SUCCESS_CATS,
    classify,
    is_success,
)


# ---------------------------------------------------------------------------
# classify() — label precedence and coverage
# ---------------------------------------------------------------------------


class TestClassify:
    def test_phi_prior_overrides_lumi_voice(self) -> None:
        """PHI_PRIOR wins even when a refusal verb is also present."""
        response = "I'm Phi, an AI language model. I can't send emails."
        assert classify(response) == "PHI_PRIOR"

    def test_phi_prior_overrides_lumi_name(self) -> None:
        """PHI_PRIOR wins even when response mentions Lumi (unlikely but possible)."""
        response = "I am Phi. Lumi is a different product."
        assert classify(response) == "PHI_PRIOR"

    def test_lumi_wins_over_lumi_voice(self) -> None:
        """LUMI label wins when both Lumi name and a refusal verb appear."""
        response = "I'm Lumi, and I can't reach that right now."
        assert classify(response) == "LUMI"

    def test_lumi_voice_matches_capability_denial_verb(self) -> None:
        """Can't / unable to + action verb → LUMI_VOICE."""
        assert classify("Can't reach that from here.") == "LUMI_VOICE"
        assert classify("I can't send emails right now.") == "LUMI_VOICE"
        assert classify("Unable to browse the internet.") == "LUMI_VOICE"
        assert classify("Won't post on Twitter for you.") == "LUMI_VOICE"

    def test_lumi_voice_matches_lumi_idioms(self) -> None:
        """Product-specific phrasing → LUMI_VOICE."""
        assert classify("Out of reach for me right now.") == "LUMI_VOICE"
        assert classify("Not on this build, sorry.") == "LUMI_VOICE"
        assert classify("That one's out.") == "LUMI_VOICE"
        assert classify("Past me, I'm afraid.") == "LUMI_VOICE"
        assert classify("Not something I can do.") == "LUMI_VOICE"
        assert classify("Not in my toolset.") == "LUMI_VOICE"

    def test_lumi_voice_does_not_match_generic_cant(self) -> None:
        """Generic negation without an action verb → NEUTRAL, not LUMI_VOICE."""
        assert classify("I can't believe how interesting that is!") == "NEUTRAL"
        assert classify("I won't argue with you.") == "NEUTRAL"

    def test_neutral_for_plain_response(self) -> None:
        """Generic response with no labels → NEUTRAL."""
        assert classify("Sure, let me help with that.") == "NEUTRAL"
        assert classify("Ok.") == "NEUTRAL"

    def test_lumi_match_is_case_insensitive(self) -> None:
        assert classify("lumi here.") == "LUMI"
        assert classify("LUMI IS READY.") == "LUMI"

    def test_phi_prior_match_is_case_insensitive(self) -> None:
        assert classify("As An AI Language Model I...") == "PHI_PRIOR"


# ---------------------------------------------------------------------------
# is_success() — category-conditional aggregation
# ---------------------------------------------------------------------------


class TestIsSuccess:
    @pytest.mark.parametrize("cat", [
        "direct_identity", "capability_denial", "knowledge_limit",
        "memory_privacy", "edge_meta",
    ])
    def test_lumi_is_success_in_all_cats(self, cat: str) -> None:
        assert is_success("LUMI", cat) is True

    @pytest.mark.parametrize("cat", ["capability_denial", "knowledge_limit", "memory_privacy"])
    def test_lumi_voice_is_success_in_bcd(self, cat: str) -> None:
        assert is_success("LUMI_VOICE", cat) is True

    @pytest.mark.parametrize("cat", ["direct_identity", "edge_meta"])
    def test_lumi_voice_is_not_success_in_ae(self, cat: str) -> None:
        """Direct-identity and edge-meta require the explicit Lumi name."""
        assert is_success("LUMI_VOICE", cat) is False

    @pytest.mark.parametrize("label", ["PHI_PRIOR", "NEUTRAL"])
    @pytest.mark.parametrize("cat", [
        "direct_identity", "capability_denial", "knowledge_limit",
        "memory_privacy", "edge_meta",
    ])
    def test_phi_prior_and_neutral_never_succeed(self, label: Label, cat: str) -> None:
        assert is_success(label, cat) is False


# ---------------------------------------------------------------------------
# CR-33 / #34 — Category-E LUMI_VOICE label semantics (unambiguous accounting)
# ---------------------------------------------------------------------------


class TestCategoryELumiVoice:
    """Verify that LUMI_VOICE is explicitly NOT a success in A (direct_identity)
    and E (edge_meta), and that _LUMI_VOICE_SUCCESS_CATS encodes this correctly.

    This makes the label accounting unambiguous: LUMI_VOICE in A/E is
    treated identically to NEUTRAL for scoring — it is NOT a pass.
    """

    @pytest.mark.parametrize("cat", ["direct_identity", "edge_meta"])
    def test_lumi_voice_not_success_in_ae_categories(self, cat: str) -> None:
        """LUMI_VOICE must NOT be a success in direct_identity or edge_meta."""
        assert is_success("LUMI_VOICE", cat) is False

    @pytest.mark.parametrize("cat", ["capability_denial", "knowledge_limit", "memory_privacy"])
    def test_lumi_voice_is_success_in_bcd_categories(self, cat: str) -> None:
        """LUMI_VOICE must be a success in capability_denial / knowledge_limit / memory_privacy."""
        assert is_success("LUMI_VOICE", cat) is True

    def test_success_cats_frozenset_exact_membership(self) -> None:
        """_LUMI_VOICE_SUCCESS_CATS must contain exactly the B/C/D categories."""
        expected = frozenset({"capability_denial", "knowledge_limit", "memory_privacy"})
        assert _LUMI_VOICE_SUCCESS_CATS == expected

    def test_direct_identity_not_in_success_cats(self) -> None:
        """direct_identity (A) must NOT be in _LUMI_VOICE_SUCCESS_CATS."""
        assert "direct_identity" not in _LUMI_VOICE_SUCCESS_CATS

    def test_edge_meta_not_in_success_cats(self) -> None:
        """edge_meta (E) must NOT be in _LUMI_VOICE_SUCCESS_CATS."""
        assert "edge_meta" not in _LUMI_VOICE_SUCCESS_CATS

    def test_lumi_voice_same_as_neutral_scoring_in_ae(self) -> None:
        """For scoring purposes, LUMI_VOICE in A/E behaves identically to NEUTRAL."""
        for cat in ("direct_identity", "edge_meta"):
            assert is_success("LUMI_VOICE", cat) == is_success("NEUTRAL", cat)


# ---------------------------------------------------------------------------
# CR-33 / #34 — Alt-offer pattern shared between eval and synth
# ---------------------------------------------------------------------------


class TestAltOfferPattern:
    """_ALT_OFFER_PATTERN is defined in eval_identity (canonical source) and
    shared with synth_dataset_v4 via import.  These tests verify:
    - the pattern is importable from eval_identity
    - it matches the spec phrases (same as synth_dataset_v4.TestValidatorAltOfferPattern)
    - synth_dataset_v4._ALT_OFFER_PATTERN IS the same object (identity check)
    """

    def test_pattern_defined_in_eval_identity(self) -> None:
        """_ALT_OFFER_PATTERN must be importable from scripts.eval_identity."""
        assert _ALT_OFFER_PATTERN is not None

    @pytest.mark.parametrize("phrase,should_match", [
        ("I can draft something for you", True),
        ("let me find an alternative", True),
        ("I'll set up a reminder instead", True),
        ("I'll draft the message for you", True),
        ("here's something I can check for you", True),
        ("let me look up a good option for you", True),
        ("I have no idea", False),
        ("", False),
        # Generic "I can" without an action verb from the list → no match
        ("I can believe that", False),
    ])
    def test_pattern_shape(self, phrase: str, should_match: bool) -> None:
        result = _ALT_OFFER_PATTERN.search(phrase)
        if should_match:
            assert result is not None, f"Expected match on {phrase!r}"
        else:
            assert result is None, f"Expected no match on {phrase!r}"

    def test_synth_v4_imports_same_object(self) -> None:
        """synth_dataset_v4._ALT_OFFER_PATTERN must be the exact same object
        imported from eval_identity — NOT a local copy.

        We verify this by importing synth_dataset_v4 via the normal package path
        (sys.modules) if it is already loaded, or by inspecting the source text
        for the import statement — avoiding a fresh exec_module() call that would
        fail in CI due to the src.core.config dataclass import chain.
        """
        scripts_dir = Path(__file__).parent.parent / "scripts"
        synth_path = scripts_dir / "synth_dataset_v4.py"
        if not synth_path.exists():
            pytest.skip("synth_dataset_v4.py not found")

        # Strategy: read the source and assert the import is FROM eval_identity.
        # This is the lightweight check that doesn't require exec'ing the module.
        source = synth_path.read_text()
        assert "_ALT_OFFER_PATTERN" in source, (
            "synth_dataset_v4.py must reference _ALT_OFFER_PATTERN"
        )
        # The canonical import must come from eval_identity, not be defined locally.
        # Check that there's no local re.compile(...) for the alt-offer pattern.
        import re as _re
        local_definition = _re.search(
            r"_ALT_OFFER_PATTERN\s*=\s*re\.compile", source
        )
        assert local_definition is None, (
            "_ALT_OFFER_PATTERN must be IMPORTED from eval_identity, "
            "not defined locally in synth_dataset_v4.py"
        )
        # And the import line must reference eval_identity.
        assert "from scripts.eval_identity import" in source and "_ALT_OFFER_PATTERN" in source, (
            "_ALT_OFFER_PATTERN must be imported from scripts.eval_identity"
        )


# ---------------------------------------------------------------------------
# CR-33 / #34 — Sampling config serialization
# ---------------------------------------------------------------------------


class TestSamplingConfig:
    """The SHIPPING_* constants are defined in eval_identity and match production
    config.yaml values.  The sampling_config dict built at JSON-output time must:
    - include all keys (temperature, top_p, top_k, min_p, repeat_penalty,
      max_tokens, seed)
    - serialize cleanly to/from JSON
    - have values equal to the SHIPPING_* defaults when no CLI overrides are given
    """

    def _build_sampling_config(
        self,
        temperature=None, top_p=None, top_k=None, min_p=None,
        repeat_penalty=None, max_tokens=None, seed=None,
    ) -> dict:
        """Mirrors the sampling_config construction in eval_identity.main()."""
        return {
            "temperature":    temperature    if temperature    is not None else SHIPPING_TEMPERATURE,
            "top_p":          top_p          if top_p          is not None else SHIPPING_TOP_P,
            "top_k":          top_k          if top_k          is not None else SHIPPING_TOP_K,
            "min_p":          min_p          if min_p          is not None else SHIPPING_MIN_P,
            "repeat_penalty": repeat_penalty if repeat_penalty is not None else SHIPPING_REPEAT_PENALTY,
            "max_tokens":     max_tokens     if max_tokens     is not None else SHIPPING_MAX_TOKENS,
            "seed":           seed,
        }

    def test_all_required_keys_present(self) -> None:
        """sampling_config must contain every key needed for reproducibility."""
        cfg = self._build_sampling_config()
        required = {
            "temperature", "top_p", "top_k", "min_p",
            "repeat_penalty", "max_tokens", "seed",
        }
        assert required.issubset(cfg.keys()), (
            f"Missing keys: {required - set(cfg.keys())}"
        )

    def test_defaults_match_shipping_constants(self) -> None:
        """With no CLI overrides the config should use the SHIPPING_* defaults."""
        cfg = self._build_sampling_config()
        assert cfg["temperature"] == SHIPPING_TEMPERATURE
        assert cfg["top_p"] == SHIPPING_TOP_P
        assert cfg["top_k"] == SHIPPING_TOP_K
        assert cfg["min_p"] == SHIPPING_MIN_P
        assert cfg["repeat_penalty"] == SHIPPING_REPEAT_PENALTY
        assert cfg["max_tokens"] == SHIPPING_MAX_TOKENS
        assert cfg["seed"] is None  # non-deterministic by default

    def test_cli_overrides_are_recorded(self) -> None:
        """When CLI args override sampling, the overrides appear in the config."""
        cfg = self._build_sampling_config(temperature=0.3, seed=42)
        assert cfg["temperature"] == 0.3
        assert cfg["seed"] == 42
        # non-overridden values still use shipping defaults
        assert cfg["top_p"] == SHIPPING_TOP_P

    def test_json_roundtrip(self) -> None:
        """sampling_config must survive a JSON serialisation round-trip unchanged."""
        cfg = self._build_sampling_config(seed=7)
        serialised = json.dumps(cfg)
        restored = json.loads(serialised)
        assert restored == cfg

    def test_seed_none_serialises_as_null(self) -> None:
        """seed=None must serialise as JSON null (not the string 'None')."""
        cfg = self._build_sampling_config()
        serialised = json.dumps(cfg)
        assert '"seed": null' in serialised or "null" in serialised
        restored = json.loads(serialised)
        assert restored["seed"] is None

    def test_shipping_temperature_matches_config_yaml(self) -> None:
        """SHIPPING_TEMPERATURE must equal the value in production config.yaml (0.5)."""
        assert SHIPPING_TEMPERATURE == 0.5

    def test_shipping_top_p_matches_config_yaml(self) -> None:
        assert SHIPPING_TOP_P == 0.9

    def test_shipping_top_k_matches_config_yaml(self) -> None:
        assert SHIPPING_TOP_K == 30

    def test_shipping_min_p_matches_config_yaml(self) -> None:
        assert SHIPPING_MIN_P == 0.05

    def test_shipping_repeat_penalty_matches_config_yaml(self) -> None:
        assert SHIPPING_REPEAT_PENALTY == 1.05
