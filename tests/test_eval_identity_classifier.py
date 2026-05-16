"""Unit tests for the eval_identity.py v2 classifier.

Tests the four-label classification logic and the category-conditional
is_success() helper without running any model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_identity import Label, classify, is_success


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
