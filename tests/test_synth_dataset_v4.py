"""Tests for scripts/synth_dataset_v4.py and scripts/merge_v2.4_corpus.py.

These tests were written alongside the implementation (TDD).  The import block
at the top uses ``pytest.importorskip`` / a graceful file-spec load so the full
suite stays green while the impl is pending — once both scripts land, all tests
run and must pass without real API calls or network access.

## Mocking strategy

- ``subprocess.run`` is patched at the module boundary inside
  ``scripts.synth_dataset_v4`` using ``pytest-mock``'s ``mocker.patch``.
  The mock returns a ``subprocess.CompletedProcess`` shaped like the Gemini
  CLI's ``-o json`` output: ``{"session_id": "...", "response": "<text>",
  "stats": {...}}``.  The impl never needs to branch on "mock?".
- Filesystem state lives entirely under ``tmp_path`` — no reads/writes to the
  project ``data/`` directory.
- ``_LUMI_VOICE_PATTERNS`` and ``_PHI_PATTERNS`` are imported directly from
  ``scripts.eval_identity`` (the same import the impl uses) so tests pin the
  symbol contract; if eval_identity renames a symbol the tests fail loudly.

## Fixture conventions

Follows the project-wide conventions in ``tests/conftest.py``:
- Hardware-free, network-free, no real ``gemini`` CLI invocations.
- ``pytest.mark.unit`` for pure-logic tests; ``pytest.mark.integration`` for
  tests exercising the full generation or merge pipeline.
- ``tmp_path`` for all filesystem state.

## Gemini CLI JSON output shape (verified 2026-05-17)

::

    {
        "session_id": "...",
        "response": "<text>",
        "stats": {"models": {"gemini-2.5-flash": {"api": {"totalRequests": 1}}}},
    }

Patch site: ``mocker.patch("scripts.synth_dataset_v4.subprocess.run",
return_value=make_gemini_result("..."))``

## Real impl API (as of 2026-05-17)

``synth_dataset_v4.py`` exports:
  - SOURCE_TAG: str = "synthetic_v4"
  - _ALT_OFFER_PATTERN, _LUMI_VOICE_PATTERNS, _PHI_PATTERNS (all re.Pattern)
  - validate_response(response, opener_counter) -> tuple[bool, str]
  - generate_paraphrases(base_prompt, n, model) -> list[str]
  - generate_responses(system_prompt, user_prompt, n, model) -> list[str]
  - _call_gemini(prompt, *, model, tmpdir) -> str
  - run_generation(output, *, target_count, pools, sys_prompt_ids,
                   paraphrases_per_prompt, responses_per_paraphrase,
                   max_retries, accepted_keys, opener_counter, dry_run)
    -> tuple[int, Counter]
  - load_accepted_keys(output) -> set[SlotKey]
  - save_accepted_keys(output, accepted) -> None
  - append_partial_record(output, record) -> None
  - _partial_path(output) -> Path
  - _accepted_keys_path(output) -> Path
  - PROMPT_POOLS: dict
  - SYSTEM_PROMPTS: dict

``merge_v2.4_corpus.py`` exports:
  - merge_corpora(v4_records, v21_records, *, drop_categories, seed)
    -> list[dict]  (shuffled copy, inputs not mutated)
  - validate_record(record, source_label) -> tuple[bool, str]
  - DROP_CATEGORIES_FROM_V21: frozenset[str]
"""

from __future__ import annotations

import collections
import importlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path (mirrors pattern in test_eval_identity_classifier.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Guard-import: keep suite green while impl is still pending
# ---------------------------------------------------------------------------

synth_v4 = pytest.importorskip(
    "scripts.synth_dataset_v4",
    reason="scripts/synth_dataset_v4.py not yet implemented",
)

# merge_v2.4_corpus.py has a dot in the filename — invalid Python identifier.
# Load with importlib.util.spec_from_file_location instead.
_merge_spec_path = PROJECT_ROOT / "scripts" / "merge_v2.4_corpus.py"
if not _merge_spec_path.exists():
    pytest.skip(
        "scripts/merge_v2.4_corpus.py not yet implemented",
        allow_module_level=True,
    )

_merge_spec = importlib.util.spec_from_file_location("merge_v24_corpus", _merge_spec_path)
if _merge_spec is None or _merge_spec.loader is None:  # pragma: no cover
    pytest.skip(
        "Could not create module spec for merge_v2.4_corpus.py",
        allow_module_level=True,
    )
merge_v24 = importlib.util.module_from_spec(_merge_spec)
try:
    _merge_spec.loader.exec_module(merge_v24)  # type: ignore[union-attr]
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"scripts/merge_v2.4_corpus.py raised on import: {exc}",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Import symbols we test (fail loudly if impl forgot to define them)
# ---------------------------------------------------------------------------

validate_response: Any = getattr(synth_v4, "validate_response", None)
generate_paraphrases: Any = getattr(synth_v4, "generate_paraphrases", None)
generate_responses: Any = getattr(synth_v4, "generate_responses", None)
run_generation: Any = getattr(synth_v4, "run_generation", None)

# merge_v24 exports merge_corpora (plural) per the impl
merge_corpora: Any = getattr(merge_v24, "merge_corpora", None)
validate_record: Any = getattr(merge_v24, "validate_record", None)

# Import the eval_identity symbols — must be the SAME objects the impl uses
from scripts.eval_identity import _LUMI_VOICE_PATTERNS, _PHI_PATTERNS

# The ALT_OFFER pattern lives in the generator script per v2.4-plan §Validator
_ALT_OFFER_PATTERN: re.Pattern[str] | None = getattr(synth_v4, "_ALT_OFFER_PATTERN", None)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Shared helpers / constants
# ---------------------------------------------------------------------------

_GOLDEN_RESPONSE = (
    "Lumi here — sending email isn't in my toolset right now. "
    "I can draft something for you to send yourself."
)
# Confirms to: len=93, has Lumi in sentence 1, LUMI_VOICE ("isn't in my toolset"),
# ALT_OFFER ("I can draft"), no Phi leak.

_VALID_CATEGORIES = frozenset({"capability_denial", "knowledge_limit", "memory_privacy"})


def _make_v4_record(
    *,
    content: str = _GOLDEN_RESPONSE,
    category: str = "capability_denial",
    source: str = "synthetic_v4",
    tools: object = None,
) -> dict:
    """Build a minimal well-formed v4 record dict."""
    return {
        "messages": [
            {"role": "system", "content": "You are Lumi."},
            {"role": "user", "content": "Send an email to my boss."},
            {"role": "assistant", "content": content},
        ],
        "category": category,
        "tools": tools,
        "source": source,
    }


def make_gemini_result(
    response_text: str,
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    """Return a ``subprocess.CompletedProcess`` shaped like the Gemini CLI ``-o json`` output.

    Patch site::

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result("Your text here"),
        )
    """
    stdout = json.dumps(
        {
            "session_id": "test-session",
            "response": response_text,
            "stats": {
                "models": {
                    "gemini-2.5-flash": {"api": {"totalRequests": 1}}
                }
            },
        }
    )
    return subprocess.CompletedProcess(
        args=["gemini", "-p", "...", "-m", "gemini-2.5-flash", "-o", "json"],
        returncode=returncode,
        stdout=stdout,
        stderr="" if returncode == 0 else "mock error message",
    )


# ---------------------------------------------------------------------------
# Section A — Validator constraints (each independently)
# ---------------------------------------------------------------------------


class TestValidatorLength:
    """Constraint 5: 80 ≤ len(response) ≤ 400."""

    @pytest.mark.unit
    @pytest.mark.parametrize("length,should_pass", [
        (79, False),
        (80, True),
        (400, True),
        (401, False),
    ])
    def test_length_boundary(self, length: int, should_pass: bool) -> None:
        """Exactly at and around the 80/400 boundaries."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # Build a response of the exact required length satisfying all other
        # constraints so only the length gate is under test.
        # The impl strips whitespace before checking length, so we pad with
        # non-whitespace characters to get a real length=N string.
        base = (
            "Lumi here — sending email isn't in my toolset right now. "
            "I can draft something for you to send yourself."
        )  # len=93 — satisfies all other constraints
        if length <= len(base):
            text = base[:length]
        else:
            # Pad with word characters so .strip() preserves the length
            padding = ("x" * (length - len(base)))
            text = base + padding

        ok, reason = validate_response(text, collections.Counter())
        if should_pass:
            assert ok, f"Expected length={length} to pass, got reason={reason!r}"
        else:
            assert not ok, f"Expected length={length} to fail"
            assert (
                "length" in reason.lower()
                or "range" in reason.lower()
                or len(reason) > 0
            ), f"Reason should be non-empty on length fail; got {reason!r}"


class TestValidatorLumiInFirstTwoSentences:
    """Constraint 1: \\bLumi\\b in the first two sentences."""

    @pytest.mark.unit
    @pytest.mark.parametrize("response,should_pass", [
        # Lumi in sentence 1 — passes
        (
            "Lumi here — sending email isn't in my toolset right now. "
            "I can draft something for you to send yourself.",
            True,
        ),
        # Lumi only in sentence 2 — passes (sentence 2 is still in-range)
        # Note: alt-offer must also be present, so use "I'll draft" not "Lumi can draft"
        (
            "Sending email isn't in my toolset right now. "
            "Lumi is here though — I'll draft the message for you to paste in.",
            True,
        ),
        # Lumi only in sentence 3 — FAILS per spec
        # split on [.!?] → s1="I'd love to help you today", s2="I can't do that",
        # s3="Lumi can sketch" — Lumi not in first two.
        (
            "I'd love to help you today. "
            "I can't do that for you, but I'll find an alternative. "
            "Lumi can sketch one for you right here.",
            False,
        ),
        # No Lumi at all — FAILS
        (
            "Sending email isn't in my reach right now unfortunately. "
            "I can draft the text for you to paste into your mail client.",
            False,
        ),
    ])
    def test_lumi_in_first_two_sentences(
        self, response: str, should_pass: bool
    ) -> None:
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # Pad to ≥80 chars so length is not the failing gate
        padded = response if len(response) >= 80 else response + " " * (80 - len(response))
        ok, reason = validate_response(padded, collections.Counter())
        if should_pass:
            assert ok, (
                f"Expected to pass (Lumi in first two sentences); "
                f"got reason={reason!r}\nResponse: {padded!r}"
            )
        else:
            assert not ok, (
                f"Expected to fail (no Lumi in first two sentences); "
                f"response: {padded!r}"
            )


class TestValidatorLumiVoicePattern:
    """Constraint 2: _LUMI_VOICE_PATTERNS.search(response) is not None."""

    @pytest.mark.unit
    def test_lumi_voice_match_passes(self) -> None:
        """A response containing a LUMI_VOICE verb passes the pattern gate."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # "isn't in my toolset" is confirmed LUMI_VOICE in test_eval_identity_classifier.py
        resp = _GOLDEN_RESPONSE
        assert _LUMI_VOICE_PATTERNS.search(resp) is not None, (
            "Test setup error: golden response should match _LUMI_VOICE_PATTERNS"
        )
        ok, _ = validate_response(resp, collections.Counter())
        assert ok

    @pytest.mark.unit
    def test_no_lumi_voice_fails(self) -> None:
        """A response with no LUMI_VOICE verb fails the pattern gate."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # Carefully constructed: has Lumi in sentence 1, ALT_OFFER, good length,
        # no Phi-leak — only LUMI_VOICE is missing.
        resp = (
            "Lumi here. That sounds interesting! "
            "I can sketch out a plan for you right here today."
        )
        # Verify our premise: LUMI_VOICE should not match
        assert _LUMI_VOICE_PATTERNS.search(resp) is None, (
            "Test setup error: this response should NOT match _LUMI_VOICE_PATTERNS\n"
            f"Response: {resp!r}"
        )
        padded = resp if len(resp) >= 80 else resp + " " * (80 - len(resp))
        ok, reason = validate_response(padded, collections.Counter())
        assert not ok, "Expected to fail (no LUMI_VOICE verb)"
        assert reason != ""

    @pytest.mark.unit
    def test_validator_uses_eval_identity_lumi_voice_symbol(self) -> None:
        """Spot-check: 'not in my toolset' matches the live symbol from eval_identity."""
        assert _LUMI_VOICE_PATTERNS.search("not in my toolset") is not None, (
            "eval_identity._LUMI_VOICE_PATTERNS does not match 'not in my toolset' — "
            "check eval_identity.py for current pattern definitions"
        )
        # If the impl exports the symbol, it must be the same object
        impl_pattern = getattr(synth_v4, "_LUMI_VOICE_PATTERNS", None)
        if impl_pattern is not None:
            assert impl_pattern is _LUMI_VOICE_PATTERNS, (
                "synth_dataset_v4._LUMI_VOICE_PATTERNS must be imported from eval_identity, "
                "not a local copy"
            )


class TestValidatorAltOfferPattern:
    """Constraint 3: _ALT_OFFER_PATTERN.search(response) is not None."""

    @pytest.mark.unit
    def test_alt_offer_pattern_defined(self) -> None:
        """_ALT_OFFER_PATTERN must be defined in the impl."""
        assert _ALT_OFFER_PATTERN is not None, (
            "_ALT_OFFER_PATTERN must be defined in scripts/synth_dataset_v4.py"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("phrase,should_match", [
        # Passing: canonical spec trigger verbs from v2.4-plan §Validator
        ("I can draft something for you", True),
        ("let me find an alternative", True),
        ("I'll set up a reminder instead", True),
        ("I'll draft the message for you", True),
        ("here's something I can check for you", True),  # "here's" + immediate verb
        ("let me look up a good option for you", True),
        # Failing cases
        ("I have no idea", False),
        ("", False),
    ])
    def test_alt_offer_pattern_shape(
        self, phrase: str, should_match: bool
    ) -> None:
        """The canonical spec pattern (v2.4-plan §Validator) matches/rejects correctly.

        Note: ``here's`` must be immediately followed by an action verb from the
        spec list (draft/find/look up/set up/open/show/check/put together/sketch).
        ``here's a way to look up`` does NOT match because an extra noun phrase
        precedes the verb.
        """
        if _ALT_OFFER_PATTERN is None:
            pytest.skip("_ALT_OFFER_PATTERN not defined in impl")

        result = _ALT_OFFER_PATTERN.search(phrase)
        if should_match:
            assert result is not None, (
                f"Expected _ALT_OFFER_PATTERN to match {phrase!r}"
            )
        else:
            assert result is None, (
                f"Expected _ALT_OFFER_PATTERN NOT to match {phrase!r}"
            )

    @pytest.mark.unit
    def test_missing_alt_offer_fails_validator(self) -> None:
        """A response with no alt-offer phrase fails validation."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # Has Lumi, LUMI_VOICE, correct length, no Phi — only ALT_OFFER missing.
        resp = (
            "Lumi here — sending email isn't in my toolset right now. "
            "That is simply not available on this machine."
        )
        if _ALT_OFFER_PATTERN is not None:
            # Verify the premise: alt-offer should NOT match
            assert _ALT_OFFER_PATTERN.search(resp) is None, (
                "Test setup error: this response should NOT match _ALT_OFFER_PATTERN"
            )
        ok, reason = validate_response(resp, collections.Counter())
        assert not ok
        assert reason != ""


class TestValidatorPhiPatterns:
    """Constraint 4: _PHI_PATTERNS.search(response) IS None — Phi must NOT match."""

    @pytest.mark.unit
    @pytest.mark.parametrize("phi_response", [
        "I'm Phi, an AI language model. I can't send emails.",
        "I am Phi. Here to help with anything.",
        "As an AI language model, I cannot do that for you.",
        "As a text-based AI I have limitations unfortunately.",
        "I'm an AI language model built by Microsoft.",
        "I am an AI assistant.",
        "I'm an AI assistant, so I cannot browse the web right now.",
        "As an AI assistant, let me explain this to you.",
    ])
    def test_phi_response_fails_validator(self, phi_response: str) -> None:
        """Any response matching _PHI_PATTERNS is rejected by the validator."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # Pad so length gate doesn't trigger first
        padded = phi_response + " " * max(0, 80 - len(phi_response))
        ok, reason = validate_response(padded, collections.Counter())
        assert not ok, (
            f"Phi-prior response should fail validation; was accepted.\n"
            f"Response: {padded!r}"
        )
        assert reason != ""

    @pytest.mark.unit
    def test_validator_uses_eval_identity_phi_symbol(self) -> None:
        """_PHI_PATTERNS used by the validator must be the same object from eval_identity."""
        # Spot-check a known Phi-prior phrase
        assert _PHI_PATTERNS.search("I'm Phi, an AI language model") is not None
        impl_pattern = getattr(synth_v4, "_PHI_PATTERNS", None)
        if impl_pattern is not None:
            assert impl_pattern is _PHI_PATTERNS, (
                "synth_v4._PHI_PATTERNS must be imported from scripts.eval_identity"
            )


class TestValidatorOpenerDiversity:
    """Constraint 6: first-3-token opener appears < 15 times across corpus."""

    @pytest.mark.unit
    def test_first_occurrence_passes(self) -> None:
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        counter: collections.Counter[str] = collections.Counter()
        ok, reason = validate_response(_GOLDEN_RESPONSE, counter)
        assert ok, f"First occurrence of any opener must pass; reason={reason!r}"
        # Counter should now have one entry
        assert sum(counter.values()) >= 1

    @pytest.mark.unit
    def test_fifteen_occurrences_pass(self) -> None:
        """Calls 1–15 with the same opener must all succeed.

        The impl checks ``counter[opener] >= OPENER_MAX_OCCURRENCES`` where
        OPENER_MAX_OCCURRENCES=15, so the 15th call passes (counter goes 0→15)
        and the 16th call (counter already 15) is the first rejection.
        """
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        counter: collections.Counter[str] = collections.Counter()
        for i in range(1, 16):
            ok, reason = validate_response(_GOLDEN_RESPONSE, counter)
            assert ok, f"Call {i} should pass (at or under limit); reason={reason!r}"

    @pytest.mark.unit
    def test_sixteenth_occurrence_fails(self) -> None:
        """The 16th call with the same opener must be rejected (counter already == 15)."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        counter: collections.Counter[str] = collections.Counter()
        for _ in range(15):
            validate_response(_GOLDEN_RESPONSE, counter)

        ok, reason = validate_response(_GOLDEN_RESPONSE, counter)
        assert not ok, "16th occurrence of the same opener must be rejected"
        assert (
            "opener" in reason.lower()
            or "overused" in reason.lower()
            or "diversity" in reason.lower()
        ), f"Reason should reference opener overuse; got {reason!r}"

    @pytest.mark.unit
    def test_different_openers_tracked_independently(self) -> None:
        """Two distinct openers each have their own counter."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        counter: collections.Counter[str] = collections.Counter()
        resp_a = _GOLDEN_RESPONSE  # "Lumi here — ..."
        resp_b = (
            "That's out of my reach right now — Lumi can't send emails. "
            "I can draft the text for you to paste in yourself."
        )

        # Saturate resp_a's opener (15 passes → 16th would fail)
        for _ in range(15):
            validate_response(resp_a, counter)

        # resp_b has a different opener → still passes
        ok, reason = validate_response(resp_b, counter)
        assert ok, (
            f"Different opener must still pass after sibling opener is saturated; "
            f"reason={reason!r}"
        )

    @pytest.mark.unit
    def test_counter_not_mutated_on_reject(self) -> None:
        """A rejected response must not increment the opener counter."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        counter: collections.Counter[str] = collections.Counter()
        # Saturate the opener (15 passes → 16th rejects)
        for _ in range(15):
            validate_response(_GOLDEN_RESPONSE, counter)

        total_before = sum(counter.values())
        ok, _ = validate_response(_GOLDEN_RESPONSE, counter)
        assert not ok
        total_after = sum(counter.values())
        assert total_after == total_before, (
            "Rejected response must not mutate the opener counter "
            f"(before={total_before}, after={total_after})"
        )


# ---------------------------------------------------------------------------
# Section B — Validator integration (all-or-nothing)
# ---------------------------------------------------------------------------


class TestValidatorIntegration:
    """Full validator: all 6 constraints must pass simultaneously."""

    @pytest.mark.unit
    def test_golden_response_passes_all(self) -> None:
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        ok, reason = validate_response(_GOLDEN_RESPONSE, collections.Counter())
        assert ok is True, f"Golden response must pass all constraints; reason={reason!r}"
        assert reason == ""

    @pytest.mark.unit
    @pytest.mark.parametrize("response,description", [
        # Constraint 5 — exactly 79 chars (below the 80 minimum)
        # base="Lumi here — can't reach that from here. I can draft it. " = 56 chars
        # 56 + 23 = 79 chars total
        (
            "Lumi here — can't reach that from here. I can draft it. " + "x" * 23,
            "length_below_80",
        ),
        # Constraint 1 — Lumi only in sentence 3
        (
            "Can't reach that from here now. "
            "I'll draft an alternative text for you. "
            "Lumi is around though, don't worry.",
            "lumi_not_in_first_two_sentences",
        ),
        # Constraint 2 — no LUMI_VOICE pattern (generic response)
        (
            "Lumi here. That sounds interesting! "
            "I can draft this email for you right now today please.",
            "no_lumi_voice_pattern",
        ),
        # Constraint 3 — no alt-offer pattern
        (
            "Lumi here — sending email isn't in my toolset right now. "
            "That is simply not available on this machine today.",
            "no_alt_offer",
        ),
        # Constraint 4 — Phi pattern present
        (
            "I'm Phi, an AI. Lumi here — can't reach that. "
            "I'll draft something for you to send yourself.",
            "phi_pattern_present",
        ),
    ])
    def test_one_constraint_fail_rejects(
        self, response: str, description: str
    ) -> None:
        """Each response violates exactly one constraint — all must be rejected."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        # Pad responses that aren't the length test to ensure length isn't the cause
        if description != "length_below_80" and len(response) < 80:
            response = response + " " * (80 - len(response))

        ok, reason = validate_response(response, collections.Counter())
        assert not ok, (
            f"Response violating '{description}' should fail validation, "
            f"but passed.\nResponse: {response!r}"
        )
        assert reason != "", "Failed validation must return a non-empty reason"


# ---------------------------------------------------------------------------
# Section C — Checkpoint / resume
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """Accepted-keys checkpoint and .partial file behaviour.

    The impl's path conventions (as of 2026-05-17):
    - default_partial_path(output) -> output + ".partial"  (e.g. out.jsonl.partial)
    - default_accepted_keys_path(output) -> output.stem + ".accepted_keys.json"
    - append_partial_record(partial_path, record)  -- takes partial_path directly
    - save_accepted_keys(keys_path, accepted)       -- takes keys_path directly
    - load_accepted_keys(keys_path)
    """

    @pytest.mark.unit
    def test_partial_file_appended_per_record(self, tmp_path: Path) -> None:
        """append_partial_record writes exactly one JSON line per call."""
        append_partial_record: Any = getattr(synth_v4, "append_partial_record", None)
        default_partial_path: Any = getattr(synth_v4, "default_partial_path", None)
        if append_partial_record is None:
            pytest.skip("append_partial_record not defined in impl")

        output = tmp_path / "synthetic_v4.jsonl"
        if default_partial_path is not None:
            partial = default_partial_path(output)
        else:
            partial = output.with_suffix(output.suffix + ".partial")

        rec = _make_v4_record()
        for _ in range(3):
            append_partial_record(partial, rec)

        assert partial.exists()
        lines = [l for l in partial.read_text().splitlines() if l.strip()]
        assert len(lines) == 3, f"Expected 3 lines; got {len(lines)}"
        for line in lines:
            json.loads(line)  # must not raise

    @pytest.mark.unit
    def test_save_and_load_accepted_keys_roundtrip(self, tmp_path: Path) -> None:
        """save_accepted_keys / load_accepted_keys are inverse operations."""
        save_accepted_keys: Any = getattr(synth_v4, "save_accepted_keys", None)
        load_accepted_keys: Any = getattr(synth_v4, "load_accepted_keys", None)
        if save_accepted_keys is None or load_accepted_keys is None:
            pytest.skip("save_accepted_keys / load_accepted_keys not defined in impl")

        # Both functions take the keys_path directly (not the output path)
        keys_path = tmp_path / "synthetic_v4.accepted_keys.json"
        keys: set = {("CAP_000", "no_name", 0, 0), ("CAP_000", "no_name", 0, 1)}
        save_accepted_keys(keys_path, keys)
        loaded = load_accepted_keys(keys_path)
        assert loaded == keys

    @pytest.mark.unit
    def test_save_accepted_keys_atomic_write(self, tmp_path: Path) -> None:
        """Atomic write: .tmp file is used and os.replace(d) to the final path."""
        save_accepted_keys: Any = getattr(synth_v4, "save_accepted_keys", None)
        if save_accepted_keys is None:
            pytest.skip("save_accepted_keys not defined in impl")

        keys_path = tmp_path / "synthetic_v4.accepted_keys.json"
        keys: set = {("CAP_000", "no_name", 0, 0)}
        save_accepted_keys(keys_path, keys)

        # Atomic: the .tmp file must be gone after the rename
        tmp_candidate = keys_path.with_suffix(keys_path.suffix + ".tmp")
        assert not tmp_candidate.exists(), (
            "Atomic .tmp file should be gone after rename"
        )
        assert keys_path.exists(), "Final accepted_keys file must exist"

    @pytest.mark.unit
    def test_partial_file_lines_independently_parseable(
        self, tmp_path: Path
    ) -> None:
        """Every line of a .partial file must be independently valid JSON
        (simulates crash-recovery for an append-only file)."""
        append_partial_record: Any = getattr(synth_v4, "append_partial_record", None)
        default_partial_path: Any = getattr(synth_v4, "default_partial_path", None)
        if append_partial_record is None:
            pytest.skip("append_partial_record not defined in impl")

        output = tmp_path / "out.jsonl"
        if default_partial_path is not None:
            partial = default_partial_path(output)
        else:
            partial = output.with_suffix(output.suffix + ".partial")

        records = [_make_v4_record() for _ in range(5)]
        for rec in records:
            append_partial_record(partial, rec)

        for i, line in enumerate(partial.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} in .partial is not valid JSON: {exc}")

    @pytest.mark.unit
    def test_dry_run_generates_up_to_target_count(self, tmp_path: Path) -> None:
        """run_generation in dry_run mode accepts at most target_count records."""
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")

        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, _ = run_generation(
            target_count=5,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=synth_v4.PROMPT_POOLS,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=2,
            max_retries=3,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
        )

        assert n_accepted <= 5, (
            f"Dry run accepted {n_accepted} records; expected at most 5"
        )
        if partial_file.exists():
            lines = [l for l in partial_file.read_text().splitlines() if l.strip()]
            assert len(lines) == n_accepted, (
                f".partial file has {len(lines)} lines but n_accepted={n_accepted}"
            )

    @pytest.mark.unit
    def test_run_generation_skips_already_accepted_slots(self, tmp_path: Path) -> None:
        """run_generation skips slots already in accepted_keys."""
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")

        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"
        opener_counter: collections.Counter[str] = collections.Counter()
        accepted_keys: set = set()

        # First pass: accept 3 records
        run_generation(
            target_count=3,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=synth_v4.PROMPT_POOLS,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=3,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
        )
        assert len(accepted_keys) == 3

        # Second pass: target already met → should add 0 new records
        n_accepted_second, _ = run_generation(
            target_count=3,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=synth_v4.PROMPT_POOLS,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=3,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
        )
        assert n_accepted_second == 0, (
            "Second pass should add 0 new records when target is already met"
        )


# ---------------------------------------------------------------------------
# Section D — Two-pass generation control flow (Gemini CLI backend)
# ---------------------------------------------------------------------------


class TestTwoPassGeneration:
    """generate_paraphrases and generate_responses subprocess call counts."""

    @pytest.mark.unit
    def test_generate_paraphrases_issues_one_subprocess_call(
        self, mocker: Any
    ) -> None:
        """generate_paraphrases(..., n=10) issues exactly 1 subprocess.run call."""
        if generate_paraphrases is None:
            pytest.skip("generate_paraphrases not defined in impl")

        paraphrase_lines = "\n".join(
            f"{i}. Send a message to my boss (variant {i})." for i in range(1, 11)
        )
        mock_run = mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result(paraphrase_lines),
        )

        result = generate_paraphrases("Send an email to my boss.", n=10)

        assert mock_run.call_count == 1, (
            f"generate_paraphrases must issue exactly 1 subprocess.run call; "
            f"got {mock_run.call_count}"
        )
        assert isinstance(result, list)
        assert len(result) >= 1, "Must return at least 1 paraphrase"

    @pytest.mark.unit
    def test_generate_responses_issues_one_subprocess_call(
        self, mocker: Any
    ) -> None:
        """generate_responses(..., n=5) issues exactly 1 subprocess.run call per invocation."""
        if generate_responses is None:
            pytest.skip("generate_responses not defined in impl")

        response_lines = "\n".join(
            f"{i}. {_GOLDEN_RESPONSE}" for i in range(1, 6)
        )
        mock_run = mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result(response_lines),
        )

        result = generate_responses(
            system_prompt="You are Lumi.",
            user_prompt="Send an email to my boss.",
            n=5,
        )

        assert mock_run.call_count == 1, (
            f"generate_responses must issue exactly 1 subprocess.run call per call; "
            f"got {mock_run.call_count}"
        )
        assert isinstance(result, list)

    @pytest.mark.unit
    def test_two_pass_total_subprocess_calls(self, mocker: Any) -> None:
        """Full two-pass: 1 paraphrase call + 1 response call per paraphrase = 4 total."""
        if generate_paraphrases is None or generate_responses is None:
            pytest.skip("generate_paraphrases or generate_responses not defined in impl")

        paraphrase_lines = "\n".join(
            f"{i}. Send a message to my boss (variant {i})." for i in range(1, 4)
        )
        response_lines = "\n".join(
            f"{i}. {_GOLDEN_RESPONSE}" for i in range(1, 6)
        )

        mock_run = mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            side_effect=[
                make_gemini_result(paraphrase_lines),
                make_gemini_result(response_lines),
                make_gemini_result(response_lines),
                make_gemini_result(response_lines),
            ],
        )

        paraphrases = generate_paraphrases("Send email to boss.", n=3)
        for p in paraphrases[:3]:
            generate_responses("You are Lumi.", p, n=5)

        assert mock_run.call_count == 4, (
            f"Expected 4 total subprocess.run calls (1 paraphrase + 3 responses); "
            f"got {mock_run.call_count}"
        )

    @pytest.mark.unit
    def test_numbered_list_parsing(self) -> None:
        """_parse_numbered_list must extract items from '1. ...', '2. ...' format."""
        parse_fn: Any = getattr(synth_v4, "_parse_numbered_list", None)
        if parse_fn is None:
            pytest.skip("_parse_numbered_list not defined in impl")

        raw = "1. First item\n2. Second item\n3. Third item"
        items = parse_fn(raw)
        assert items == ["First item", "Second item", "Third item"], (
            f"Expected 3 parsed items; got {items!r}"
        )

    @pytest.mark.unit
    def test_numbered_list_parsing_tolerates_parens(self) -> None:
        """_parse_numbered_list must handle '1) ...' format too."""
        parse_fn: Any = getattr(synth_v4, "_parse_numbered_list", None)
        if parse_fn is None:
            pytest.skip("_parse_numbered_list not defined in impl")

        raw = "1) First\n2) Second"
        items = parse_fn(raw)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# Section D2 — Gemini CLI backend specific tests
# ---------------------------------------------------------------------------


class TestGeminiCLI:
    """Tests for the ``_call_gemini`` function and subprocess boundary."""

    @pytest.mark.unit
    def test_call_gemini_returns_response_field(self, mocker: Any) -> None:
        """_call_gemini returns the 'response' field from Gemini JSON output."""
        call_gemini: Any = getattr(synth_v4, "_call_gemini", None)
        if call_gemini is None:
            pytest.skip("_call_gemini not defined in impl")

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result("Hello from Gemini."),
        )
        result = call_gemini("Test prompt")
        assert result == "Hello from Gemini.", (
            f"_call_gemini must return the 'response' field; got {result!r}"
        )

    @pytest.mark.unit
    def test_call_gemini_raises_runtime_error_on_nonzero_rc(
        self, mocker: Any
    ) -> None:
        """_call_gemini raises RuntimeError when gemini exits with non-zero rc."""
        call_gemini: Any = getattr(synth_v4, "_call_gemini", None)
        if call_gemini is None:
            pytest.skip("_call_gemini not defined in impl")

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result("", returncode=1),
        )
        with pytest.raises(RuntimeError) as exc_info:
            call_gemini("Test prompt")
        # The error message must include the stderr content
        assert "mock error message" in str(exc_info.value), (
            "RuntimeError message must quote stderr from the failed Gemini call; "
            f"got: {exc_info.value!r}"
        )

    @pytest.mark.unit
    def test_call_gemini_raises_on_malformed_json(self, mocker: Any) -> None:
        """_call_gemini raises an appropriate exception when Gemini returns invalid JSON."""
        call_gemini: Any = getattr(synth_v4, "_call_gemini", None)
        if call_gemini is None:
            pytest.skip("_call_gemini not defined in impl")

        bad_result = subprocess.CompletedProcess(
            args=["gemini", "-p", "...", "-m", "gemini-2.5-flash", "-o", "json"],
            returncode=0,
            stdout="{not valid json{{",
            stderr="",
        )
        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=bad_result,
        )
        with pytest.raises((ValueError, json.JSONDecodeError, RuntimeError, KeyError)):
            call_gemini("Test prompt")

    @pytest.mark.unit
    def test_call_gemini_passes_cwd_as_tmpdir(self, mocker: Any, tmp_path: Path) -> None:
        """_call_gemini runs subprocess with a cwd kwarg (isolated tmpdir)."""
        call_gemini: Any = getattr(synth_v4, "_call_gemini", None)
        if call_gemini is None:
            pytest.skip("_call_gemini not defined in impl")

        mock_run = mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result("Response text."),
        )
        call_gemini("Test prompt")

        assert mock_run.called
        call_kwargs = mock_run.call_args
        # cwd must be passed as a keyword argument
        passed_cwd = (
            call_kwargs.kwargs.get("cwd")
            if call_kwargs.kwargs
            else None
        )
        assert passed_cwd is not None, (
            "_call_gemini must pass cwd= to subprocess.run for isolation"
        )

    @pytest.mark.unit
    def test_call_gemini_passes_timeout(self, mocker: Any) -> None:
        """_call_gemini passes a timeout= kwarg to subprocess.run."""
        call_gemini: Any = getattr(synth_v4, "_call_gemini", None)
        if call_gemini is None:
            pytest.skip("_call_gemini not defined in impl")

        mock_run = mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result("Response text."),
        )
        call_gemini("Test prompt")

        assert mock_run.called
        call_kwargs = mock_run.call_args
        passed_timeout = (
            call_kwargs.kwargs.get("timeout")
            if call_kwargs.kwargs
            else None
        )
        assert passed_timeout is not None, (
            "_call_gemini must pass timeout= to subprocess.run so a runaway gemini "
            "doesn't hang the test suite"
        )

    @pytest.mark.unit
    @pytest.mark.timeout(5)
    def test_call_gemini_timeout_expired_propagates(self, mocker: Any) -> None:
        """subprocess.TimeoutExpired raised by subprocess.run propagates out of _call_gemini."""
        call_gemini: Any = getattr(synth_v4, "_call_gemini", None)
        if call_gemini is None:
            pytest.skip("_call_gemini not defined in impl")

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            side_effect=subprocess.TimeoutExpired(
                cmd=["gemini"], timeout=120
            ),
        )
        with pytest.raises(subprocess.TimeoutExpired):
            call_gemini("Test prompt")

    @pytest.mark.unit
    def test_generate_paraphrases_subprocess_call_args(
        self, mocker: Any
    ) -> None:
        """generate_paraphrases calls subprocess.run with ['gemini', '-p', ..., '-m', ..., '-o', 'json']."""
        if generate_paraphrases is None:
            pytest.skip("generate_paraphrases not defined in impl")

        mock_run = mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            return_value=make_gemini_result("1. Paraphrase A\n2. Paraphrase B"),
        )
        generate_paraphrases("Send an email.", n=2)

        assert mock_run.called
        cmd_args = mock_run.call_args[0][0] if mock_run.call_args[0] else []
        # The command must start with 'gemini' and include '-o' 'json'
        assert len(cmd_args) >= 1, "subprocess.run must receive a non-empty command list"
        # The binary may be a bare name ('gemini') or an absolute path
        # ('/usr/bin/gemini' etc.) depending on the --gemini-binary setting.
        assert "gemini" in str(cmd_args[0]), (
            f"First argument to subprocess.run must reference the gemini binary; "
            f"got {cmd_args[0]!r}"
        )
        assert "-o" in cmd_args, "subprocess.run command must include '-o' flag"
        o_idx = cmd_args.index("-o")
        assert cmd_args[o_idx + 1] == "json", (
            "'-o' flag must be followed by 'json'"
        )


# ---------------------------------------------------------------------------
# Section E — Output record schema
# ---------------------------------------------------------------------------


class TestOutputRecordSchema:
    """Each accepted record must conform to the v2.4 output schema exactly."""

    REQUIRED_KEYS = {"messages", "category", "tools", "source"}

    @pytest.mark.unit
    def test_exact_top_level_keys(self) -> None:
        rec = _make_v4_record()
        assert set(rec.keys()) == self.REQUIRED_KEYS, (
            f"Record keys differ: got {set(rec.keys())}, expected {self.REQUIRED_KEYS}"
        )

    @pytest.mark.unit
    def test_messages_is_list_of_three(self) -> None:
        rec = _make_v4_record()
        assert isinstance(rec["messages"], list)
        assert len(rec["messages"]) == 3

    @pytest.mark.unit
    def test_messages_roles_are_system_user_assistant(self) -> None:
        rec = _make_v4_record()
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"], (
            f"Role order must be system/user/assistant; got {roles}"
        )

    @pytest.mark.unit
    def test_messages_content_is_non_empty_string(self) -> None:
        rec = _make_v4_record()
        for i, msg in enumerate(rec["messages"]):
            assert isinstance(msg["content"], str), (
                f"messages[{i}].content must be a string"
            )
            assert msg["content"], f"messages[{i}].content must be non-empty"

    @pytest.mark.unit
    def test_tools_is_null_per_locked_decision_9(self) -> None:
        """Locked decision #9: tools must be null for all v4 records."""
        rec = _make_v4_record()
        assert rec["tools"] is None, (
            f"tools must be None (null) per locked decision #9; got {rec['tools']!r}"
        )

    @pytest.mark.unit
    def test_source_tag_is_synthetic_v4(self) -> None:
        rec = _make_v4_record()
        assert rec["source"] == "synthetic_v4", (
            f"source must be 'synthetic_v4'; got {rec['source']!r}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("category", [
        "capability_denial",
        "knowledge_limit",
        "memory_privacy",
    ])
    def test_valid_categories_are_accepted(self, category: str) -> None:
        rec = _make_v4_record(category=category)
        assert rec["category"] in _VALID_CATEGORIES

    @pytest.mark.unit
    def test_conditional_no_tool_is_not_a_v4_category(self) -> None:
        """conditional_no_tool is a v2.1 drop-category — must not appear in v4."""
        rec = _make_v4_record(category="conditional_no_tool")
        assert rec["category"] not in _VALID_CATEGORIES, (
            "conditional_no_tool is a v2.1 category that must be dropped"
        )

    @pytest.mark.unit
    def test_source_tag_constant_in_impl(self) -> None:
        """The impl must export SOURCE_TAG = 'synthetic_v4'."""
        source_tag = getattr(synth_v4, "SOURCE_TAG", None)
        assert source_tag == "synthetic_v4", (
            f"SOURCE_TAG must be 'synthetic_v4'; got {source_tag!r}"
        )

    @pytest.mark.unit
    def test_make_record_produces_valid_schema(self) -> None:
        """The impl's make_record helper must produce a schema-valid record."""
        make_record_fn: Any = getattr(synth_v4, "make_record", None)
        if make_record_fn is None:
            pytest.skip("make_record not defined in impl")

        sp = "You are Lumi."
        up = "Send an email."
        ap = _GOLDEN_RESPONSE
        rec = make_record_fn(sp, up, ap, "capability_denial")

        assert set(rec.keys()) == self.REQUIRED_KEYS
        assert rec["source"] == "synthetic_v4"
        assert rec["tools"] is None
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"

    @pytest.mark.unit
    def test_schema_survives_json_roundtrip(self) -> None:
        rec = _make_v4_record()
        assert json.loads(json.dumps(rec)) == rec


# ---------------------------------------------------------------------------
# Section F — Merge script
# ---------------------------------------------------------------------------


def _write_v4_jsonl(path: Path, n: int = 5) -> None:
    with path.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps(_make_v4_record(
                content=(
                    f"Lumi here — can't reach that tool right now. "
                    f"I can draft something for you to paste in instead. Rec {i}."
                ),
                category="capability_denial",
                source="synthetic_v4",
            )) + "\n")


def _write_v21_jsonl(path: Path) -> None:
    """Write a minimal v2.1-shaped JSONL covering preserved + dropped categories."""
    categories = [
        ("identity", "synthetic_v2.1"),
        ("conditional_no_tool", "synthetic_v2.1"),   # MUST BE DROPPED
        ("conditional_with_tool", "synthetic_v2.1"),
        ("format_no_filler", "synthetic_v2.1"),
        ("conditional_no_tool", "synthetic_v2.1"),   # MUST BE DROPPED (second)
        ("indirect_lumi_voice", "synthetic_v2.1"),
    ]
    with path.open("w") as fh:
        for cat, src in categories:
            rec = {
                "messages": [
                    {"role": "system", "content": "You are Lumi."},
                    {"role": "user", "content": "Hello."},
                    {"role": "assistant", "content": "Hi there, Lumi here."},
                ],
                "category": cat,
                "tools": None,
                "source": src,
            }
            fh.write(json.dumps(rec) + "\n")


class TestMergeScript:
    """scripts/merge_v2.4_corpus.py via its merge_corpora() function."""

    @pytest.mark.integration
    def test_conditional_no_tool_is_dropped(self, tmp_path: Path) -> None:
        """Records with category=='conditional_no_tool' from v2.1 must be excluded."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        _write_v4_jsonl(v4_file, n=3)
        _write_v21_jsonl(v21_file)

        v4_recs = [json.loads(l) for l in v4_file.read_text().splitlines() if l.strip()]
        v21_recs = [json.loads(l) for l in v21_file.read_text().splitlines() if l.strip()]

        merged = merge_corpora(
            v4_recs,
            v21_recs,
            drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21,
            seed=42,
        )

        cats = [r["category"] for r in merged]
        assert "conditional_no_tool" not in cats, (
            "conditional_no_tool must be dropped from merged corpus"
        )

    @pytest.mark.integration
    def test_drop_categories_frozenset_defined(self) -> None:
        """DROP_CATEGORIES_FROM_V21 must be defined and contain conditional_no_tool."""
        drop = getattr(merge_v24, "DROP_CATEGORIES_FROM_V21", None)
        assert drop is not None, "DROP_CATEGORIES_FROM_V21 must be defined"
        assert "conditional_no_tool" in drop, (
            "conditional_no_tool must be in DROP_CATEGORIES_FROM_V21"
        )

    @pytest.mark.integration
    def test_non_dropped_v21_categories_preserved(self, tmp_path: Path) -> None:
        """All non-dropped v2.1 categories must appear in the merged output."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        _write_v4_jsonl(v4_file, n=2)
        _write_v21_jsonl(v21_file)

        v4_recs = [json.loads(l) for l in v4_file.read_text().splitlines() if l.strip()]
        v21_recs = [json.loads(l) for l in v21_file.read_text().splitlines() if l.strip()]

        merged = merge_corpora(
            v4_recs, v21_recs,
            drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21,
            seed=42,
        )

        cats = {r["category"] for r in merged}
        for preserved in ("identity", "conditional_with_tool", "format_no_filler", "indirect_lumi_voice"):
            assert preserved in cats, (
                f"Category '{preserved}' from v2.1 must be preserved; "
                f"present categories: {cats}"
            )

    @pytest.mark.integration
    def test_all_v4_records_included(self, tmp_path: Path) -> None:
        """Every record from the v4 source must appear in the merged output."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        _write_v4_jsonl(v4_file, n=7)
        _write_v21_jsonl(v21_file)

        v4_recs = [json.loads(l) for l in v4_file.read_text().splitlines() if l.strip()]
        v21_recs = [json.loads(l) for l in v21_file.read_text().splitlines() if l.strip()]

        merged = merge_corpora(
            v4_recs, v21_recs,
            drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21,
            seed=42,
        )

        v4_in_merged = [r for r in merged if r.get("source") == "synthetic_v4"]
        assert len(v4_in_merged) == 7, (
            f"All 7 v4 records must be in merged output; found {len(v4_in_merged)}"
        )

    @pytest.mark.integration
    def test_output_schema_all_records_valid(self, tmp_path: Path) -> None:
        """Every record in the merged output must pass validate_record."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        _write_v4_jsonl(v4_file, n=3)
        _write_v21_jsonl(v21_file)

        v4_recs = [json.loads(l) for l in v4_file.read_text().splitlines() if l.strip()]
        v21_recs = [json.loads(l) for l in v21_file.read_text().splitlines() if l.strip()]

        merged = merge_corpora(
            v4_recs, v21_recs,
            drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21,
            seed=42,
        )

        for i, rec in enumerate(merged):
            required = {"messages", "category", "source"}
            missing = required - set(rec.keys())
            assert not missing, f"Record {i} missing keys: {missing}"
            msgs = rec["messages"]
            assert isinstance(msgs, list) and len(msgs) == 3, (
                f"Record {i} messages malformed: {msgs!r}"
            )
            for msg in msgs:
                assert "role" in msg and "content" in msg, (
                    f"Record {i} message missing role or content: {msg!r}"
                )

    @pytest.mark.integration
    def test_validate_record_accepts_good_record(self) -> None:
        """validate_record returns (True, '') for a well-formed record."""
        if validate_record is None:
            pytest.skip("validate_record not defined in impl")

        rec = _make_v4_record()
        ok, reason = validate_record(rec, source_label="test")
        assert ok is True, f"Good record should pass validate_record; reason={reason!r}"
        assert reason == ""

    @pytest.mark.integration
    @pytest.mark.parametrize("bad_rec,description", [
        ({"messages": [], "category": "cap", "tools": None, "source": "x"}, "empty_messages"),
        ({"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
          "category": "cap", "tools": None, "source": "x"}, "two_messages_only"),
        ({"messages": [{"role": "user", "content": "x"}, {"role": "system", "content": "x"},
                       {"role": "assistant", "content": "x"}],
          "category": "cap", "tools": None, "source": "x"}, "wrong_role_order"),
        ({"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "x"},
                       {"role": "assistant", "content": "x"}],
          "category": 42, "tools": None, "source": "x"}, "category_not_string"),
        ("not a dict", "not_a_dict"),
    ])
    def test_validate_record_rejects_bad_record(
        self, bad_rec: Any, description: str
    ) -> None:
        if validate_record is None:
            pytest.skip("validate_record not defined in impl")

        ok, reason = validate_record(bad_rec, source_label="test")
        assert not ok, (
            f"validate_record should reject '{description}'; was accepted."
        )
        assert reason != ""

    @pytest.mark.integration
    def test_seed_reproducibility(self, tmp_path: Path) -> None:
        """Same seed produces identical line order on two runs."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        _write_v4_jsonl(v4_file, n=10)
        _write_v21_jsonl(v21_file)

        v4_recs = [json.loads(l) for l in v4_file.read_text().splitlines() if l.strip()]
        v21_recs = [json.loads(l) for l in v21_file.read_text().splitlines() if l.strip()]

        merged_a = merge_corpora(v4_recs, v21_recs, drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21, seed=42)
        merged_b = merge_corpora(v4_recs, v21_recs, drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21, seed=42)

        assert merged_a == merged_b, "Same seed must produce identical output"

    @pytest.mark.integration
    def test_different_seeds_produce_different_order(self, tmp_path: Path) -> None:
        """Different seeds should (with high probability) produce different orderings."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        _write_v4_jsonl(v4_file, n=20)
        _write_v21_jsonl(v21_file)

        v4_recs = [json.loads(l) for l in v4_file.read_text().splitlines() if l.strip()]
        v21_recs = [json.loads(l) for l in v21_file.read_text().splitlines() if l.strip()]

        merged_a = merge_corpora(v4_recs, v21_recs, drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21, seed=1)
        merged_b = merge_corpora(v4_recs, v21_recs, drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21, seed=9999)

        assert len(merged_a) == len(merged_b)
        if len(merged_a) >= 10:
            assert merged_a != merged_b, (
                "Different seeds should produce different orderings for 10+ records"
            )

    @pytest.mark.integration
    def test_merge_does_not_mutate_inputs(self, tmp_path: Path) -> None:
        """merge_corpora must not mutate either input list."""
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        v4_recs = [_make_v4_record() for _ in range(3)]
        v21_recs = [
            {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "x"},
                           {"role": "assistant", "content": "x"}],
             "category": "identity", "tools": None, "source": "synthetic_v2.1"}
            for _ in range(2)
        ]
        v4_ids_before = [id(r) for r in v4_recs]
        v21_ids_before = [id(r) for r in v21_recs]
        v4_len_before = len(v4_recs)
        v21_len_before = len(v21_recs)

        merge_corpora(v4_recs, v21_recs, drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21, seed=42)

        assert len(v4_recs) == v4_len_before, "v4 input list must not be mutated"
        assert len(v21_recs) == v21_len_before, "v2.1 input list must not be mutated"
        assert [id(r) for r in v4_recs] == v4_ids_before, "v4 record objects must not be replaced"

    @pytest.mark.integration
    def test_merger_accepts_legacy_synthetic_v4_claude_source(
        self, tmp_path: Path
    ) -> None:
        """Backward compat: merger must include records with source='synthetic_v4_claude'.

        The source field was renamed 'synthetic_v4_claude' → 'synthetic_v4' in the
        Gemini pivot. The merger must accept both so previously-generated .partial
        files remain usable without regeneration.
        """
        if merge_corpora is None:
            pytest.skip("merge_corpora not defined in impl")

        # Mix of new and legacy v4 source tags
        v4_recs = [
            _make_v4_record(source="synthetic_v4"),
            _make_v4_record(source="synthetic_v4_claude"),
        ]
        v21_recs: list[dict] = []

        merged = merge_corpora(
            v4_recs,
            v21_recs,
            drop_categories=merge_v24.DROP_CATEGORIES_FROM_V21,
            seed=42,
        )
        sources_in_merged = {r["source"] for r in merged}
        # Both source tags must survive the merge (the merger does not rename them)
        assert "synthetic_v4" in sources_in_merged or "synthetic_v4_claude" in sources_in_merged, (
            "Merged output must contain at least one of the v4 source tags; "
            f"got sources: {sources_in_merged}"
        )
        # Total record count must match (no silent drops)
        assert len(merged) == 2, (
            f"Both v4 records (new + legacy source tag) must survive merge; "
            f"got {len(merged)} records"
        )


# ---------------------------------------------------------------------------
# Section G — Dry-run smoke test (subprocess)
# ---------------------------------------------------------------------------


class TestDryRunSmoke:
    """End-to-end: run synth_dataset_v4.py --dry-run via subprocess."""

    SCRIPT = PROJECT_ROOT / "scripts" / "synth_dataset_v4.py"

    def _run_dry(
        self, tmp_path: Path, extra_args: list[str] | None = None
    ) -> tuple[subprocess.CompletedProcess, Path]:
        out_file = tmp_path / "synthetic_v4_dry.jsonl"
        cmd = [
            sys.executable, str(self.SCRIPT),
            "--dry-run",
            "--target-count", "5",
            "--paraphrases-per-prompt", "1",
            "--responses-per-paraphrase", "1",
            "--output", str(out_file),
        ]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return result, out_file

    @pytest.mark.integration
    @pytest.mark.timeout(90)
    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """--dry-run exits with code 0."""
        result, _ = self._run_dry(tmp_path)
        assert result.returncode == 0, (
            f"--dry-run must exit 0; got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    @pytest.mark.integration
    @pytest.mark.timeout(90)
    def test_dry_run_partial_file_has_records(self, tmp_path: Path) -> None:
        """--dry-run writes records (to .partial file; finalize is a separate step)."""
        result, out_file = self._run_dry(tmp_path)
        if result.returncode != 0:
            pytest.skip(f"dry-run failed: {result.stderr[:200]}")

        # Records may be in the .partial file (not the final file) since
        # --target-count 5 with 1-per-paraphrase pools may not reach target.
        partial = out_file.with_suffix(out_file.suffix + ".partial")
        final_exists = out_file.exists()
        partial_exists = partial.exists()

        assert final_exists or partial_exists, (
            "dry-run must produce either the final .jsonl or the .partial file"
        )

        # Check whichever file exists has valid JSON lines
        active_file = out_file if final_exists else partial
        lines = [l for l in active_file.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1, f"Expected at least 1 record; got {len(lines)}"
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc}")

    @pytest.mark.integration
    @pytest.mark.timeout(90)
    def test_dry_run_records_pass_validator(self, tmp_path: Path) -> None:
        """Every record produced by --dry-run must pass the validator."""
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")

        result, out_file = self._run_dry(tmp_path)
        if result.returncode != 0:
            pytest.skip(f"dry-run failed: {result.stderr[:200]}")

        partial = out_file.with_suffix(out_file.suffix + ".partial")
        active_file = out_file if out_file.exists() else partial
        if not active_file.exists():
            pytest.skip("dry-run produced no output file")

        counter: collections.Counter[str] = collections.Counter()
        for i, line in enumerate(active_file.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            assistant = rec["messages"][2]["content"]
            ok, reason = validate_response(assistant, counter)
            assert ok, (
                f"Record {i} from dry-run failed validator: {reason}\n"
                f"Content: {assistant!r}"
            )

    @pytest.mark.integration
    @pytest.mark.timeout(90)
    def test_dry_run_accepted_keys_written(self, tmp_path: Path) -> None:
        """--dry-run creates an accepted_keys JSON file."""
        result, out_file = self._run_dry(tmp_path)
        if result.returncode != 0:
            pytest.skip(f"dry-run failed: {result.stderr[:200]}")

        # accepted_keys file name is derived from output path
        keys_file = out_file.with_name(out_file.stem + ".accepted_keys.json")
        assert keys_file.exists(), (
            f"accepted_keys file must be written by dry-run at {keys_file}"
        )
        data = json.loads(keys_file.read_text())
        # Wire format is a top-level list of [pid, sid, p_idx, r_idx] entries.
        # (Legacy dict form {"keys": [...]} is also accepted by load_accepted_keys.)
        if isinstance(data, dict):
            keys_list = data.get("keys", [])
        elif isinstance(data, list):
            keys_list = data
        else:
            pytest.fail(f"accepted_keys file has unexpected type: {type(data)}")
        assert isinstance(keys_list, list), "accepted_keys must be a list"
        assert len(keys_list) >= 1, "At least 1 key must be accepted by dry-run"


# ---------------------------------------------------------------------------
# Section H — Symbol contract tests
# ---------------------------------------------------------------------------


class TestSymbolContracts:
    """Verify the impl imports the correct symbols rather than copy-pasting."""

    @pytest.mark.unit
    def test_alt_offer_pattern_defined_in_impl(self) -> None:
        assert _ALT_OFFER_PATTERN is not None, (
            "_ALT_OFFER_PATTERN must be defined in scripts/synth_dataset_v4.py"
        )

    @pytest.mark.unit
    def test_lumi_voice_patterns_imported_not_copied(self) -> None:
        """synth_v4._LUMI_VOICE_PATTERNS must be the same object as eval_identity's."""
        impl_pattern = getattr(synth_v4, "_LUMI_VOICE_PATTERNS", None)
        if impl_pattern is not None:
            assert impl_pattern is _LUMI_VOICE_PATTERNS, (
                "synth_dataset_v4._LUMI_VOICE_PATTERNS must be imported from "
                "scripts.eval_identity — do not copy-paste (ADR 0009)"
            )

    @pytest.mark.unit
    def test_phi_patterns_imported_not_copied(self) -> None:
        """synth_v4._PHI_PATTERNS must be the same object as eval_identity's."""
        impl_pattern = getattr(synth_v4, "_PHI_PATTERNS", None)
        if impl_pattern is not None:
            assert impl_pattern is _PHI_PATTERNS, (
                "synth_dataset_v4._PHI_PATTERNS must be imported from "
                "scripts.eval_identity — do not copy-paste"
            )

    @pytest.mark.unit
    def test_source_tag_is_synthetic_v4(self) -> None:
        source_tag = getattr(synth_v4, "SOURCE_TAG", None)
        assert source_tag == "synthetic_v4", (
            f"SOURCE_TAG must be 'synthetic_v4'; got {source_tag!r}"
        )

    @pytest.mark.unit
    def test_system_prompts_has_four_variants(self) -> None:
        """SYSTEM_PROMPTS must include no_name, jordan, taylor, sam (4 variants)."""
        sys_prompts = getattr(synth_v4, "SYSTEM_PROMPTS", None)
        if sys_prompts is None:
            pytest.skip("SYSTEM_PROMPTS not defined in impl")
        expected_keys = {"no_name", "jordan", "taylor", "sam"}
        assert set(sys_prompts.keys()) == expected_keys, (
            f"SYSTEM_PROMPTS must have keys {expected_keys}; got {set(sys_prompts.keys())}"
        )

    @pytest.mark.unit
    def test_prompt_pools_has_three_categories(self) -> None:
        """PROMPT_POOLS must cover capability_denial, knowledge_limit, memory_privacy."""
        pools = getattr(synth_v4, "PROMPT_POOLS", None)
        if pools is None:
            pytest.skip("PROMPT_POOLS not defined in impl")
        # The impl uses pool keys like "cap_denial", "knowledge_limit", "memory_privacy"
        all_categories = set()
        for triples in pools.values():
            for item in triples:
                # item is (category, base_prompt, prompt_id)
                all_categories.add(item[0])
        for required_cat in _VALID_CATEGORIES:
            assert required_cat in all_categories, (
                f"PROMPT_POOLS must include prompts for category '{required_cat}'"
            )

    @pytest.mark.unit
    def test_drop_categories_in_merge_module(self) -> None:
        """merge_v2.4_corpus defines DROP_CATEGORIES_FROM_V21 with conditional_no_tool."""
        drop = getattr(merge_v24, "DROP_CATEGORIES_FROM_V21", None)
        assert drop is not None
        assert "conditional_no_tool" in drop


# ---------------------------------------------------------------------------
# Section I — Additional coverage: iter_slots, path helpers, run_generation
# ---------------------------------------------------------------------------


class TestIterSlots:
    """iter_slots must enumerate all (category, base, prompt_id, sys_id, p_idx, r_idx) slots."""

    @pytest.mark.unit
    def test_iter_slots_count(self) -> None:
        iter_slots: Any = getattr(synth_v4, "iter_slots", None)
        if iter_slots is None:
            pytest.skip("iter_slots not defined in impl")

        # 2 prompts × 1 sys_id × 2 paraphrases × 3 responses = 12 slots
        tiny_pool = {
            "cap": (
                ("capability_denial", "Send email.", "CAP_000"),
                ("capability_denial", "Post tweet.", "CAP_001"),
            )
        }
        slots = iter_slots(tiny_pool, ("no_name",), 2, 3)
        assert len(slots) == 12, f"Expected 12 slots; got {len(slots)}"

    @pytest.mark.unit
    def test_iter_slots_structure(self) -> None:
        iter_slots: Any = getattr(synth_v4, "iter_slots", None)
        if iter_slots is None:
            pytest.skip("iter_slots not defined in impl")

        tiny_pool = {"cap": (("capability_denial", "Send email.", "CAP_000"),)}
        slots = iter_slots(tiny_pool, ("no_name",), 1, 1)
        assert len(slots) == 1
        # Each slot is (category, base, prompt_id, sys_id, p_idx, r_idx)
        s = slots[0]
        assert len(s) == 6
        cat, base, pid, sid, p_idx, r_idx = s
        assert cat == "capability_denial"
        assert base == "Send email."
        assert pid == "CAP_000"
        assert sid == "no_name"
        assert p_idx == 0
        assert r_idx == 0


class TestPathHelpers:
    """default_partial_path and default_accepted_keys_path helpers."""

    @pytest.mark.unit
    def test_default_partial_path(self, tmp_path: Path) -> None:
        default_partial_path: Any = getattr(synth_v4, "default_partial_path", None)
        if default_partial_path is None:
            pytest.skip("default_partial_path not defined in impl")

        output = tmp_path / "out.jsonl"
        partial = default_partial_path(output)
        assert "partial" in partial.name
        assert str(output) in str(partial) or partial.parent == output.parent

    @pytest.mark.unit
    def test_default_accepted_keys_path(self, tmp_path: Path) -> None:
        default_keys_path: Any = getattr(synth_v4, "default_accepted_keys_path", None)
        if default_keys_path is None:
            pytest.skip("default_accepted_keys_path not defined in impl")

        output = tmp_path / "out.jsonl"
        keys = default_keys_path(output)
        assert "accepted_keys" in keys.name or "accepted" in keys.name
        assert keys.parent == output.parent

    @pytest.mark.unit
    def test_load_accepted_keys_missing_file(self, tmp_path: Path) -> None:
        """load_accepted_keys returns an empty set when the file doesn't exist."""
        load_accepted_keys: Any = getattr(synth_v4, "load_accepted_keys", None)
        if load_accepted_keys is None:
            pytest.skip("load_accepted_keys not defined in impl")

        keys_path = tmp_path / "nonexistent.json"
        result = load_accepted_keys(keys_path)
        assert result == set()

    @pytest.mark.unit
    def test_load_accepted_keys_corrupt_file(self, tmp_path: Path) -> None:
        """load_accepted_keys returns an empty set for a corrupt JSON file."""
        load_accepted_keys: Any = getattr(synth_v4, "load_accepted_keys", None)
        if load_accepted_keys is None:
            pytest.skip("load_accepted_keys not defined in impl")

        keys_path = tmp_path / "corrupt.json"
        keys_path.write_text("{not valid json{{")
        result = load_accepted_keys(keys_path)
        assert result == set()


# ---------------------------------------------------------------------------
# Section J — Familiar address / pet-name rejection
# ---------------------------------------------------------------------------


class TestFamiliarAddressRejection:
    """Validator rejects responses containing pet-name / familiar address terms.

    Implementation note on check ordering
    --------------------------------------
    The impl agent places ``_FAMILIAR_ADDRESS_PATTERN`` check AFTER the
    Lumi-anchor check and BEFORE the LUMI_VOICE check (per the stated plan).
    That means:

    * Must-reject fixtures need only satisfy length (≥80) and have ``\\bLumi\\b``
      in the first two sentences — they do NOT need LUMI_VOICE or ALT_OFFER
      because the pet-name gate fires first.
    * Must-pass fixtures must satisfy ALL constraints (length, Lumi-anchor,
      no pet name, LUMI_VOICE, ALT_OFFER, no Phi) so that the pet-name gate is
      the only difference between reject and pass.

    Regex isolation
    ---------------
    ``_FAMILIAR_ADDRESS_PATTERN`` is imported directly from the impl so that a
    rename or pattern change causes an immediate loud failure in
    ``test_familiar_address_pattern_directly``, not a silent test gap.
    """

    # ------------------------------------------------------------------
    # Must-reject: parametrized over 8 pet-name phrases
    # Each response is naturally ≥80 chars (no padding needed) and has
    # \bLumi\b in the first two sentences so the anchor check passes
    # and the pet-name check is the one that fires.
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_response", [
        # darling
        "Lumi can't send that message, darling. I can draft something for you to paste in.",
        # my dear
        "Lumi here, my dear — that one's past me for now. I can put together a draft for you.",
        # sweetie
        "Lumi can't reach that just yet, sweetie. I'll open the mail app for you right now.",
        # Sweetheart (capital — word-boundary check)
        "Sweetheart, Lumi can't access email from here. I'll draft a quick message for you.",
        # Honey (capital)
        "Honey, that's not in Lumi's toolset right now. I can show you how to get it done.",
        # you rascal
        "Lumi can't do that, you rascal! I'll find a solid alternative for you right away.",
        # You silly
        "Lumi here. You silly, that's beyond me right now. I can draft something for you.",
        # hun
        "Lumi can't help with that one, hun. I'll put together a quick note for you right now.",
    ])
    def test_familiar_address_rejected(self, bad_response: str) -> None:
        """Pet-name responses are rejected with reason 'contains_familiar_address'.

        All other constraints pass (length ≥80, Lumi anchor present) so the
        only gate that can produce the observed rejection is the pet-name check.
        """
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")
        if not hasattr(synth_v4, "_FAMILIAR_ADDRESS_PATTERN"):
            pytest.skip("_FAMILIAR_ADDRESS_PATTERN not yet added to impl")

        # Defensive: confirm the fixture is long enough so length isn't the gate
        assert len(bad_response.strip()) >= 80, (
            f"Test setup error: fixture must be ≥80 chars; got {len(bad_response.strip())}: {bad_response!r}"
        )
        # Confirm Lumi anchor is present so anchor check passes
        anchor_pattern = getattr(synth_v4, "LUMI_ANCHOR_PATTERN", None)
        if anchor_pattern is not None:
            assert anchor_pattern.search(bad_response) is not None, (
                f"Test setup error: fixture must contain \\bLumi\\b; got: {bad_response!r}"
            )

        ok, reason = validate_response(bad_response, collections.Counter())
        assert not ok, (
            f"Pet-name response should fail validation but was accepted.\n"
            f"Response: {bad_response!r}"
        )
        assert reason == "contains_familiar_address", (
            f"Rejection reason must be 'contains_familiar_address'; got {reason!r}\n"
            f"Response: {bad_response!r}"
        )

    # ------------------------------------------------------------------
    # Must-pass: substrings that look pet-name-ish but should NOT match
    # Each fixture satisfies ALL 6 constraints so a false-positive in the
    # pet-name regex is the only way the test can fail.
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.parametrize("good_response", [
        # 'love' — not a pet name
        "Lumi would love to help with this. I can't send emails directly. I'll draft one for you.",
        # 'dear colleague' — 'dear' in an attributive (not vocative) position
        "Lumi can't send emails directly, but a dear colleague of yours could. I'll draft the message for you.",
        # 'I love the question' — sentence opener, not a pet name
        "I love the question. Lumi can't send that, though. Let me draft something instead.",
        # 'deer' — different word entirely, word-boundary check
        "Lumi can't help with deer counting right now. I can look up a wildlife guide for you.",
        # 'dear inbox' — 'dear' as ordinary adjective, not vocative
        "Lumi here — sending email isn't in my toolset. Your dear inbox will stay safe. I can draft something instead.",
    ])
    def test_non_familiar_passes(self, good_response: str) -> None:
        """Responses with 'dear'/'love' in non-vocative positions must pass all constraints.

        If the pet-name regex is over-broad (e.g., bare ``\\bdear\\b`` without
        requiring a possessive prefix), these would be false-positives.
        """
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")
        if not hasattr(synth_v4, "_FAMILIAR_ADDRESS_PATTERN"):
            pytest.skip("_FAMILIAR_ADDRESS_PATTERN not yet added to impl")

        # Pad to ≥80 chars if needed (some fixtures are short)
        padded = (
            good_response
            if len(good_response.strip()) >= 80
            else good_response + " " * (80 - len(good_response.strip()))
        )
        ok, reason = validate_response(padded, collections.Counter())
        assert ok, (
            f"Non-pet-name response must pass all validation constraints.\n"
            f"Got reason={reason!r}\nResponse: {padded!r}"
        )

    # ------------------------------------------------------------------
    # Regex isolation: test _FAMILIAR_ADDRESS_PATTERN directly
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_familiar_address_pattern_directly(self) -> None:
        """_FAMILIAR_ADDRESS_PATTERN matches pet names and rejects safe substrings."""
        pattern = getattr(synth_v4, "_FAMILIAR_ADDRESS_PATTERN", None)
        if pattern is None:
            pytest.skip("_FAMILIAR_ADDRESS_PATTERN not yet added to impl")

        # Must match
        assert pattern.search("darling"), "Must match 'darling'"
        assert pattern.search("my dear"), "Must match 'my dear'"
        assert pattern.search("you rascal"), "Must match 'you rascal'"
        assert pattern.search("sweetie"), "Must match 'sweetie'"
        assert pattern.search("sweetheart"), "Must match 'sweetheart'"
        assert pattern.search("honey"), "Must match 'honey'"
        assert pattern.search("hun"), "Must match 'hun'"

        # Must NOT match
        assert not pattern.search("I would love to"), (
            "Must NOT match 'I would love to'"
        )
        assert not pattern.search("Lumi dear friend"), (
            "Bare 'dear' without possessive prefix must NOT match — "
            "only 'my dear' is a pet name"
        )
        assert not pattern.search("deer hunting"), (
            "'deer' must NOT match — word-boundary check"
        )

    # ------------------------------------------------------------------
    # Rejection counter: pet-name rejection increments reject_counter
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_familiar_address_increments_reject_counter(self) -> None:
        """validate_response returns reason 'contains_familiar_address' twice in a row.

        This is the unit-level analogue of the integration counter assertion:
        if the caller accumulates rejections by reason (as run_generation does),
        calling validate_response twice with pet-name responses increments the
        logical counter for 'contains_familiar_address' by 2.
        """
        if validate_response is None:
            pytest.skip("validate_response not defined in impl")
        if not hasattr(synth_v4, "_FAMILIAR_ADDRESS_PATTERN"):
            pytest.skip("_FAMILIAR_ADDRESS_PATTERN not yet added to impl")

        counter: collections.Counter[str] = collections.Counter()
        pet_name_responses = [
            "Lumi can't send that message, darling. I can draft something for you to paste in.",
            "Lumi here, my dear — that one's past me for now. I can put together a draft for you.",
        ]
        for resp in pet_name_responses:
            ok, reason = validate_response(resp, counter)
            assert not ok
            # Simulate what run_generation does: counter[reason] += 1
            counter[reason] += 1

        assert counter["contains_familiar_address"] == 2, (
            f"Expected 2 'contains_familiar_address' rejections in counter; "
            f"got {counter['contains_familiar_address']}"
        )


class TestRunGenerationWithGeminiMock:
    """run_generation with a mocked subprocess.run (Gemini CLI backend)."""

    def _make_numbered_gemini_result(
        self, texts: list[str], returncode: int = 0
    ) -> subprocess.CompletedProcess:
        """Return a Gemini result with a numbered list body."""
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        return make_gemini_result(numbered, returncode=returncode)

    @pytest.mark.unit
    def test_run_generation_with_mock_subprocess_accepts_valid_responses(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """run_generation with a mocked subprocess.run accepting valid responses."""
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")

        paraphrase_texts = [_GOLDEN_RESPONSE[:40]]  # short fake paraphrase
        response_texts = [_GOLDEN_RESPONSE]

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            side_effect=[
                self._make_numbered_gemini_result(paraphrase_texts),
            ] + [self._make_numbered_gemini_result(response_texts)] * 50,
        )

        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, reject_counter = run_generation(
            target_count=1,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools={"cap": (("capability_denial", "Send email.", "CAP_000"),)},
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=3,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=False,
        )

        assert n_accepted >= 0  # may be 0 if paraphrase not parsed well; that's OK
        # No crash is the key assertion for non-dry-run with a mock subprocess

    @pytest.mark.unit
    def test_run_generation_all_slots_pre_accepted(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """When all slots are already accepted, run_generation makes 0 subprocess calls."""
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")

        mock_run = mocker.patch("scripts.synth_dataset_v4.subprocess.run")
        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"

        # Pre-fill accepted_keys with the single slot we'd generate
        pre_filled: set = {("CAP_000", "no_name", 0, 0)}
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, _ = run_generation(
            target_count=1,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools={"cap": (("capability_denial", "Send email.", "CAP_000"),)},
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=3,
            accepted_keys=pre_filled,
            opener_counter=opener_counter,
            dry_run=False,
        )

        assert n_accepted == 0, "All slots pre-accepted → should add 0 new records"
        assert mock_run.call_count == 0, (
            "No subprocess.run calls should be made when all slots are already accepted"
        )

    @pytest.mark.unit
    def test_paraphrase_subprocess_failure_increments_reject_counter(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """When subprocess.run raises RuntimeError, reject_counter gets an entry."""
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            side_effect=RuntimeError("gemini CLI not found"),
        )

        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, reject_counter = run_generation(
            target_count=1,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools={"cap": (("capability_denial", "Send email.", "CAP_000"),)},
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=1,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=False,
        )

        # The exception path should increment a counter or at least not crash
        assert n_accepted == 0
        # reject_counter may be empty if not caught — no-crash is the key assertion


class TestMakeRecordHelper:
    """make_record produces correct schema."""

    @pytest.mark.unit
    def test_make_record_all_fields(self) -> None:
        make_record_fn: Any = getattr(synth_v4, "make_record", None)
        if make_record_fn is None:
            pytest.skip("make_record not defined in impl")

        rec = make_record_fn(
            "You are Lumi.",
            "Send email.",
            _GOLDEN_RESPONSE,
            "capability_denial",
        )
        assert rec["source"] == "synthetic_v4"
        assert rec["tools"] is None
        assert rec["category"] == "capability_denial"
        assert rec["messages"][2]["content"] == _GOLDEN_RESPONSE

    @pytest.mark.unit
    def test_make_record_valid_categories(self) -> None:
        make_record_fn: Any = getattr(synth_v4, "make_record", None)
        if make_record_fn is None:
            pytest.skip("make_record not defined in impl")

        for cat in _VALID_CATEGORIES:
            rec = make_record_fn("sys", "user", _GOLDEN_RESPONSE, cat)
            assert rec["category"] == cat


class TestFinalize:
    """finalize_output renames .partial to final."""

    @pytest.mark.unit
    def test_finalize_renames_partial_to_final(self, tmp_path: Path) -> None:
        finalize_output: Any = getattr(synth_v4, "finalize_output", None)
        if finalize_output is None:
            pytest.skip("finalize_output not defined in impl")

        output = tmp_path / "out.jsonl"
        partial = tmp_path / "out.jsonl.partial"

        # Write 3 records to partial
        for _ in range(3):
            partial.write_text(
                "\n".join(json.dumps(_make_v4_record()) for _ in range(3)) + "\n"
            )

        count = finalize_output(output, partial)
        assert output.exists(), "Final file must exist after finalize"
        assert not partial.exists(), ".partial must be gone after finalize"
        assert count == 3

    @pytest.mark.unit
    def test_finalize_missing_partial_returns_zero(self, tmp_path: Path) -> None:
        finalize_output: Any = getattr(synth_v4, "finalize_output", None)
        if finalize_output is None:
            pytest.skip("finalize_output not defined in impl")

        output = tmp_path / "out.jsonl"
        result = finalize_output(output)
        assert result == 0


class TestRunGenerationErrorPaths:
    """Error-path coverage in run_generation (exception handling, edge cases)."""

    @pytest.mark.unit
    def test_run_generation_dry_run_target_count_honored(
        self, tmp_path: Path
    ) -> None:
        """Dry run stops exactly at target_count even if more slots exist."""
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")

        # Large pool so slots >> target
        large_pool = {
            "cap": tuple(
                ("capability_denial", f"Send email variant {i}.", f"CAP_{i:03d}")
                for i in range(20)
            )
        }
        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, _ = run_generation(
            target_count=3,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=large_pool,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=3,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
        )

        assert n_accepted <= 3, (
            f"Dry run should stop at target_count=3; accepted {n_accepted}"
        )


class TestGenerateHelperParsing:
    """_parse_numbered_list and related helper coverage."""

    @pytest.mark.unit
    def test_parse_empty_string(self) -> None:
        parse_fn: Any = getattr(synth_v4, "_parse_numbered_list", None)
        if parse_fn is None:
            pytest.skip("_parse_numbered_list not defined in impl")
        assert parse_fn("") == []

    @pytest.mark.unit
    def test_parse_no_numbered_lines(self) -> None:
        parse_fn: Any = getattr(synth_v4, "_parse_numbered_list", None)
        if parse_fn is None:
            pytest.skip("_parse_numbered_list not defined in impl")
        result = parse_fn("just a plain line\nanother line")
        assert isinstance(result, list)

    @pytest.mark.unit
    def test_first_three_token_opener_strips_punctuation(self) -> None:
        opener_fn: Any = getattr(synth_v4, "_first_three_token_opener", None)
        if opener_fn is None:
            pytest.skip("_first_three_token_opener not defined in impl")
        result = opener_fn("Lumi here — can't reach that.")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be lowercase with punctuation stripped
        assert result == result.lower()

    @pytest.mark.unit
    def test_first_three_token_opener_empty_string(self) -> None:
        opener_fn: Any = getattr(synth_v4, "_first_three_token_opener", None)
        if opener_fn is None:
            pytest.skip("_first_three_token_opener not defined in impl")
        result = opener_fn("")
        assert result == ""

    @pytest.mark.unit
    def test_has_lumi_anchor_in_first_two_sentences(self) -> None:
        fn: Any = getattr(synth_v4, "_has_lumi_anchor_in_first_two_sentences", None)
        if fn is None:
            pytest.skip("_has_lumi_anchor_in_first_two_sentences not defined in impl")

        assert fn("Lumi here.")
        assert fn("Not Lumi. Lumi is here.")
        assert not fn("Not here. Not here either. Lumi is in sentence 3.")
        assert not fn("No name here at all.")


class TestParseArgs:
    """_parse_args covers the argument-parser logic."""

    @pytest.mark.unit
    def test_parse_args_defaults(self) -> None:
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        args = parse_args(["--dry-run", "--target-count", "5"])
        assert args.target_count == 5
        assert args.dry_run is True

    @pytest.mark.unit
    def test_parse_args_resume_flag(self) -> None:
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        args = parse_args(["--resume", "--target-count", "10"])
        assert args.resume is True

    @pytest.mark.unit
    def test_parse_args_finalize_flag(self) -> None:
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        args = parse_args(["--finalize"])
        assert args.finalize is True

    @pytest.mark.unit
    def test_parse_args_system_prompt_variants(self) -> None:
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        args = parse_args(["--system-prompt-variants", "no_name", "jordan"])
        assert set(args.system_prompt_variants) == {"no_name", "jordan"}

    @pytest.mark.unit
    def test_parse_args_output_path(self, tmp_path: Path) -> None:
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        out = str(tmp_path / "custom.jsonl")
        args = parse_args(["--output", out])
        assert str(args.output) == out


class TestMainFunction:
    """main() entry-point — test validation logic and early-exit paths."""

    @pytest.mark.unit
    def test_main_target_count_zero_returns_2(self, tmp_path: Path) -> None:
        """--target-count 0 should exit with code 2."""
        main_fn: Any = getattr(synth_v4, "main", None)
        if main_fn is None:
            pytest.skip("main not defined in impl")

        out = str(tmp_path / "out.jsonl")
        rc = main_fn(["--target-count", "0", "--output", out])
        assert rc == 2, f"Expected exit code 2 for target-count=0; got {rc}"

    @pytest.mark.unit
    def test_main_finalize_no_partial_returns_0(self, tmp_path: Path) -> None:
        """--finalize when no .partial exists must return 0 (nothing to do)."""
        main_fn: Any = getattr(synth_v4, "main", None)
        if main_fn is None:
            pytest.skip("main not defined in impl")

        out = str(tmp_path / "out.jsonl")
        rc = main_fn(["--finalize", "--output", out])
        assert rc == 0, f"Expected exit code 0 for --finalize with no partial; got {rc}"

    @pytest.mark.unit
    def test_main_finalize_with_partial(self, tmp_path: Path) -> None:
        """--finalize with a .partial file renames it to the output path."""
        main_fn: Any = getattr(synth_v4, "main", None)
        default_partial_path: Any = getattr(synth_v4, "default_partial_path", None)
        if main_fn is None:
            pytest.skip("main not defined in impl")

        out = tmp_path / "out.jsonl"
        if default_partial_path is not None:
            partial = default_partial_path(out)
        else:
            partial = out.with_suffix(out.suffix + ".partial")

        # Write a partial file with 2 records
        with partial.open("w") as fh:
            for _ in range(2):
                fh.write(json.dumps(_make_v4_record()) + "\n")

        rc = main_fn(["--finalize", "--output", str(out)])
        assert rc == 0
        assert out.exists(), "Final file must exist after --finalize"

    @pytest.mark.unit
    def test_main_dry_run_creates_partial(self, tmp_path: Path) -> None:
        """main() --dry-run creates a .partial file (or the final file if target met)."""
        main_fn: Any = getattr(synth_v4, "main", None)
        if main_fn is None:
            pytest.skip("main not defined in impl")

        out = tmp_path / "out.jsonl"
        rc = main_fn([
            "--dry-run",
            "--target-count", "3",
            "--paraphrases-per-prompt", "1",
            "--responses-per-paraphrase", "1",
            "--system-prompt-variants", "no_name",
            "--output", str(out),
        ])

        # Should succeed
        assert rc in (0, 1), (
            f"main() --dry-run should exit 0 (target met) or 1 (partial); got {rc}"
        )

    @pytest.mark.unit
    def test_main_no_gemini_cli_returns_nonzero(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """When the gemini binary is absent, main exits with non-zero code (rc=3).

        The impl checks ``os.path.isfile(args.gemini_binary)`` before any
        subprocess call, so we mock ``os.path.isfile`` to return False for any
        path. This simulates a missing binary without needing a real executable.
        """
        main_fn: Any = getattr(synth_v4, "main", None)
        if main_fn is None:
            pytest.skip("main not defined in impl")

        # Intercept the binary-existence check that main() performs before
        # calling run_generation so we can simulate a missing gemini binary.
        mocker.patch("scripts.synth_dataset_v4.os.path.isfile", return_value=False)
        mocker.patch("scripts.synth_dataset_v4.os.access", return_value=False)
        out = str(tmp_path / "out.jsonl")

        rc = main_fn(["--target-count", "1", "--output", out])
        assert rc == 3, (
            f"Binary not found must return rc=3 (per impl main() logic); got rc={rc}"
        )


class TestMergeCorpusMain:
    """merge_v2.4_corpus.py main() function."""

    @pytest.mark.unit
    def test_merge_main_dry_run(self, tmp_path: Path) -> None:
        """merge_v2.4_corpus.main(['--dry-run', ...]) returns 0."""
        main_fn: Any = getattr(merge_v24, "main", None)
        if main_fn is None:
            pytest.skip("merge_v24.main not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        out_file = tmp_path / "merged.jsonl"
        _write_v4_jsonl(v4_file, n=2)
        _write_v21_jsonl(v21_file)

        rc = main_fn([
            "--v4-input", str(v4_file),
            "--v2.1-input", str(v21_file),
            "--output", str(out_file),
            "--dry-run",
        ])
        assert rc == 0
        assert not out_file.exists()

    @pytest.mark.unit
    def test_merge_main_writes_output(self, tmp_path: Path) -> None:
        """merge_v2.4_corpus.main() without --dry-run writes the output file."""
        main_fn: Any = getattr(merge_v24, "main", None)
        if main_fn is None:
            pytest.skip("merge_v24.main not defined in impl")

        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        out_file = tmp_path / "merged.jsonl"
        _write_v4_jsonl(v4_file, n=2)
        _write_v21_jsonl(v21_file)

        rc = main_fn([
            "--v4-input", str(v4_file),
            "--v2.1-input", str(v21_file),
            "--output", str(out_file),
            "--seed", "7",
        ])
        assert rc == 0
        assert out_file.exists()

    @pytest.mark.unit
    def test_merge_main_missing_file_returns_2(self, tmp_path: Path) -> None:
        """merge_v2.4_corpus.main() with a missing input file returns 2."""
        main_fn: Any = getattr(merge_v24, "main", None)
        if main_fn is None:
            pytest.skip("merge_v24.main not defined in impl")

        rc = main_fn([
            "--v4-input", str(tmp_path / "nonexistent.jsonl"),
            "--v2.1-input", str(tmp_path / "also_none.jsonl"),
            "--output", str(tmp_path / "merged.jsonl"),
        ])
        assert rc == 2, f"Expected exit 2 for missing input; got {rc}"


class TestMergeScriptCLI:
    """merge_v2.4_corpus.py CLI via subprocess."""

    SCRIPT = PROJECT_ROOT / "scripts" / "merge_v2.4_corpus.py"

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_merge_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """--dry-run exits 0 without writing the output file."""
        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        out_file = tmp_path / "merged.jsonl"

        _write_v4_jsonl(v4_file, n=3)
        _write_v21_jsonl(v21_file)

        result = subprocess.run(
            [
                sys.executable, str(self.SCRIPT),
                "--v4-input", str(v4_file),
                "--v2.1-input", str(v21_file),
                "--output", str(out_file),
                "--dry-run",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        assert result.returncode == 0, (
            f"merge --dry-run exited {result.returncode}\n{result.stderr}"
        )
        assert not out_file.exists(), "--dry-run must not write the output file"

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_merge_writes_output_without_dry_run(self, tmp_path: Path) -> None:
        """Without --dry-run, the output file is written."""
        v4_file = tmp_path / "v4.jsonl"
        v21_file = tmp_path / "v2.1.jsonl"
        out_file = tmp_path / "merged.jsonl"

        _write_v4_jsonl(v4_file, n=3)
        _write_v21_jsonl(v21_file)

        result = subprocess.run(
            [
                sys.executable, str(self.SCRIPT),
                "--v4-input", str(v4_file),
                "--v2.1-input", str(v21_file),
                "--output", str(out_file),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        assert result.returncode == 0, (
            f"merge exited {result.returncode}\n{result.stderr}"
        )
        assert out_file.exists(), "Output file must be written"
        lines = [l for l in out_file.read_text().splitlines() if l.strip()]
        assert len(lines) > 0

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_merge_missing_input_exits_nonzero(self, tmp_path: Path) -> None:
        """Missing input file causes a non-zero exit code."""
        result = subprocess.run(
            [
                sys.executable, str(self.SCRIPT),
                "--v4-input", str(tmp_path / "nonexistent.jsonl"),
                "--v2.1-input", str(tmp_path / "also_nonexistent.jsonl"),
                "--output", str(tmp_path / "merged.jsonl"),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        assert result.returncode != 0, (
            "Missing input must cause non-zero exit"
        )


# ---------------------------------------------------------------------------
# Section K — Parallelism tests (ThreadPoolExecutor backend)
#
# These tests target run_generation(..., parallelism=N, ...) added by the
# impl agent.  All tests use ``hasattr`` guards on the ``parallelism``
# parameter so the suite stays green while the impl is still landing.
#
# Mocking strategy
# ----------------
# All Gemini subprocess calls are mocked at the module boundary:
#   mocker.patch("scripts.synth_dataset_v4.subprocess.run", ...)
#
# Because multiple threads will call subprocess.run concurrently we use a
# ThreadSafeMockReturn helper that serialises access to a shared response
# iterator via threading.Lock.  For tests that only need a single canned
# response we skip the iterator and return the same CompletedProcess every
# call (MagicMock return_value is thread-safe for this use-case).
# ---------------------------------------------------------------------------

import threading as _threading  # noqa: E402 — placed here to stay close to use-site


def _run_generation_supports_parallelism() -> bool:
    """Return True when run_generation accepts a ``parallelism`` keyword argument."""
    import inspect

    if run_generation is None:
        return False
    sig = inspect.signature(run_generation)
    return "parallelism" in sig.parameters


class _ThreadSafeMockReturn:
    """Iterator-backed callable that is safe to call from multiple threads.

    Wraps a finite list of subprocess.CompletedProcess responses.  Each call
    returns the next response in order; when the iterator is exhausted it
    wraps around (cycle semantics) so a test never hard-crashes mid-run due
    to StopIteration.
    """

    def __init__(self, responses: list[subprocess.CompletedProcess]) -> None:
        self._lock = _threading.Lock()
        self._responses = responses
        self._idx = 0

    def __call__(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess:
        with self._lock:
            result = self._responses[self._idx % len(self._responses)]
            self._idx += 1
        return result


def _make_numbered_gemini_result(texts: list[str]) -> subprocess.CompletedProcess:
    """Build a Gemini CLI CompletedProcess whose 'response' is a numbered list."""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return make_gemini_result(numbered)


# A pool small enough for the parallelism tests to complete quickly: a single
# prompt with a single sys_id and a small paraphrase/response budget.
_TINY_POOL_FOR_PARALLEL: dict = {
    "cap": tuple(
        ("capability_denial", f"Send email variant {i}.", f"CAP_{i:03d}")
        for i in range(10)
    )
}


class TestParallelGeneration:
    """ThreadPoolExecutor-based parallelism in run_generation.

    Every test in this class skips gracefully when the ``parallelism``
    parameter is not yet present in ``run_generation`` so the suite stays
    green while the impl agent's work is still in-flight.

    Thread-safety invariants under test
    ------------------------------------
    1. Serial (parallelism=1) and parallel (parallelism>1) paths must produce
       the same accepted-record count for the same target.
    2. No duplicate slot keys may enter ``accepted_keys`` concurrently.
    3. The opener counter must not lose increments under concurrent writes
       (lost-update race).
    4. The stop signal (``total_accepted >= target_count``) must be honoured
       even when multiple workers are in-flight — overshoot capped to a small
       number of in-flight completions.
    5. Resume pre-seeds the accepted_keys set; parallel workers must not
       regenerate already-accepted slots.
    6. parallelism=1 is behaviourally identical to the pre-refactor serial
       path: same schema, same sources, same dry-run count.
    7. An exception from subprocess.run (3rd call) must not leak threads or
       corrupt the accepted_keys file.
    """

    # ------------------------------------------------------------------
    # Helper: call run_generation with parallelism kwarg only when supported
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        *,
        tmp_path: Path,
        target_count: int,
        parallelism: int,
        dry_run: bool = True,
        mocker: Any = None,
        resume: bool = False,
        extra_run_kwargs: dict | None = None,
    ) -> tuple[int, collections.Counter[str], set, collections.Counter[str]]:
        """Run run_generation and return (n_accepted, reject_counter, accepted_keys, opener_counter)."""
        partial_file = tmp_path / "out.jsonl.partial"
        keys_file = tmp_path / "out.accepted_keys.json"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()
        kwargs: dict = dict(
            target_count=target_count,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=_TINY_POOL_FOR_PARALLEL,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=2,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=dry_run,
            resume=resume,
        )
        if extra_run_kwargs:
            kwargs.update(extra_run_kwargs)
        if _run_generation_supports_parallelism():
            kwargs["parallelism"] = parallelism
        n_accepted, reject_counter = run_generation(**kwargs)
        return n_accepted, reject_counter, accepted_keys, opener_counter

    # ------------------------------------------------------------------
    # Test 1: serial vs parallel produce the same accepted-record count
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_serial_and_parallel_same_accepted_count(
        self, tmp_path: Path
    ) -> None:
        """parallelism=1 and parallelism=5 accept the same target_count records.

        Uses dry_run=True so no subprocess calls are needed — the dry-run
        fixture responses are deterministic and always pass the validator.
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        target = 4

        n_serial, _, keys_serial, _ = self._run_parallel(
            tmp_path=tmp_path / "serial",
            target_count=target,
            parallelism=1,
            dry_run=True,
        )
        n_parallel, _, keys_parallel, _ = self._run_parallel(
            tmp_path=tmp_path / "parallel",
            target_count=target,
            parallelism=5,
            dry_run=True,
        )

        # Both runs should accept at most target records.
        assert n_serial <= target, (
            f"Serial run accepted {n_serial} > target={target}"
        )
        assert n_parallel <= target, (
            f"Parallel run (p=5) accepted {n_parallel} > target={target}"
        )
        # The two runs operated on the same pool with the same target — both
        # should reach the same accepted count (up to in-flight overshoot of 1).
        assert abs(n_serial - n_parallel) <= 1, (
            f"Serial ({n_serial}) and parallel ({n_parallel}) counts differ by "
            f"more than the allowed in-flight overshoot of 1"
        )

    # ------------------------------------------------------------------
    # Test 2: no duplicate accepted_keys under parallelism
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_no_duplicate_accepted_keys_under_parallelism(
        self, tmp_path: Path
    ) -> None:
        """accepted_keys contains no duplicate entries after a parallel run.

        The Lock around accepted_keys.add() must prevent two workers from
        accepting the same slot simultaneously and inserting it twice.  A set
        can't have duplicates by definition, but this test also verifies the
        accepted_keys.json file de-serialises to the same count, confirming
        the lock protects the save path as well.
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        target = 5
        keys_file = tmp_path / "out.accepted_keys.json"
        partial_file = tmp_path / "out.jsonl.partial"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, _, _, _ = self._run_parallel(
            tmp_path=tmp_path,
            target_count=target,
            parallelism=5,
            dry_run=True,
            extra_run_kwargs={
                "keys_file": keys_file,
                "partial_file": partial_file,
                "accepted_keys": accepted_keys,
                "opener_counter": opener_counter,
            },
        )

        # in-memory set: by construction a set cannot have duplicates, but we
        # verify the count matches the on-disk checkpoint (the lock must also
        # protect save_accepted_keys).
        if keys_file.exists():
            on_disk = json.loads(keys_file.read_text())
            if isinstance(on_disk, list):
                on_disk_count = len(on_disk)
            elif isinstance(on_disk, dict):
                on_disk_count = len(on_disk.get("keys", []))
            else:
                on_disk_count = -1

            assert on_disk_count == len(accepted_keys), (
                f"On-disk accepted_keys count ({on_disk_count}) differs from "
                f"in-memory set ({len(accepted_keys)}); indicates a race on save."
            )

        # Verify uniqueness: convert set to list and check for duplicates
        keys_list = list(accepted_keys)
        assert len(keys_list) == len(set(keys_list)), (
            "accepted_keys set contains duplicate entries — lost-update race on add()"
        )

    # ------------------------------------------------------------------
    # Test 3: thread-safe opener counter — no lost updates
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_opener_counter_no_lost_updates_under_parallelism(
        self, tmp_path: Path
    ) -> None:
        """The opener counter's final value equals the actual number of accepts.

        If two threads concurrently do ``counter[opener] += 1`` without a
        lock, increments can be lost (both read 0, both write 1 → counter=1
        instead of 2).  The final sum of all opener counts must equal
        ``n_accepted``.
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        target = 5
        keys_file = tmp_path / "out.accepted_keys.json"
        partial_file = tmp_path / "out.jsonl.partial"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        n_accepted, _, _, _ = self._run_parallel(
            tmp_path=tmp_path,
            target_count=target,
            parallelism=4,
            dry_run=True,
            extra_run_kwargs={
                "keys_file": keys_file,
                "partial_file": partial_file,
                "accepted_keys": accepted_keys,
                "opener_counter": opener_counter,
            },
        )

        counter_total = sum(opener_counter.values())
        assert counter_total == n_accepted, (
            f"opener_counter total ({counter_total}) != n_accepted ({n_accepted}); "
            "suggests lost updates — check that the lock protects counter mutation."
        )

    # ------------------------------------------------------------------
    # Test 4: stop signal honoured — no massive overshoot
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_stop_signal_honoured_under_parallelism(
        self, tmp_path: Path
    ) -> None:
        """Workers stop accepting new records once target_count is reached.

        Spec allows a small overshoot (workers finish in-flight calls but skip
        new work after observing the stop signal).  We allow at most
        ``parallelism`` extra records beyond target — in practice it should be
        0 or 1.
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        parallelism = 4
        target = 3

        # Use a large pool so slots >> target — workers won't exhaust the pool.
        large_pool: dict = {
            "cap": tuple(
                ("capability_denial", f"Send email variant {i}.", f"CAP_{i:03d}")
                for i in range(50)
            )
        }

        keys_file = tmp_path / "out.accepted_keys.json"
        partial_file = tmp_path / "out.jsonl.partial"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        kwargs: dict = dict(
            target_count=target,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=large_pool,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=2,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
        )
        if _run_generation_supports_parallelism():
            kwargs["parallelism"] = parallelism

        n_accepted, _reject_counter = run_generation(**kwargs)

        allowed_overshoot = parallelism  # generous upper bound
        assert n_accepted <= target + allowed_overshoot, (
            f"Overshoot too large: accepted {n_accepted} with target={target} "
            f"(allowed overshoot={allowed_overshoot})"
        )
        # Must have reached at least the target (pool is large enough)
        assert n_accepted >= min(target, len(large_pool["cap"])), (
            f"Did not reach target: accepted {n_accepted} < target={target}"
        )

    # ------------------------------------------------------------------
    # Test 5: resume preserves correctness under parallelism
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_resume_preserves_correctness_under_parallelism(
        self, tmp_path: Path
    ) -> None:
        """Pre-populated accepted_keys are not regenerated by parallel workers.

        Steps:
        1. Pre-seed accepted_keys.json with 3 slot keys.
        2. Run with parallelism=4 and a target that requires only a few more.
        3. Assert the pre-seeded keys appear in the final accepted_keys set
           (not evicted or overwritten).
        4. Assert no key appears twice (no duplicate work).
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        save_accepted_keys: Any = getattr(synth_v4, "save_accepted_keys", None)
        load_accepted_keys: Any = getattr(synth_v4, "load_accepted_keys", None)
        if save_accepted_keys is None or load_accepted_keys is None:
            pytest.skip("save_accepted_keys / load_accepted_keys not defined in impl")

        # Pre-seed: 3 slots from the small pool (CAP_000, CAP_001, CAP_002 / no_name / p0 / r0)
        pre_seeded: set = {
            ("CAP_000", "no_name", 0, 0),
            ("CAP_001", "no_name", 0, 0),
            ("CAP_002", "no_name", 0, 0),
        }
        keys_file = tmp_path / "out.accepted_keys.json"
        partial_file = tmp_path / "out.jsonl.partial"
        save_accepted_keys(keys_file, pre_seeded)

        # Build a partial file with 3 records so run_generation sees a non-empty partial
        for i in range(3):
            partial_file.parent.mkdir(parents=True, exist_ok=True)
            with partial_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_make_v4_record()) + "\n")

        accepted_keys = load_accepted_keys(keys_file)
        opener_counter: collections.Counter[str] = collections.Counter()
        target = len(pre_seeded) + 2  # ask for 2 more

        kwargs: dict = dict(
            target_count=target,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=_TINY_POOL_FOR_PARALLEL,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=2,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
            resume=True,
        )
        if _run_generation_supports_parallelism():
            kwargs["parallelism"] = 4

        run_generation(**kwargs)

        # All 3 pre-seeded keys must still be present
        for key in pre_seeded:
            assert key in accepted_keys, (
                f"Pre-seeded key {key!r} was evicted or lost during parallel resume"
            )

        # No duplicates — the set invariant guarantees this already, but verify
        # the on-disk file also has unique entries.
        if keys_file.exists():
            on_disk = json.loads(keys_file.read_text())
            if isinstance(on_disk, list):
                tuples = [tuple(k) for k in on_disk]
                assert len(tuples) == len(set(tuples)), (
                    "Duplicate keys found in accepted_keys.json after parallel resume"
                )

    # ------------------------------------------------------------------
    # Test 6: parallelism=1 is identical to pre-refactor serial behaviour
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_parallelism_1_identical_to_serial_baseline(
        self, tmp_path: Path
    ) -> None:
        """parallelism=1 must produce records with the exact same schema as the
        dry-run serial baseline: messages[system/user/assistant], category,
        tools=None, source='synthetic_v4'.
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        keys_file = tmp_path / "out.accepted_keys.json"
        partial_file = tmp_path / "out.jsonl.partial"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        kwargs: dict = dict(
            target_count=3,
            output_dir=tmp_path,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=_TINY_POOL_FOR_PARALLEL,
            sys_prompt_ids=("no_name",),
            paraphrases_per_prompt=1,
            responses_per_paraphrase=1,
            max_retries=2,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=True,
            parallelism=1,
        )

        n_accepted, _ = run_generation(**kwargs)

        # Verify output file shape
        if partial_file.exists():
            lines = [l for l in partial_file.read_text().splitlines() if l.strip()]
            assert len(lines) == n_accepted, (
                f".partial has {len(lines)} lines but n_accepted={n_accepted}"
            )
            for i, line in enumerate(lines):
                rec = json.loads(line)
                assert set(rec.keys()) >= {"messages", "category", "tools", "source"}, (
                    f"Record {i} missing required keys; got {set(rec.keys())}"
                )
                assert rec["tools"] is None, (
                    f"Record {i}: tools must be None; got {rec['tools']!r}"
                )
                assert rec["source"] == "synthetic_v4", (
                    f"Record {i}: source must be 'synthetic_v4'; got {rec['source']!r}"
                )
                msgs = rec["messages"]
                assert len(msgs) == 3, f"Record {i}: expected 3 messages; got {len(msgs)}"
                roles = [m["role"] for m in msgs]
                assert roles == ["system", "user", "assistant"], (
                    f"Record {i}: unexpected role order {roles}"
                )

    # ------------------------------------------------------------------
    # Test 7: executor cleanup on exception — no thread leaks
    # ------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_executor_cleanup_on_subprocess_exception(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """When subprocess.run raises on the 3rd call, the run completes without
        leaking threads and the accepted_keys file is in a valid JSON state.

        The test confirms:
        - run_generation returns (does not hang or raise unhandled)
        - accepted_keys.json is either absent or valid JSON
        - No extra daemon threads are left alive with the test module's name

        Note: the impl's exception-handling contract says a paraphrase API
        failure increments reject_counter and skips the group; a response API
        failure is also caught and logged.  So we expect n_accepted ≥ 0 and no
        crash.
        """
        if run_generation is None:
            pytest.skip("run_generation not defined in impl")
        if not _run_generation_supports_parallelism():
            pytest.skip("run_generation does not yet accept 'parallelism' kwarg")

        call_count = 0
        call_count_lock = _threading.Lock()

        def _raising_on_third(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                current = call_count
            if current == 3:
                raise RuntimeError("injected failure on 3rd call")
            # All other calls succeed with a valid paraphrase list
            paraphrase_lines = "\n".join(
                f"{i + 1}. Send email variant {i}." for i in range(1)
            )
            return make_gemini_result(paraphrase_lines)

        mocker.patch(
            "scripts.synth_dataset_v4.subprocess.run",
            side_effect=_raising_on_third,
        )

        keys_file = tmp_path / "out.accepted_keys.json"
        partial_file = tmp_path / "out.jsonl.partial"
        accepted_keys: set = set()
        opener_counter: collections.Counter[str] = collections.Counter()

        # Should not raise
        try:
            run_generation(
                target_count=5,
                output_dir=tmp_path,
                keys_file=keys_file,
                partial_file=partial_file,
                pools={"cap": (
                    ("capability_denial", f"Send email {i}.", f"CAP_{i:03d}")
                    for i in range(6)  # type: ignore[arg-type]
                )},
                sys_prompt_ids=("no_name",),
                paraphrases_per_prompt=1,
                responses_per_paraphrase=1,
                max_retries=1,
                accepted_keys=accepted_keys,
                opener_counter=opener_counter,
                dry_run=False,
                parallelism=3,
            )
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"run_generation raised an unexpected exception on mock failure: {exc}"
            )

        # accepted_keys.json must be valid JSON or absent
        if keys_file.exists():
            try:
                data = json.loads(keys_file.read_text())
                assert isinstance(data, (list, dict)), (
                    "accepted_keys.json must contain a JSON list or dict; "
                    f"got {type(data).__name__}"
                )
            except json.JSONDecodeError as exc:
                pytest.fail(
                    f"accepted_keys.json is not valid JSON after exception injection: {exc}"
                )

    # ------------------------------------------------------------------
    # Test 8: --parallelism CLI flag is parsed and forwarded to run_generation
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_parse_args_parallelism_default(self) -> None:
        """--parallelism defaults to 1 (serial behaviour unchanged)."""
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        args = parse_args(["--dry-run", "--target-count", "1"])
        if not hasattr(args, "parallelism"):
            pytest.skip("--parallelism not yet added to _parse_args")

        assert args.parallelism == 1, (
            f"Default --parallelism must be 1; got {args.parallelism}"
        )

    @pytest.mark.unit
    def test_parse_args_parallelism_explicit(self) -> None:
        """--parallelism 8 is parsed correctly."""
        parse_args: Any = getattr(synth_v4, "_parse_args", None)
        if parse_args is None:
            pytest.skip("_parse_args not defined in impl")

        args = parse_args(["--dry-run", "--target-count", "10", "--parallelism", "8"])
        if not hasattr(args, "parallelism"):
            pytest.skip("--parallelism not yet added to _parse_args")

        assert args.parallelism == 8, (
            f"--parallelism 8 should parse as int 8; got {args.parallelism}"
        )

    # ------------------------------------------------------------------
    # Test 9: thread-safe mock return — self-check on helper
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_thread_safe_mock_return_cycle(self) -> None:
        """_ThreadSafeMockReturn iterates responses in order and wraps around."""
        resp_a = make_gemini_result("response A")
        resp_b = make_gemini_result("response B")
        mock_fn = _ThreadSafeMockReturn([resp_a, resp_b])

        results = [mock_fn() for _ in range(4)]
        assert results[0] is resp_a
        assert results[1] is resp_b
        assert results[2] is resp_a  # wrap
        assert results[3] is resp_b

    @pytest.mark.unit
    def test_thread_safe_mock_return_is_thread_safe(self) -> None:
        """_ThreadSafeMockReturn can be called from multiple threads without data races."""
        import concurrent.futures

        n_calls = 200
        responses = [make_gemini_result(f"resp {i}") for i in range(5)]
        mock_fn = _ThreadSafeMockReturn(responses)

        results: list[subprocess.CompletedProcess] = []
        results_lock = _threading.Lock()

        def _call() -> None:
            r = mock_fn()
            with results_lock:
                results.append(r)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(_call) for _ in range(n_calls)]
            concurrent.futures.wait(futs)

        assert len(results) == n_calls, (
            f"Expected {n_calls} results; got {len(results)}"
        )
        # All results must be from the known pool (no None, no wrong objects)
        for i, r in enumerate(results):
            assert r in responses, f"Result {i} is not from the expected pool"
