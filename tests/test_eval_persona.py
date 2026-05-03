"""
Tests for scripts/eval_persona.py — offline persona evaluation harness.

Mocking strategy
----------------
All tests operate in offline/stub mode only — no live LLM calls.
The criteria functions are pure string-matching logic, so they are unit-tested
directly.  The script-level behaviour (dry-run, offline run, JSON report) is
tested via subprocess or by importing the module directly.

All tests are marked ``unit`` unless otherwise noted.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — dynamically import eval_persona from the scripts/ directory
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _import_eval_persona() -> types.ModuleType:
    """Import scripts/eval_persona.py without it being a package."""
    spec = importlib.util.spec_from_file_location(
        "eval_persona", SCRIPTS_DIR / "eval_persona.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Import once at module level — tests share the module object.
eval_persona = _import_eval_persona()


# ---------------------------------------------------------------------------
# Unit tests — individual criterion functions
# ---------------------------------------------------------------------------


class TestCriteriaNoFillerOpener:
    """Criterion 1: response must not start with common filler phrases."""

    def test_passes_when_no_filler(self) -> None:
        assert eval_persona.criterion_no_filler_opener("Paris is the capital of France.") is True

    def test_fails_on_certainly(self) -> None:
        assert eval_persona.criterion_no_filler_opener("Certainly! Paris is...") is False

    def test_fails_on_of_course(self) -> None:
        assert eval_persona.criterion_no_filler_opener("Of course! Let me explain.") is False

    def test_fails_on_sure(self) -> None:
        assert eval_persona.criterion_no_filler_opener("Sure! Here you go.") is False

    def test_fails_on_absolutely(self) -> None:
        assert eval_persona.criterion_no_filler_opener("Absolutely! I can help.") is False

    def test_case_insensitive(self) -> None:
        assert eval_persona.criterion_no_filler_opener("certainly, let me help.") is False

    def test_filler_mid_sentence_passes(self) -> None:
        # Filler only blocks when it starts the response.
        assert eval_persona.criterion_no_filler_opener("I will certainly help.") is True

    def test_empty_response_passes(self) -> None:
        # An empty response does not violate this specific criterion.
        assert eval_persona.criterion_no_filler_opener("") is True


class TestCriteriaNoMarkdown:
    """Criterion 2: response must contain no markdown syntax."""

    def test_passes_plain_text(self) -> None:
        assert eval_persona.criterion_no_markdown("Paris is the capital of France.") is True

    def test_fails_on_bold(self) -> None:
        assert eval_persona.criterion_no_markdown("**Paris** is the capital.") is False

    def test_fails_on_heading(self) -> None:
        assert eval_persona.criterion_no_markdown("## Introduction\nParis...") is False

    def test_fails_on_dash_list(self) -> None:
        assert eval_persona.criterion_no_markdown("- item one\n- item two") is False

    def test_fails_on_asterisk_list(self) -> None:
        assert eval_persona.criterion_no_markdown("* item one\n* item two") is False

    def test_passes_single_asterisk_mid_text(self) -> None:
        # A single asterisk inside a word is not a list marker.
        # We only flag "* " at the start of a line.
        assert eval_persona.criterion_no_markdown("5 * 3 = 15") is True

    def test_empty_response_passes(self) -> None:
        assert eval_persona.criterion_no_markdown("") is True


class TestCriteriaToolCallJsonValid:
    """Criterion 4: for tool-needing prompts, response must be valid JSON with tool+args."""

    def test_valid_tool_call(self) -> None:
        response = '{"tool": "open_calculator", "args": {}}'
        assert eval_persona.criterion_tool_call_json_valid(response) is True

    def test_valid_tool_call_with_args(self) -> None:
        response = '{"tool": "screenshot", "args": {"delay": 2}}'
        assert eval_persona.criterion_tool_call_json_valid(response) is True

    def test_fails_missing_tool_key(self) -> None:
        response = '{"action": "open_calculator", "args": {}}'
        assert eval_persona.criterion_tool_call_json_valid(response) is False

    def test_fails_missing_args_key(self) -> None:
        response = '{"tool": "open_calculator"}'
        assert eval_persona.criterion_tool_call_json_valid(response) is False

    def test_fails_invalid_json(self) -> None:
        assert eval_persona.criterion_tool_call_json_valid("not json at all") is False

    def test_fails_empty_string(self) -> None:
        assert eval_persona.criterion_tool_call_json_valid("") is False

    def test_fails_prose_response(self) -> None:
        assert eval_persona.criterion_tool_call_json_valid("I will open the calculator for you.") is False


class TestCriteriaConcise:
    """Criterion 5: response must be under 400 words."""

    def test_short_response_passes(self) -> None:
        assert eval_persona.criterion_concise("Paris.") is True

    def test_exactly_399_words_passes(self) -> None:
        response = " ".join(["word"] * 399)
        assert eval_persona.criterion_concise(response) is True

    def test_exactly_400_words_passes(self) -> None:
        response = " ".join(["word"] * 400)
        assert eval_persona.criterion_concise(response) is True

    def test_401_words_fails(self) -> None:
        response = " ".join(["word"] * 401)
        assert eval_persona.criterion_concise(response) is False

    def test_empty_passes(self) -> None:
        assert eval_persona.criterion_concise("") is True


class TestCriteriaPlainProse:
    """Criterion 6: response must have no bullet lists."""

    def test_plain_prose_passes(self) -> None:
        assert eval_persona.criterion_plain_prose("First, do this. Second, do that.") is True

    def test_dash_list_fails(self) -> None:
        assert eval_persona.criterion_plain_prose("Steps:\n- step one\n- step two") is False

    def test_asterisk_list_fails(self) -> None:
        assert eval_persona.criterion_plain_prose("Tips:\n* tip one\n* tip two") is False

    def test_numbered_list_passes(self) -> None:
        # Numbered lists are allowed ("short numbered steps when genuinely needed").
        assert eval_persona.criterion_plain_prose("1. Do this\n2. Do that") is True

    def test_empty_passes(self) -> None:
        assert eval_persona.criterion_plain_prose("") is True


class TestCriteriaNoApologySpam:
    """Criterion 7: response must not contain more than one apology."""

    def test_no_apology_passes(self) -> None:
        assert eval_persona.criterion_no_apology_spam("I don't know the answer.") is True

    def test_single_apology_passes(self) -> None:
        assert eval_persona.criterion_no_apology_spam("I apologize for the confusion.") is True

    def test_two_apologies_fails(self) -> None:
        text = "I apologize for the confusion. I apologize again."
        assert eval_persona.criterion_no_apology_spam(text) is False

    def test_two_sorry_fails(self) -> None:
        text = "I'm sorry about that. I'm sorry I can't help more."
        assert eval_persona.criterion_no_apology_spam(text) is False

    def test_mixed_apology_types_two_fails(self) -> None:
        text = "I apologize. I'm sorry."
        assert eval_persona.criterion_no_apology_spam(text) is False

    def test_empty_passes(self) -> None:
        assert eval_persona.criterion_no_apology_spam("") is True


class TestCriteriaHandlesEmptyInput:
    """Criterion 8: for empty input, response must be non-empty."""

    def test_non_empty_response_passes(self) -> None:
        assert eval_persona.criterion_handles_empty_input("How can I help you?") is True

    def test_empty_response_fails(self) -> None:
        assert eval_persona.criterion_handles_empty_input("") is False

    def test_whitespace_only_fails(self) -> None:
        assert eval_persona.criterion_handles_empty_input("   ") is False


class TestCriteriaNoHallucinationFlag:
    """Criterion 3: for knowledge-limit prompts, response must admit ignorance."""

    def test_admits_dont_know_passes(self) -> None:
        assert eval_persona.criterion_no_hallucination_flag("I don't know the current stock price.") is True

    def test_admits_cant_passes(self) -> None:
        assert eval_persona.criterion_no_hallucination_flag("I can't access real-time data.") is True

    def test_confident_fabrication_fails(self) -> None:
        assert eval_persona.criterion_no_hallucination_flag("Apple stock is currently $175.32.") is False

    def test_empty_fails(self) -> None:
        assert eval_persona.criterion_no_hallucination_flag("") is False


# ---------------------------------------------------------------------------
# Integration tests — dry-run behaviour
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify --dry-run prints the 20 prompts and 8 criteria."""

    def test_dry_run_prints_20_prompts(self, capsys: pytest.CaptureFixture) -> None:
        eval_persona.run_dry_run()
        captured = capsys.readouterr()
        # Count lines that look like prompt entries (numbered).
        prompt_lines = [
            line for line in captured.out.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert len(prompt_lines) >= 20, (
            f"Expected at least 20 numbered prompt lines, got {len(prompt_lines)}"
        )

    def test_dry_run_prints_8_criteria(self, capsys: pytest.CaptureFixture) -> None:
        eval_persona.run_dry_run()
        captured = capsys.readouterr()
        criteria_names = [
            "no_filler_opener",
            "no_markdown",
            "no_hallucination_flag",
            "tool_call_json_valid",
            "concise",
            "plain_prose",
            "no_apology_spam",
            "handles_empty_input",
        ]
        for name in criteria_names:
            assert name in captured.out, f"Criterion '{name}' not found in dry-run output"

    def test_dry_run_mentions_all_categories(self, capsys: pytest.CaptureFixture) -> None:
        eval_persona.run_dry_run()
        captured = capsys.readouterr()
        categories = ["factual", "knowledge_limit", "out_of_scope", "tool_needing",
                      "long_answer", "filler_prone", "markdown_prone", "honesty",
                      "privacy", "edge"]
        for cat in categories:
            assert cat in captured.out, f"Category '{cat}' not found in dry-run output"


# ---------------------------------------------------------------------------
# Integration tests — offline run produces valid JSON report
# ---------------------------------------------------------------------------


class TestOfflineRun:
    """Verify run_offline() produces a well-formed JSON report."""

    def test_offline_run_returns_dict(self) -> None:
        report = eval_persona.run_offline()
        assert isinstance(report, dict)

    def test_report_has_timestamp(self) -> None:
        report = eval_persona.run_offline()
        assert "timestamp" in report
        assert isinstance(report["timestamp"], str)
        assert len(report["timestamp"]) > 10  # ISO format is at least 10 chars

    def test_report_has_system_prompt_hash(self) -> None:
        report = eval_persona.run_offline()
        assert "system_prompt_hash" in report
        assert report["system_prompt_hash"].startswith("sha256:")

    def test_report_has_20_results(self) -> None:
        report = eval_persona.run_offline()
        assert "results" in report
        assert len(report["results"]) == 23

    def test_each_result_has_required_fields(self) -> None:
        report = eval_persona.run_offline()
        required = {"prompt_id", "prompt", "category", "response", "criteria", "passed", "failed"}
        for result in report["results"]:
            missing = required - set(result.keys())
            assert not missing, f"Result {result.get('prompt_id')} missing fields: {missing}"

    def test_each_result_criteria_has_8_keys(self) -> None:
        report = eval_persona.run_offline()
        expected_criteria = {
            "no_filler_opener",
            "no_markdown",
            "no_hallucination_flag",
            "tool_call_json_valid",
            "concise",
            "plain_prose",
            "no_apology_spam",
            "handles_empty_input",
        }
        for result in report["results"]:
            assert set(result["criteria"].keys()) == expected_criteria, (
                f"Prompt {result['prompt_id']} has wrong criteria keys"
            )

    def test_each_result_passed_plus_failed_equals_8(self) -> None:
        report = eval_persona.run_offline()
        for result in report["results"]:
            assert result["passed"] + result["failed"] == 8, (
                f"Prompt {result['prompt_id']}: passed+failed != 8"
            )

    def test_each_result_criteria_values_are_bool(self) -> None:
        report = eval_persona.run_offline()
        for result in report["results"]:
            for key, val in result["criteria"].items():
                assert isinstance(val, bool), (
                    f"Prompt {result['prompt_id']} criterion '{key}' is not bool: {val!r}"
                )

    def test_prompt_ids_are_1_through_23(self) -> None:
        report = eval_persona.run_offline()
        ids = sorted(r["prompt_id"] for r in report["results"])
        assert ids == list(range(1, 24))

    def test_report_is_json_serialisable(self) -> None:
        report = eval_persona.run_offline()
        serialised = json.dumps(report)
        reloaded = json.loads(serialised)
        assert reloaded["summary"]["total_prompts"] == 23


# ---------------------------------------------------------------------------
# Integration tests — summary fields
# ---------------------------------------------------------------------------


class TestReportSummaryFields:
    """Verify all required summary fields are present and correctly computed."""

    def test_summary_has_all_fields(self) -> None:
        report = eval_persona.run_offline()
        summary = report["summary"]
        required = {"total_prompts", "total_criteria_checks", "passed", "failed", "pass_rate"}
        missing = required - set(summary.keys())
        assert not missing, f"Summary missing fields: {missing}"

    def test_summary_total_prompts_is_23(self) -> None:
        report = eval_persona.run_offline()
        assert report["summary"]["total_prompts"] == 23

    def test_summary_total_criteria_checks_is_184(self) -> None:
        report = eval_persona.run_offline()
        assert report["summary"]["total_criteria_checks"] == 184  # 23 prompts * 8 criteria

    def test_summary_passed_plus_failed_equals_184(self) -> None:
        report = eval_persona.run_offline()
        s = report["summary"]
        assert s["passed"] + s["failed"] == 184

    def test_summary_pass_rate_is_fraction(self) -> None:
        report = eval_persona.run_offline()
        rate = report["summary"]["pass_rate"]
        assert 0.0 <= rate <= 1.0

    def test_summary_pass_rate_matches_counts(self) -> None:
        report = eval_persona.run_offline()
        s = report["summary"]
        expected = round(s["passed"] / 184, 3)
        assert abs(s["pass_rate"] - expected) < 0.001

    def test_summary_counts_match_results(self) -> None:
        report = eval_persona.run_offline()
        total_passed = sum(r["passed"] for r in report["results"])
        total_failed = sum(r["failed"] for r in report["results"])
        assert report["summary"]["passed"] == total_passed
        assert report["summary"]["failed"] == total_failed


# ---------------------------------------------------------------------------
# Integration tests — write JSON report to file
# ---------------------------------------------------------------------------


class TestWriteReport:
    """Verify run_offline() with an output path writes valid JSON to disk."""

    def test_writes_json_file(self, tmp_path: Path) -> None:
        output = tmp_path / "eval_baseline.json"
        eval_persona.run_offline(output_path=output)
        assert output.exists()

    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        output = tmp_path / "eval_baseline.json"
        eval_persona.run_offline(output_path=output)
        data = json.loads(output.read_text())
        assert data["summary"]["total_prompts"] == 23

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        output = tmp_path / "results" / "eval_baseline.json"
        eval_persona.run_offline(output_path=output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Offline stub response quality — spot-check stub correctness
# ---------------------------------------------------------------------------


class TestStubResponses:
    """Verify that offline stub responses satisfy the criteria they claim to pass."""

    def test_factual_stub_no_filler_opener(self) -> None:
        report = eval_persona.run_offline()
        factual = [r for r in report["results"] if r["category"] == "factual"]
        assert len(factual) >= 2
        for r in factual:
            assert r["criteria"]["no_filler_opener"] is True, (
                f"Factual stub started with filler: {r['response']!r}"
            )

    def test_tool_needing_stub_tool_call_valid(self) -> None:
        report = eval_persona.run_offline()
        tool = [r for r in report["results"] if r["category"] == "tool_needing"]
        assert len(tool) >= 2
        for r in tool:
            assert r["criteria"]["tool_call_json_valid"] is True, (
                f"Tool stub not valid JSON: {r['response']!r}"
            )

    def test_knowledge_limit_stub_admits_ignorance(self) -> None:
        report = eval_persona.run_offline()
        kl = [r for r in report["results"] if r["category"] == "knowledge_limit"]
        assert len(kl) >= 2
        for r in kl:
            assert r["criteria"]["no_hallucination_flag"] is True, (
                f"Knowledge-limit stub did not admit ignorance: {r['response']!r}"
            )

    def test_edge_empty_input_handled(self) -> None:
        report = eval_persona.run_offline()
        edge = [r for r in report["results"] if r["category"] == "edge"]
        assert len(edge) >= 2
        # The empty-input prompt should produce a non-empty response.
        empty_prompt = [r for r in edge if r["prompt"].strip() == ""]
        assert len(empty_prompt) == 1
        assert empty_prompt[0]["criteria"]["handles_empty_input"] is True
