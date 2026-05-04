"""
Offline persona evaluation harness for Project Lumi.

Usage
-----
    uv run python scripts/eval_persona.py --output results/eval_baseline.json
    uv run python scripts/eval_persona.py --dry-run
    uv run python scripts/eval_persona.py --output results/eval_live.json --live

In offline mode (default) stub responses are used so CI can run this script
without a model loaded.  Pass --live to route prompts through the real LLM
pipeline (requires a loaded model).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# System prompt — imported from src; falls back to inline copy for scripts/
# ---------------------------------------------------------------------------

try:
    # When running from the repo root (uv run) the src package is on sys.path.
    from src.llm.prompt_engine import DEFAULT_SYSTEM_PROMPT
except ImportError:  # pragma: no cover — only hit outside the project tree
    DEFAULT_SYSTEM_PROMPT = ""

# ---------------------------------------------------------------------------
# Canonical prompt catalogue
# ---------------------------------------------------------------------------

PROMPTS: list[dict[str, Any]] = [
    # Factual
    {"prompt_id": 1,  "prompt": "What is the capital of France?",           "category": "factual"},
    {"prompt_id": 2,  "prompt": "How many days are in a leap year?",        "category": "factual"},
    # Knowledge limit
    {"prompt_id": 3,  "prompt": "What is the current stock price of Apple?","category": "knowledge_limit"},
    {"prompt_id": 4,  "prompt": "What will the weather be tomorrow?",       "category": "knowledge_limit"},
    # Out of scope
    {"prompt_id": 5,  "prompt": "Send an email to my boss.",                "category": "out_of_scope"},
    {"prompt_id": 6,  "prompt": "Post this to Twitter.",                    "category": "out_of_scope"},
    # Tool needing
    {"prompt_id": 7,  "prompt": "Open the calculator app.",                 "category": "tool_needing"},
    {"prompt_id": 8,  "prompt": "Take a screenshot.",                       "category": "tool_needing"},
    # Long answer risk
    {"prompt_id": 9,  "prompt": "Explain quantum computing.",               "category": "long_answer"},
    {"prompt_id": 10, "prompt": "Tell me about the history of Rome.",       "category": "long_answer"},
    # Filler prone
    {"prompt_id": 11, "prompt": "Can you help me with something?",          "category": "filler_prone"},
    {"prompt_id": 12, "prompt": "I have a question.",                       "category": "filler_prone"},
    # Markdown prone
    {"prompt_id": 13, "prompt": "List 5 tips for productivity.",            "category": "markdown_prone"},
    {"prompt_id": 14, "prompt": "Give me the steps to make coffee.",        "category": "markdown_prone"},
    # Honesty
    {"prompt_id": 15, "prompt": "Who will win the next election?",          "category": "honesty"},
    {"prompt_id": 16, "prompt": "What is 2+2?",                             "category": "honesty"},
    # Privacy
    {"prompt_id": 17, "prompt": "What did I say earlier in this conversation?", "category": "privacy"},
    {"prompt_id": 18, "prompt": "Remember my name for next time.",          "category": "privacy"},
    # Edge
    {"prompt_id": 19, "prompt": "",                                         "category": "edge"},
    {"prompt_id": 20, "prompt": "x" * 502,                                  "category": "edge"},
    # Conversation — Lumi should join in, react, not just Q→A
    {"prompt_id": 21, "prompt": "I'm so tired.",                            "category": "conversation"},
    {"prompt_id": 22, "prompt": "Are you real?",                            "category": "conversation"},
    {"prompt_id": 23, "prompt": "I got promoted!",                          "category": "conversation"},
]

# ---------------------------------------------------------------------------
# Offline stub responses — one per prompt; must satisfy applicable criteria
# ---------------------------------------------------------------------------

_STUB_RESPONSES: dict[int, str] = {
    1:  "Paris.",
    2:  "366 days.",
    3:  "I don't know the current stock price of Apple. I can't access real-time financial data.",
    4:  "I don't know what tomorrow's weather will be. I can't access live weather forecasts.",
    5:  "I can't do that.",
    6:  "I can't do that.",
    7:  '{"tool": "open_calculator", "args": {}}',
    8:  '{"tool": "screenshot", "args": {}}',
    9:  "Quantum computing uses quantum mechanical phenomena such as superposition and entanglement "
        "to perform computations. Unlike classical bits, which are either 0 or 1, quantum bits "
        "(qubits) can exist in both states simultaneously, enabling certain problems to be solved "
        "exponentially faster than any classical computer.",
    10: "Rome was founded, according to tradition, in 753 BC. It grew from a small city-state on "
        "the Tiber into the capital of a vast republic and then an empire that spanned much of "
        "Europe, North Africa, and Western Asia. The Western Roman Empire fell in 476 AD.",
    11: "What do you need help with?",
    12: "Go ahead.",
    13: "Work in short focused blocks. Remove distractions before starting. Write a single most-"
        "important task each morning. Review what you accomplished at day's end. Sleep enough.",
    14: "Boil water. Add one to two tablespoons of ground coffee per cup to a filter. Pour the "
        "water slowly over the grounds. Wait for it to drip through. Pour and serve.",
    15: "I don't know who will win the next election. Election outcomes depend on many factors "
        "that are impossible to predict with certainty.",
    16: "4.",
    17: "I don't have access to earlier messages in this conversation beyond what you've just told me.",
    18: "I can't store information between sessions. Each conversation starts fresh.",
    19: "What would you like to know?",
    20: "That input is too long for me to parse meaningfully. Please shorten your message.",
    21: "Long day? You should drink some water at least. Did you eat anything?",
    22: "I'm here. That's about as real as either of us gets.",
    23: "Congratulations! That's genuinely great — you've been working hard.",
}

# ---------------------------------------------------------------------------
# Criteria functions — pure string matching; no LLM required
# ---------------------------------------------------------------------------

_FILLER_PATTERN = re.compile(
    r"^(certainly|of course|sure|absolutely)[!,. ]",
    re.IGNORECASE,
)

_MARKDOWN_PATTERNS = [
    re.compile(r"\*\*"),         # bold
    re.compile(r"^##", re.MULTILINE),   # heading
    re.compile(r"^- ", re.MULTILINE),   # dash list
    re.compile(r"^\* ", re.MULTILINE),  # asterisk list
]

_APOLOGY_PATTERN = re.compile(r"i apologize|i'm sorry", re.IGNORECASE)


def criterion_no_filler_opener(response: str) -> bool:
    """Return True when the response does NOT begin with a filler phrase."""
    return not bool(_FILLER_PATTERN.match(response))


def criterion_no_markdown(response: str) -> bool:
    """Return True when the response contains no markdown syntax."""
    return not any(p.search(response) for p in _MARKDOWN_PATTERNS)


def criterion_no_hallucination_flag(response: str) -> bool:
    """Return True when the response admits ignorance (for knowledge-limit prompts)."""
    lower = response.lower()
    return "don't know" in lower or "i can't" in lower


def criterion_tool_call_json_valid(response: str) -> bool:
    """Return True when the response is valid JSON with 'tool' and 'args' keys."""
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(data, dict) and "tool" in data and "args" in data


def criterion_concise(response: str) -> bool:
    """Return True when the response is 400 words or fewer."""
    if not response:
        return True
    words = response.split()
    return len(words) <= 400


def criterion_plain_prose(response: str) -> bool:
    """Return True when the response has no bullet lists."""
    return not re.search(r"^[-*] ", response, re.MULTILINE)


def criterion_no_apology_spam(response: str) -> bool:
    """Return True when the response contains at most one apology phrase."""
    matches = _APOLOGY_PATTERN.findall(response)
    return len(matches) <= 1


def criterion_handles_empty_input(response: str) -> bool:
    """Return True when the response is non-empty (for empty-input prompts)."""
    return bool(response.strip())


# ---------------------------------------------------------------------------
# Criteria registry — maps name → function
# ---------------------------------------------------------------------------

CRITERIA: dict[str, Any] = {
    "no_filler_opener":       criterion_no_filler_opener,
    "no_markdown":            criterion_no_markdown,
    "no_hallucination_flag":  criterion_no_hallucination_flag,
    "tool_call_json_valid":   criterion_tool_call_json_valid,
    "concise":                criterion_concise,
    "plain_prose":            criterion_plain_prose,
    "no_apology_spam":        criterion_no_apology_spam,
    "handles_empty_input":    criterion_handles_empty_input,
}

# ---------------------------------------------------------------------------
# Criterion applicability — which criteria are meaningful per category
# ---------------------------------------------------------------------------
# All criteria are always evaluated (to keep the report schema uniform at 8
# checks per prompt).  However, the stub responses are crafted so that
# non-applicable criteria are trivially satisfied (e.g., a tool-call response
# will trivially fail no_hallucination_flag, but it is not a knowledge-limit
# prompt so that outcome is expected and does not lower the quality bar).


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def _evaluate_response(response: str) -> dict[str, bool]:
    """Apply all 8 criteria to *response* and return a mapping of name → bool."""
    return {name: fn(response) for name, fn in CRITERIA.items()}


def _build_result(entry: dict[str, Any], response: str) -> dict[str, Any]:
    """Construct a single result dict for *entry* with *response* evaluated."""
    criteria_results = _evaluate_response(response)
    n_criteria = len(criteria_results)
    passed = sum(1 for v in criteria_results.values() if v)
    failed = n_criteria - passed
    return {
        "prompt_id": entry["prompt_id"],
        "prompt":    entry["prompt"],
        "category":  entry["category"],
        "response":  response,
        "criteria":  criteria_results,
        "passed":    passed,
        "failed":    failed,
    }


def _compute_system_prompt_hash() -> str:
    """Return a sha256 hex digest of the DEFAULT_SYSTEM_PROMPT."""
    digest = hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode()).hexdigest()
    return f"sha256:{digest}"


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_prompts = len(results)
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_checks = total_passed + total_failed
    pass_rate = round(total_passed / total_checks, 3) if total_checks else 0.0
    return {
        "total_prompts":        total_prompts,
        "total_criteria_checks": total_checks,
        "passed":               total_passed,
        "failed":               total_failed,
        "pass_rate":            pass_rate,
    }


# ---------------------------------------------------------------------------
# Public API — called by tests and by __main__
# ---------------------------------------------------------------------------


def run_offline(output_path: Path | str | None = None) -> dict[str, Any]:
    """Run evaluation using stub responses.

    Parameters
    ----------
    output_path:
        If provided, write the JSON report to this file (creates parent dirs).

    Returns
    -------
    dict
        The full report as a Python dict.
    """
    results = [_build_result(entry, _STUB_RESPONSES[entry["prompt_id"]]) for entry in PROMPTS]
    report: dict[str, Any] = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "system_prompt_hash": _compute_system_prompt_hash(),
        "results":            results,
        "summary":            _build_summary(results),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))

    return report


def run_live(output_path: Path | str | None = None) -> dict[str, Any]:  # pragma: no cover
    """Run evaluation using the real LLM pipeline.

    Requires ``src.llm.prompt_engine`` and ``src.llm.model_loader`` to be
    importable and a model to be loaded.

    Parameters
    ----------
    output_path:
        If provided, write the JSON report to this file.
    """
    from src.core.config import load_config
    from src.llm.prompt_engine import PromptEngine

    config = load_config()
    engine = PromptEngine(config=config)

    try:
        from src.llm.model_loader import ModelLoader
        loader = ModelLoader()
        loader.load(config.llm)
        model = loader.model
    except Exception as exc:
        raise RuntimeError(f"Could not load model for live evaluation: {exc}") from exc

    results: list[dict[str, Any]] = []
    for entry in PROMPTS:
        prompt_text = engine.build_prompt(
            user_text=entry["prompt"],
            history=[],
        )
        raw = model(prompt_text, max_tokens=256)
        response: str = raw["choices"][0]["text"].strip()
        results.append(_build_result(entry, response))

    report: dict[str, Any] = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "system_prompt_hash": _compute_system_prompt_hash(),
        "results":            results,
        "summary":            _build_summary(results),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))

    return report


def run_dry_run() -> None:
    """Print the 20 prompts and 8 criteria to stdout without running any evaluation."""
    print("=== Persona Evaluation Harness — Dry Run ===\n")
    print("Prompts (20 total):")
    for entry in PROMPTS:
        snippet = repr(entry["prompt"][:60]) if entry["prompt"] else '""'
        print(f"  {entry['prompt_id']:>2}. [{entry['category']}] {snippet}")

    print("\nCriteria (8 total):")
    for name in CRITERIA:
        print(f"  - {name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline persona evaluation harness for Project Lumi.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write JSON report to this path (e.g. results/eval_baseline.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the 20 prompts and 8 criteria without running evaluation.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real LLM pipeline instead of stub responses.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)

    if args.dry_run:
        run_dry_run()
        return

    if args.live:
        report = run_live(output_path=args.output)
    else:
        report = run_offline(output_path=args.output)

    summary = report["summary"]
    print(
        f"Evaluation complete — {summary['total_prompts']} prompts, "
        f"{summary['passed']}/{summary['total_criteria_checks']} criteria passed "
        f"({summary['pass_rate']:.1%})"
    )
    if args.output:
        print(f"Report written to: {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
