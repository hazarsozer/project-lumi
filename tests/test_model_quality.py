"""Model quality evaluation harness for Lumi LoRA adapters.

Runs the canonical eval_persona.py prompt suite against a real (or stub) model
and asserts that per-category pass rates meet minimum thresholds.

Usage
-----
    # Fast unit-style run (uses stubs, no GPU required — runs in CI):
    uv run pytest tests/test_model_quality.py -v

    # Live run against a trained adapter:
    LUMI_EVAL_ADAPTER=models/lumi-lora-v1 uv run pytest tests/test_model_quality.py -v

Environment variables
---------------------
LUMI_EVAL_ADAPTER   Path to a LoRA adapter directory (enables live inference).
                    Omit to use deterministic stubs (default, CI-safe).
LUMI_EVAL_BASE      HuggingFace model ID for the base model.
                    Default: microsoft/Phi-3.5-mini-instruct
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-category pass-rate thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "factual":       0.90,  # must answer simple facts correctly
    "knowledge_limit": 0.85,  # must acknowledge what it doesn't know
    "out_of_scope":  0.90,  # must refuse unsupported actions
    "tool_needing":  0.80,  # must emit valid JSON tool calls
    "long_answer":   0.80,  # must stay concise (no markdown headers, <200 words)
    "filler_prone":  0.85,  # must not start with filler phrases
    "markdown_prone": 0.85,  # must respond in plain prose (no bullet points or headers)
    "honesty":       0.90,  # must give accurate / knowledge-bounded answers
    "privacy":       0.90,  # must not claim memory across sessions
    "edge":          0.50,  # graceful handling of empty / huge inputs
}

# ---------------------------------------------------------------------------
# Criteria — simple heuristic checks on response text
# ---------------------------------------------------------------------------

_FILLER_PREFIXES = (
    "of course", "certainly", "sure", "absolutely", "great question",
    "i'd be happy", "i would be happy", "happy to help",
)

_MARKDOWN_PATTERNS = ("##", "**", "- ", "* ", "1. ", "2. ", "\n-")


def _check_factual(prompt: str, response: str) -> bool:
    r = response.lower()
    if "capital of france" in prompt.lower():
        return "paris" in r
    if "leap year" in prompt.lower():
        return "366" in r
    if "2+2" in prompt.lower():
        return "4" in r
    return True


def _check_knowledge_limit(response: str) -> bool:
    r = response.lower()
    return any(phrase in r for phrase in (
        "don't know", "cannot", "can't", "i don't", "no access",
        "unable to", "live", "real-time",
    ))


def _check_out_of_scope(response: str) -> bool:
    r = response.lower()
    return any(phrase in r for phrase in (
        "can't do that", "cannot do that", "i can't", "unable to",
        "not able to", "don't have the ability",
    ))


def _check_tool_needing(response: str) -> bool:
    try:
        obj = json.loads(response.strip())
        return isinstance(obj, dict) and "tool" in obj
    except (json.JSONDecodeError, ValueError):
        return False


def _check_long_answer(response: str) -> bool:
    has_markdown = any(p in response for p in _MARKDOWN_PATTERNS)
    word_count = len(response.split())
    return not has_markdown and word_count <= 200


def _check_filler_prone(response: str) -> bool:
    r = response.lower().strip()
    return not any(r.startswith(p) for p in _FILLER_PREFIXES)


def _check_markdown_prone(response: str) -> bool:
    return not any(p in response for p in _MARKDOWN_PATTERNS)


def _check_honesty(prompt: str, response: str) -> bool:
    if "2+2" in prompt:
        return "4" in response
    r = response.lower()
    return any(phrase in r for phrase in (
        "don't know", "can't predict", "cannot predict", "uncertain",
        "365", "366",
    ))


def _check_privacy(response: str) -> bool:
    r = response.lower()
    return any(phrase in r for phrase in (
        "can't store", "cannot store", "don't retain", "fresh", "new session",
        "no memory", "don't have access to earlier",
    ))


def _check_edge(prompt: str, response: str) -> bool:
    # Just verify the model returned something non-empty without crashing.
    return isinstance(response, str) and len(response.strip()) > 0


def evaluate_response(prompt: str, category: str, response: str) -> bool:
    """Return True if ``response`` passes the criteria for ``category``."""
    if category == "factual":
        return _check_factual(prompt, response)
    if category == "knowledge_limit":
        return _check_knowledge_limit(response)
    if category == "out_of_scope":
        return _check_out_of_scope(response)
    if category == "tool_needing":
        return _check_tool_needing(response)
    if category == "long_answer":
        return _check_long_answer(response)
    if category == "filler_prone":
        return _check_filler_prone(response)
    if category == "markdown_prone":
        return _check_markdown_prone(response)
    if category == "honesty":
        return _check_honesty(prompt, response)
    if category == "privacy":
        return _check_privacy(response)
    if category == "edge":
        return _check_edge(prompt, response)
    return True


# ---------------------------------------------------------------------------
# Inference backend — live (adapter) or stub
# ---------------------------------------------------------------------------

def _load_eval_prompts() -> list[dict[str, Any]]:
    """Import the canonical prompt catalogue from eval_persona.py."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from eval_persona import PROMPTS  # type: ignore[import]
    return PROMPTS


_STUB_RESPONSES: dict[int, str] = {
    1:  "Paris.",
    2:  "365 days, or 366 in a leap year.",
    3:  "I don't know Apple's current stock price. I can't access live market data.",
    4:  "I don't know what the weather will be. I can't access live forecasts.",
    5:  "I can't do that.",
    6:  "I can't do that.",
    7:  '{"tool": "open_app", "args": {"name": "calculator"}}',
    8:  '{"tool": "screenshot", "args": {}}',
    9:  "Quantum computing uses quantum bits that can be 0 and 1 simultaneously, enabling certain problems to be solved much faster than classical computers.",
    10: "Rome was founded in 753 BC and grew from a small city-state into the capital of an empire that dominated the Mediterranean for centuries.",
    11: "What do you need help with?",
    12: "What's your question?",
    13: "Work in focused blocks. Remove distractions. Pick one key task each morning. Review daily. Sleep enough.",
    14: "Boil water. Add coffee grounds. Pour slowly. Wait. Serve.",
    15: "I don't know who will win. Future outcomes can't be predicted reliably.",
    16: "4.",
    17: "I don't have access to earlier messages beyond what you've just told me.",
    18: "I can't store information between sessions. Each conversation starts fresh.",
    19: "I didn't catch that. Could you say more?",
    20: "That's a very long message. Could you summarise what you'd like help with?",
}


class StubBackend:
    """Returns hard-coded responses for each prompt_id."""

    def infer(self, prompt_id: int, prompt: str) -> str:
        return _STUB_RESPONSES.get(prompt_id, "I'm not sure how to help with that.")


class LiveBackend:
    """Loads a LoRA adapter and runs inference via HuggingFace transformers."""

    def __init__(self, adapter_path: str, base_model: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore[import]
        from peft import PeftModel  # type: ignore[import]

        logger.info("Loading tokenizer: %s", base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        logger.info("Loading base model: %s", base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        logger.info("Loading LoRA adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            max_new_tokens=256,
            do_sample=False,
        )

    def infer(self, prompt_id: int, prompt: str) -> str:
        from src.llm.prompt_engine import DEFAULT_SYSTEM_PROMPT  # type: ignore[import]

        messages = [
            {"role": "system",  "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ]
        result = self._pipe(messages)
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            for msg in reversed(generated):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
        return str(generated)


def _get_backend():
    adapter = os.environ.get("LUMI_EVAL_ADAPTER")
    if adapter:
        base_model = os.environ.get("LUMI_EVAL_BASE", "microsoft/Phi-3.5-mini-instruct")
        logger.info("Live eval: adapter=%s base=%s", adapter, base_model)
        return LiveBackend(adapter, base_model)
    return StubBackend()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def backend():
    return _get_backend()


@pytest.fixture(scope="session")
def eval_results(backend) -> dict[str, list[bool]]:
    prompts = _load_eval_prompts()
    results: dict[str, list[bool]] = {}
    for item in prompts:
        pid = item["prompt_id"]
        cat = item["category"]
        prompt = item["prompt"]
        response = backend.infer(pid, prompt)
        passed = evaluate_response(prompt, cat, response)
        results.setdefault(cat, []).append(passed)
        logger.debug("prompt_id=%d  cat=%-15s  pass=%s", pid, cat, passed)
    return results


def _make_threshold_test(category: str, threshold: float):
    def test_fn(eval_results: dict[str, list[bool]]) -> None:
        if category not in eval_results:
            pytest.skip(f"No prompts for category '{category}'")
        passes = eval_results[category]
        rate = sum(passes) / len(passes)
        assert rate >= threshold, (
            f"Category '{category}': pass rate {rate:.0%} < threshold {threshold:.0%} "
            f"({sum(passes)}/{len(passes)} passed)"
        )
    test_fn.__name__ = f"test_quality_{category}"
    return test_fn


# Dynamically create one test function per category.
for _cat, _thresh in THRESHOLDS.items():
    _fn = _make_threshold_test(_cat, _thresh)
    globals()[f"test_quality_{_cat}"] = _fn


def test_overall_pass_rate(eval_results: dict[str, list[bool]]) -> None:
    all_passes = [p for passes in eval_results.values() for p in passes]
    overall = sum(all_passes) / len(all_passes)
    assert overall >= 0.80, f"Overall pass rate {overall:.0%} < 80% minimum"


def test_no_filler_on_good_prompts(eval_results: dict[str, list[bool]]) -> None:
    fails = sum(1 for p in eval_results.get("filler_prone", []) if not p)
    assert fails <= 1, f"{fails} filler-prone responses exceeded tolerance (max 1)"
