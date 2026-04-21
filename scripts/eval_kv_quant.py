"""Wave I1 eval harness: benchmark TurboQuant KV cache quantization.

Compares baseline FP16 vs. ``kv_cache_quant="turbo3"`` on 5 fixed prompts.
Requires a real GGUF model and llama-cpp-python with upstream PR #21089.

Exit code: 0 if delta >= -5%, 1 otherwise.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)

from src.core.config import load_config  # noqa: E402
from src.llm.model_loader import ModelLoader  # noqa: E402

PROMPTS: tuple[str, ...] = (
    "What time is it right now?",
    "Summarize the Fast Walsh-Hadamard Transform in one sentence.",
    "List three common KV cache quantization strategies.",
    "Explain why 3-bit quantization can beat 4-bit under rotation.",
    "Give a one-line definition of perplexity.",
)
MAX_NEW_TOKENS: int = 64
REGRESSION_THRESHOLD_PCT: float = -5.0  # allow up to 5% slowdown


def _run_once(config, prompt: str) -> tuple[float, float]:
    """Return (first_token_latency_s, tokens_per_second)."""
    loader = ModelLoader()
    loader.load(config)
    model = loader.model
    t0 = time.perf_counter()
    first_token_latency: float | None = None
    n_tokens = 0
    for chunk in model.create_completion(prompt, max_tokens=MAX_NEW_TOKENS, stream=True):
        if first_token_latency is None:
            first_token_latency = time.perf_counter() - t0
        n_tokens += 1
    total = time.perf_counter() - t0
    loader.unload()
    tps = n_tokens / total if total > 0 else 0.0
    return (first_token_latency or 0.0, tps)


def main() -> int:
    top = load_config()
    base_cfg = top.llm
    turbo_cfg = dataclasses.replace(base_cfg, kv_cache_quant="turbo3")

    print(f"{'prompt':<48} {'base_tps':>10} {'turbo_tps':>11} {'delta%':>8}")
    print("-" * 80)

    worst_delta_pct = 0.0
    for prompt in PROMPTS:
        _, base_tps = _run_once(base_cfg, prompt)
        _, turbo_tps = _run_once(turbo_cfg, prompt)
        delta_pct = ((turbo_tps - base_tps) / base_tps * 100.0) if base_tps else 0.0
        worst_delta_pct = min(worst_delta_pct, delta_pct)
        short = (prompt[:45] + "...") if len(prompt) > 48 else prompt
        print(f"{short:<48} {base_tps:>10.2f} {turbo_tps:>11.2f} {delta_pct:>7.1f}%")

    print("-" * 80)
    print(f"worst delta: {worst_delta_pct:.1f}% (threshold: {REGRESSION_THRESHOLD_PCT:.1f}%)")
    return 0 if worst_delta_pct >= REGRESSION_THRESHOLD_PCT else 1


if __name__ == "__main__":
    sys.exit(main())
