"""
Phase 7 Wave-0 gate: measure the base pipeline round-trip latency.

Simulates N canned queries through the Orchestrator's inference path
(TranscriptReadyEvent → LLMResponseReadyEvent) using the real ReasoningRouter
with a real model, but with TTS and audio I/O mocked out.

Usage:
    uv run python scripts/measure_base_latency.py
    uv run python scripts/measure_base_latency.py --n 30 --threshold 1.7

Exit code:
    0  — p95 latency is below the threshold (Phase 7 may proceed)
    1  — p95 latency exceeds the threshold (defer Phase 7, optimise first)
    2  — model file not found or other setup error

Gate criterion from the architect plan:
    p95 < 1.7 s  →  green, Phase 7 proceeds
    p95 ≥ 1.7 s  →  red, LightRAG retrieval would push total past 2 s
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
import queue
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Canned queries representative of real Lumi usage
# ---------------------------------------------------------------------------

QUERIES: list[str] = [
    "What time is it?",
    "Open Firefox for me.",
    "What's the weather like today?",
    "Set a reminder for my meeting at 3pm.",
    "Play some music.",
    "Tell me a quick joke.",
    "What can you help me with?",
    "Close this window.",
    "Search for Python tutorials.",
    "How do I rename a file in the terminal?",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure Lumi base pipeline latency.")
    p.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of queries to run (default: 20; queries are cycled if n > len(QUERIES))",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.7,
        help="p95 latency gate in seconds (default: 1.7)",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return p.parse_args()


def _load_components(config_path: str):
    """Import and initialise the real reasoning router from the project."""
    try:
        from src.core.config import load_config
        from src.llm.model_loader import ModelLoader
        from src.llm.prompt_engine import PromptEngine
        from src.llm.memory import ConversationMemory
        from src.llm.reasoning_router import ReasoningRouter
    except ImportError as exc:
        print(f"[ERROR] Import failed: {exc}", file=sys.stderr)
        print("Run from the project root: uv run python scripts/measure_base_latency.py", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(config_path)
    model_path = Path(cfg.llm.model_path)

    if not model_path.exists():
        print(f"[ERROR] LLM model not found at: {model_path}", file=sys.stderr)
        print("Download the model before running this benchmark.", file=sys.stderr)
        sys.exit(2)

    print(f"Loading model: {model_path} ...", flush=True)
    loader = ModelLoader(cfg.llm)
    loader.wake()

    engine = PromptEngine(cfg.llm)
    memory = ConversationMemory(cfg.llm)
    event_queue: queue.Queue = queue.Queue()

    router = ReasoningRouter(
        model_loader=loader,
        prompt_engine=engine,
        memory=memory,
        config=cfg.llm,
        event_queue=event_queue,
    )
    return router, cfg


def _run_query(router, query: str) -> float:
    """Run one query and return wall-clock seconds from call to last token."""
    cancel = threading.Event()
    collected: list[str] = []
    t0 = time.perf_counter()

    def _consume(token: str) -> None:
        collected.append(token)

    # generate() is synchronous and blocks until the response is complete.
    router.generate(
        text=query,
        cancel_flag=cancel,
        utterance_id="bench",
    )
    return time.perf_counter() - t0


def main() -> None:
    args = _parse_args()
    router, cfg = _load_components(args.config)

    print(f"\nRunning {args.n} queries  (gate: p95 < {args.threshold} s)\n")
    print(f"{'#':>3}  {'Query':<42}  {'Latency':>8}")
    print("-" * 58)

    latencies: list[float] = []
    for i in range(args.n):
        query = QUERIES[i % len(QUERIES)]
        elapsed = _run_query(router, query)
        latencies.append(elapsed)
        marker = "  ✓" if elapsed < args.threshold else "  ✗"
        print(f"{i+1:>3}  {query[:42]:<42}  {elapsed:>7.3f}s{marker}")

    print("\n" + "=" * 58)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p_max = max(latencies)
    p_min = min(latencies)

    print(f"  Queries : {args.n}")
    print(f"  Min     : {p_min:.3f} s")
    print(f"  p50     : {p50:.3f} s")
    print(f"  p95     : {p95:.3f} s   ← gate")
    print(f"  Max     : {p_max:.3f} s")
    print(f"  Gate    : {args.threshold} s")
    print("=" * 58)

    if p95 < args.threshold:
        print(f"\n[PASS] p95 {p95:.3f}s < {args.threshold}s — Phase 7 may proceed.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] p95 {p95:.3f}s ≥ {args.threshold}s — defer Phase 7.")
        print("       Adding 150–600 ms of RAG retrieval would push total past 2 s.")
        print("       Optimise the base pipeline before enabling LightRAG.")
        sys.exit(1)


if __name__ == "__main__":
    main()
