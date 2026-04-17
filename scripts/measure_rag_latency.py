"""
Phase 7 Wave-5 gate: measure end-to-end RAG retrieval + LLM inference latency.

Seeds a temporary DocumentStore with short sample documents, then runs N
canned queries through the full RAG path (retrieval → prompt injection →
LLM generation) using the real ReasoningRouter with use_rag=True.

Usage:
    uv run python scripts/measure_rag_latency.py
    uv run python scripts/measure_rag_latency.py --n 20 --threshold 2.0

Exit code:
    0  — p95 latency is below the threshold (gate passes)
    1  — p95 latency exceeds the threshold
    2  — model file not found, embedder unavailable, or other setup error

Gate criterion:
    p95 < 2.0 s  →  green (retrieval overhead stays within the 3-second voice
                            UI threshold)
    p95 ≥ 2.0 s  →  red  (base pipeline + retrieval exceeds threshold)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import statistics
import sys
import tempfile
import threading
import time
import queue
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Sample knowledge-base documents (seeded into the temp store)
# ---------------------------------------------------------------------------

_SAMPLE_DOCS: dict[str, str] = {
    "lumi_overview.txt": (
        "Project Lumi is a local, privacy-first desktop AI assistant. "
        "It runs entirely on-device using a quantised language model (Phi-3.5-mini Q4_K_M). "
        "Lumi listens for the wake word 'Hey Lumi', transcribes speech with faster-whisper, "
        "generates a response, and speaks it back using the Kokoro TTS engine. "
        "The Brain (Python) and Body (Godot 4 overlay) communicate over a raw TCP "
        "length-prefixed JSON protocol."
    ),
    "rag_system.txt": (
        "The Lumi RAG system stores personal documents in an SQLite database using "
        "FTS5 for BM25 keyword search and sqlite-vec for dense vector kNN retrieval. "
        "Retrieval uses Reciprocal Rank Fusion (RRF) to merge BM25 and kNN rankings. "
        "The embedding model is all-MiniLM-L6-v2 (384 dimensions, CPU-only). "
        "RAG is disabled by default and must be enabled via config.rag.enabled: true."
    ),
    "os_tools.txt": (
        "Lumi's OS control layer provides four tools: AppLaunchTool (opens applications "
        "from an allowlist), ClipboardTool (read/write the system clipboard via xclip), "
        "FileInfoTool (report file metadata without path traversal), and WindowListTool "
        "(list open windows via wmctrl). Tools are invoked via <tool_call> JSON blocks "
        "emitted by the LLM, parsed by tool_call_parser, and executed by ToolExecutor "
        "with a configurable timeout and an interrupt cancel flag."
    ),
    "godot_frontend.txt": (
        "The Godot 4 frontend renders a transparent borderless overlay. "
        "It connects to the Python Brain via StreamPeerTCP using a 4-byte big-endian "
        "length-prefix frame protocol. Events received include state_change, tts_start, "
        "tts_viseme, tts_stop, transcript, llm_token, rag_retrieval, and error. "
        "Events sent include interrupt and user_text. The avatar uses per-viseme-group "
        "mouth animations driven by VisemeEvent data from the Kokoro TTS engine."
    ),
    "config_guide.txt": (
        "config.yaml controls all runtime parameters. Key sections: "
        "audio (vad_threshold, sample_rate, channels), "
        "scribe (model, beam_size), "
        "llm (model_path, context_length, max_tokens, temperature), "
        "tts (model_path, speed, sample_rate), "
        "ipc (address, port, enabled), "
        "rag (db_path, corpus_dir, enabled, embedding_model, chunk_size, chunk_overlap, "
        "     retrieval_top_k, min_score, timeout_ms), "
        "tools (enabled, allowed_apps, timeout_s), "
        "vision (enabled, model_path, idle_unload_s)."
    ),
}

# Queries representative of RAG-triggering user input
_QUERIES: list[str] = [
    "Tell me about the Lumi RAG system.",
    "How does the Godot frontend communicate with the Brain?",
    "What OS tools does Lumi support?",
    "Explain the config.yaml rag section.",
    "What embedding model does Lumi use for RAG?",
    "How does Lumi handle wake word detection?",
    "What is the IPC protocol format?",
    "Which TTS engine does Lumi use?",
    "How do tool calls work in Lumi?",
    "What is the role of Reciprocal Rank Fusion?",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure Lumi RAG end-to-end latency.")
    p.add_argument("--n", type=int, default=10,
                   help="Number of queries to run (default: 10)")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="p95 latency gate in seconds (default: 2.0)")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _setup(config_path: str, tmp_dir: Path):
    """Load config, seed store, return (router, rag_config)."""
    try:
        from src.core.config import load_config
        from src.llm.model_loader import ModelLoader
        from src.llm.prompt_engine import PromptEngine
        from src.llm.memory import ConversationMemory
        from src.llm.reasoning_router import ReasoningRouter
        from src.rag.store import DocumentStore
        from src.rag.retriever import RAGRetriever
        from src.rag.embedder import get_embedder
        from src.rag.chunker import chunk_text
    except ImportError as exc:
        print(f"[ERROR] Import failed: {exc}", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(config_path)
    model_path = Path(cfg.llm.model_path)

    if not model_path.exists():
        print(f"[ERROR] LLM model not found at: {model_path}", file=sys.stderr)
        sys.exit(2)

    # Override db_path to a temp file so we don't touch the real store.
    db_path = str(tmp_dir / "bench_rag.db")
    rag_cfg = dataclasses.replace(cfg.rag, db_path=db_path, enabled=True)

    print("Loading embedder …", flush=True)
    try:
        embedder = get_embedder(rag_cfg)
    except Exception as exc:
        print(f"[ERROR] Embedder init failed: {exc}", file=sys.stderr)
        sys.exit(2)

    print("Seeding document store …", flush=True)
    store = DocumentStore(rag_cfg)
    _seed_store(store, embedder, rag_cfg, tmp_dir)

    print(f"Loading LLM model: {model_path} …", flush=True)
    loader = ModelLoader()
    loader.load(cfg.llm)

    retriever = RAGRetriever(store, rag_cfg)
    router = ReasoningRouter(
        model_loader=loader,
        prompt_engine=PromptEngine(),
        memory=ConversationMemory(cfg.llm.memory_dir),
        config=cfg.llm,
        event_queue=queue.Queue(),
        retriever=retriever,
    )
    return router


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _seed_store(store, embedder, cfg, tmp_dir: Path) -> None:
    from src.rag.chunker import chunk_text

    for filename, content in _SAMPLE_DOCS.items():
        doc_path = str(tmp_dir / filename)
        sha = _sha256(content)
        chunks = chunk_text(content, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        if not chunks:
            continue
        embeddings = embedder.encode([c.text for c in chunks])
        doc = store.upsert_document(doc_path, sha)
        store.delete_document_chunks(doc.id)
        for chunk, emb in zip(chunks, embeddings):
            cr = store.insert_chunk(
                document_id=doc.id,
                chunk_idx=chunk.chunk_idx,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            store.insert_vector(cr.id, emb)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _run_query(router, query: str) -> float:
    """Return wall-clock seconds for one full retrieval + inference cycle."""
    cancel = threading.Event()
    t0 = time.perf_counter()
    router.generate(text=query, cancel_flag=cancel, utterance_id="bench", use_rag=True)
    return time.perf_counter() - t0


def main() -> None:
    args = _parse_args()

    with tempfile.TemporaryDirectory(prefix="lumi_rag_bench_") as tmp:
        tmp_dir = Path(tmp)
        router = _setup(args.config, tmp_dir)

        print(f"\nRunning {args.n} queries  (gate: p95 < {args.threshold} s)\n")
        print(f"{'#':>3}  {'Query':<46}  {'Latency':>8}")
        print("-" * 62)

        latencies: list[float] = []
        for i in range(args.n):
            query = _QUERIES[i % len(_QUERIES)]
            elapsed = _run_query(router, query)
            latencies.append(elapsed)
            marker = "  ✓" if elapsed < args.threshold else "  ✗"
            print(f"{i+1:>3}  {query[:46]:<46}  {elapsed:>7.3f}s{marker}")

    print("\n" + "=" * 62)
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
    print("=" * 62)

    if p95 < args.threshold:
        print(f"\n[PASS] p95 {p95:.3f}s < {args.threshold}s — RAG latency gate passes.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] p95 {p95:.3f}s ≥ {args.threshold}s — retrieval overhead too high.")
        print("       Optimise the retrieval path or reduce max_tokens before enabling RAG.")
        sys.exit(1)


if __name__ == "__main__":
    main()
