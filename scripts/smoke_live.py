"""
Lumi Smoke Live — manual live-model smoke test.

Loads each real model in sequence and runs one operation per stage,
printing PASS/FAIL/SKIP with wall-clock milliseconds.

Stages:
  STT  — load faster-whisper, transcribe 1-second silent WAV
  LLM  — load ModelLoader from config, call generate()
  TTS  — load KokoroTTS from config, synthesize "Hello"
  RAG  — init DocumentStore in temp dir, upsert one doc, query it

Usage:
    uv run python scripts/smoke_live.py
    uv run python scripts/smoke_live.py --skip-llm --skip-tts

Exit code: 0 if all non-skipped stages PASS, 1 if any FAIL.

NOTE: This script is importable as a module — all logic is in functions;
only the ``if __name__ == "__main__"`` block calls main().

DO NOT run in CI — no GPU available on CI runners.
Tag any pytest usage with @pytest.mark.live.
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Colour codes (ANSI) — degrade gracefully on unsupported terminals
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    """Outcome of a single smoke stage."""

    name: str
    status: str       # "PASS" | "FAIL" | "SKIP"
    elapsed_ms: int
    message: str

    def __str__(self) -> str:
        if self.status == "PASS":
            colour = _GREEN
        elif self.status == "FAIL":
            colour = _RED
        else:
            colour = _YELLOW
        return (
            f"  [{colour}{self.status}{_RESET}] "
            f"{self.name:<8} "
            f"{self.elapsed_ms}ms"
            + (f"  — {self.message}" if self.message and self.message != "ok" else "")
        )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_stage(
    name: str,
    fn: Callable[[], Any],
    threshold_ms: int,
    validate: Callable[[Any], bool],
    skip: bool = False,
) -> SmokeResult:
    """Execute *fn* with a wall-clock timeout.

    Args:
        name:         Human-readable stage label.
        fn:           Zero-arg callable that loads/runs the model operation.
        threshold_ms: Maximum allowed wall-clock milliseconds.
        validate:     Called with fn's return value; must return True for PASS.
        skip:         When True, return SKIP without calling fn.

    Returns:
        A :class:`SmokeResult` with status PASS, FAIL, or SKIP.
    """
    if skip:
        return SmokeResult(name=name, status="SKIP", elapsed_ms=0, message="skipped by flag")

    result_box: list[Any] = []
    exc_box: list[Exception] = []

    def _target() -> None:
        try:
            result_box.append(fn())
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)

    t0 = time.perf_counter()
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=threshold_ms / 1000.0)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if thread.is_alive():
        return SmokeResult(
            name=name,
            status="FAIL",
            elapsed_ms=elapsed_ms,
            message=f"timeout after {threshold_ms}ms",
        )

    if exc_box:
        return SmokeResult(
            name=name,
            status="FAIL",
            elapsed_ms=elapsed_ms,
            message=str(exc_box[0]),
        )

    value = result_box[0] if result_box else None
    if not validate(value):
        return SmokeResult(
            name=name,
            status="FAIL",
            elapsed_ms=elapsed_ms,
            message=f"validation failed — got {value!r}",
        )

    return SmokeResult(name=name, status="PASS", elapsed_ms=elapsed_ms, message="ok")


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------


def stage_stt(skip: bool = False) -> SmokeResult:
    """STT stage — load faster-whisper, transcribe 1-second silence."""

    def _run() -> str:
        import numpy as np
        from faster_whisper import WhisperModel

        # Ensure project root is on sys.path for config loading.
        _add_project_root()
        from src.core.config import load_config

        config = load_config(_config_path())

        # Load model from local path if available, else fall back to model_size.
        model_path = config.scribe.model_path
        if Path(model_path).is_dir():
            model = WhisperModel(model_path, device="cpu", compute_type="int8")
        else:
            model = WhisperModel(
                config.scribe.model_size, device="cpu", compute_type="int8"
            )

        # Transcribe 1-second silent audio (float32 zeros, 16 kHz).
        audio = np.zeros(16_000, dtype=np.float32)
        segments, _info = model.transcribe(audio, beam_size=1)
        text = " ".join(seg.text for seg in segments)
        # Silence may produce empty string — that is acceptable.
        return text  # str (may be "")

    def _validate(result: Any) -> bool:
        return isinstance(result, str)

    return run_stage(
        name="STT",
        fn=_run,
        threshold_ms=5_000,
        validate=_validate,
        skip=skip,
    )


def stage_llm(skip: bool = False) -> SmokeResult:
    """LLM stage — load ModelLoader, call generate() with a short prompt."""

    def _run() -> str:
        import threading

        _add_project_root()
        from src.core.config import load_config
        from src.llm.memory import ConversationMemory
        from src.llm.model_loader import ModelLoader
        from src.llm.prompt_engine import PromptEngine
        from src.llm.reasoning_router import ReasoningRouter

        config = load_config(_config_path())

        loader = ModelLoader()
        loader.load(config.llm)

        memory = ConversationMemory(config.llm)
        engine = PromptEngine()
        router = ReasoningRouter(
            model_loader=loader,
            prompt_engine=engine,
            memory=memory,
            config=config.llm,
        )

        cancel = threading.Event()
        response = router.generate("Say hello in one word.", cancel)
        return response

    def _validate(result: Any) -> bool:
        return isinstance(result, str) and len(result.strip()) > 0

    return run_stage(
        name="LLM",
        fn=_run,
        threshold_ms=10_000,
        validate=_validate,
        skip=skip,
    )


def stage_tts(skip: bool = False) -> SmokeResult:
    """TTS stage — load KokoroTTS, synthesize 'Hello', assert audio > 0 bytes."""

    def _run() -> int:
        import numpy as np

        _add_project_root()
        from src.core.config import load_config

        config = load_config(_config_path())

        import kokoro_onnx  # type: ignore[import]

        kokoro = kokoro_onnx.Kokoro(config.tts.model_path, config.tts.voices_path)
        samples, _phonemes = kokoro.create(
            "Hello", voice=config.tts.voice, speed=1.0, lang="en-us"
        )
        audio_bytes = samples.astype(np.float32).tobytes()
        return len(audio_bytes)

    def _validate(result: Any) -> bool:
        return isinstance(result, int) and result > 0

    return run_stage(
        name="TTS",
        fn=_run,
        threshold_ms=5_000,
        validate=_validate,
        skip=skip,
    )


def stage_rag(skip: bool = False) -> SmokeResult:
    """RAG stage — init DocumentStore in temp dir, upsert doc, query it."""

    def _run() -> int:
        import tempfile

        _add_project_root()
        from src.core.config import RAGConfig
        from src.rag.store import DocumentStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "smoke_rag.db")
            rag_config = RAGConfig(db_path=db_path)
            store = DocumentStore(rag_config)
            store.init_schema()

            doc = store.upsert_document(
                path="/smoke/test_doc.txt", sha256="abc123def456"
            )
            store.insert_chunk(
                document_id=doc.id,
                chunk_idx=0,
                text="Lumi is a local AI assistant with voice interface.",
                char_start=0,
                char_end=51,
            )

            hits = store.search_fts("Lumi voice assistant", top_k=5)
            store.close()
            return len(hits)

    def _validate(result: Any) -> bool:
        return isinstance(result, int) and result >= 1

    return run_stage(
        name="RAG",
        fn=_run,
        threshold_ms=2_000,
        validate=_validate,
        skip=skip,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the absolute path to the Lumi project root."""
    return Path(__file__).resolve().parent.parent


def _add_project_root() -> None:
    """Ensure project root is on sys.path for src.* imports."""
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _config_path() -> str:
    """Return path to config.yaml relative to project root."""
    return str(_project_root() / "config.yaml")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run all (non-skipped) smoke stages.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        0 if all non-skipped stages PASS, 1 if any FAIL.
    """
    parser = argparse.ArgumentParser(
        description="Lumi live-model smoke test — runs real model inference"
    )
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT stage")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM stage")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS stage")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG stage")

    # Accept a positional dummy arg so tests can pass [""] without error.
    parser.add_argument("args", nargs="*", help=argparse.SUPPRESS)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    print(f"\n{_BOLD}Lumi Smoke Live{_RESET}")
    print("=" * 48)

    results = [
        stage_stt(skip=args.skip_stt),
        stage_llm(skip=args.skip_llm),
        stage_tts(skip=args.skip_tts),
        stage_rag(skip=args.skip_rag),
    ]

    for result in results:
        print(result)

    print("=" * 48)

    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print(f"{_RED}{_BOLD}{len(failures)} stage(s) FAILED.{_RESET}\n")
        return 1

    print(f"{_GREEN}{_BOLD}All stages passed.{_RESET}\n")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
