"""GGUF export pipeline for Project Lumi.

Merges a LoRA adapter (produced by ``scripts/train_lumi.py``) into its base
model, converts the merged HuggingFace checkpoint to GGUF format via
llama.cpp's ``convert_hf_to_gguf.py``, quantizes the GGUF to the requested
type (default ``Q4_K_M``), and optionally runs the persona evaluation harness
to surface any quality delta versus a stored baseline.

Key fixes in Ring 2 (2026-05-04):
  - CRITICAL: Dropped ``trust_remote_code=True`` from model/tokenizer loading.
    Phi-3.5's bundled ``modeling_phi3.py`` produces a state_dict that converts
    to corrupt GGUF (outputs token-salad). Native transformers Phi3ForCausalLM
    (transformers 4.39+) is compatible and produces correct GGUFs.
  - ``_get_tied_weight_keys`` monkey-patch is signature-agnostic (*args, **kwargs)
    and returns [] on error (NOT None, which breaks ``set(...)`` in caller).
  - Copies SentencePiece ``tokenizer.model`` from base checkpoint to merged dir.
    convert_hf_to_gguf.py requires it for Phi-3 family.
  - New ``--llama-cpp-dir`` CLI arg for llama-quantize discovery via
    ``<llama-cpp-dir>/build/bin/llama-quantize`` (no PATH pollution needed).

Requirements
------------
    uv sync --extra qlora          # installs transformers, peft, torch, gguf
    # llama.cpp built locally (not on PATH):
    #   git clone https://github.com/ggerganov/llama.cpp
    #   cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize

Usage
-----
    # Dry run — log all steps without executing anything:
    uv run python scripts/merge_and_quantize.py --dry-run

    # Full pipeline from a trained LoRA adapter (Ring 2 reproducible):
    uv run python scripts/merge_and_quantize.py \\
        --adapter-dir   models/lumi-lora-v1 \\
        --base-model    models/llm/checkpoints/phi-3.5-mini \\
        --output-dir    models/llm \\
        --output-name   lumi-phi35-v1-Q4_K_M.gguf \\
        --quant-type    Q4_K_M \\
        --llama-cpp-dir /home/hsozer/Dev/llama.cpp \\
        --skip-eval

    # Skip the persona eval step:
    uv run python scripts/merge_and_quantize.py \\
        --adapter-dir models/lumi-lora-v1 \\
        --skip-eval

Output
------
A quantized GGUF file at ``--output-dir / --output-name`` ready to be loaded
by llama-cpp-python.  The intermediate fp16 GGUF and the merged HuggingFace
directory are removed on success.

GITIGNORE: The LoRA artifacts (models/lumi-lora-v1/) and GGUF output
(models/llm/*.gguf) are gitignored — produced by the pipeline, not committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_ADAPTER_DIR = Path("models/lumi-lora-v1")
DEFAULT_OUTPUT_DIR = Path("models/llm")
DEFAULT_OUTPUT_NAME = "lumi-phi35-v1-Q4_K_M.gguf"
DEFAULT_QUANT_TYPE = "Q4_K_M"
DEFAULT_LLAMA_CPP_DIR = ""

EVAL_SCRIPT = Path(__file__).parent / "eval_persona.py"
BASELINE_PATH = Path("results/eval_baseline.json")


# ---------------------------------------------------------------------------
# Step 1 — Merge LoRA adapter into base model (Python)
# ---------------------------------------------------------------------------


def merge_lora(
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
    dry_run: bool,
) -> None:
    """Load base + adapter, merge weights, save merged HF checkpoint."""
    logger.info("Step 1 — Merging LoRA adapter into base model")
    logger.info("  base model  : %s", base_model)
    logger.info("  adapter dir : %s", adapter_dir)
    logger.info("  merged dir  : %s", merged_dir)

    if dry_run:
        logger.info("  [dry-run] skipping merge")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        from peft import PeftModel  # type: ignore[import]
    except ImportError as exc:
        logger.error("Missing dependency for merge step: %s", exc)
        logger.error("Install with: uv sync --extra qlora")
        raise

    t0 = time.monotonic()

    logger.info("  Loading tokenizer …")
    # IMPORTANT: do NOT pass trust_remote_code=True here.  The Phi-3 / Phi-3.5
    # checkpoint ships an older bundled `modeling_phi3.py` that, when used with
    # transformers ≥5.x + PEFT merge_and_unload + save_pretrained, produces a
    # safetensors file whose weights look correct in PyTorch (forward pass
    # generates coherent text) but break on the convert_hf_to_gguf.py path —
    # the resulting GGUF outputs token-salad regardless of quantisation level.
    # Native transformers Phi3ForCausalLM (transformers 4.39+) is fully
    # compatible and produces a GGUF that quantises cleanly to Q4_K_M.
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("  Loading base model (fp16) …")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,  # was `torch_dtype=` in transformers <5.0
        device_map="cpu",
    )

    logger.info("  Attaching PEFT adapter …")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    logger.info("  Merging and unloading adapter …")
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Saving merged model to %s …", merged_dir)

    # Workaround for transformers ≥5.5 bug where _get_tied_weight_keys()
    # expects each tied-weights entry to be a dict but Phi-3 returns a list
    # (or, in 5.5.x, the recursive helper takes 6 positional args and dies on
    # certain submodule shapes during save_pretrained).  We swap in a
    # signature-agnostic wrapper that swallows AttributeError / TypeError /
    # KeyError and lets save_pretrained continue; LoRA only modifies attention
    # and MLP projections so missing tied-weight info is harmless here.
    try:
        from transformers import modeling_utils as _mu  # type: ignore[import]
        _orig = _mu._get_tied_weight_keys

        def _safe_get_tied_weight_keys(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return _orig(*args, **kwargs)
            except (AttributeError, TypeError, KeyError):
                # Caller (remove_tied_weights_from_state_dict) wraps this in
                # set(...), so we MUST return an iterable, never None.
                return []

        _mu._get_tied_weight_keys = _safe_get_tied_weight_keys
    except (ImportError, AttributeError):
        pass

    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # convert_hf_to_gguf.py needs the SentencePiece tokenizer.model file for
    # Phi-3 / Llama / Mistral families.  AutoTokenizer.save_pretrained only
    # writes the fast tokenizer.json, so copy the original sentencepiece model
    # if one exists in the base checkpoint.
    base_path = Path(base_model)
    if base_path.is_dir():
        src_spm = base_path / "tokenizer.model"
        if src_spm.is_file():
            dst_spm = merged_dir / "tokenizer.model"
            shutil.copy2(src_spm, dst_spm)
            logger.info("  Copied SentencePiece tokenizer.model → %s", dst_spm)

    elapsed = time.monotonic() - t0
    logger.info("  Merge complete in %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Step 2 — Convert merged HF model to GGUF (subprocess)
# ---------------------------------------------------------------------------


def _find_convert_script(llama_cpp_dir: str) -> str:
    """Return the path to convert_hf_to_gguf.py or raise if not found."""
    if llama_cpp_dir:
        candidate = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found in --llama-cpp-dir: {llama_cpp_dir}"
        )

    # Fallback: check PATH for llama-convert wrapper
    if shutil.which("llama-convert"):
        return "llama-convert"

    raise FileNotFoundError(
        "llama.cpp convert script not found.  "
        "Provide --llama-cpp-dir or ensure 'llama-convert' is on PATH."
    )


def convert_to_gguf(
    merged_dir: Path,
    fp16_path: Path,
    llama_cpp_dir: str,
    dry_run: bool,
) -> None:
    """Convert merged HF checkpoint to fp16 GGUF."""
    logger.info("Step 2 — Converting merged model to GGUF (fp16)")
    logger.info("  source : %s", merged_dir)
    logger.info("  output : %s", fp16_path)

    if dry_run:
        logger.info("  [dry-run] skipping conversion")
        return

    try:
        convert_script = _find_convert_script(llama_cpp_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise

    fp16_path.parent.mkdir(parents=True, exist_ok=True)

    if convert_script == "llama-convert":
        cmd = [
            "llama-convert",
            str(merged_dir),
            "--outfile",
            str(fp16_path),
        ]
    else:
        cmd = [
            sys.executable,
            convert_script,
            str(merged_dir),
            "--outfile",
            str(fp16_path),
        ]

    logger.info("  Running: %s", " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.stderr:
        for line in result.stderr.splitlines():
            logger.info("  [convert] %s", line)

    if result.returncode != 0:
        logger.error("Conversion failed (exit %d)", result.returncode)
        raise subprocess.CalledProcessError(result.returncode, cmd)

    logger.info("  Conversion complete in %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Step 3 — Quantize GGUF (subprocess)
# ---------------------------------------------------------------------------


def quantize_gguf(
    fp16_path: Path,
    output_path: Path,
    quant_type: str,
    dry_run: bool,
    llama_cpp_dir: str = "",
) -> None:
    """Quantize fp16 GGUF to the requested quantization type."""
    logger.info("Step 3 — Quantizing GGUF to %s", quant_type)
    logger.info("  input  : %s", fp16_path)
    logger.info("  output : %s", output_path)

    if dry_run:
        logger.info("  [dry-run] skipping quantization")
        return

    # First check PATH; then fall back to a build/bin/llama-quantize inside the
    # --llama-cpp-dir we already located convert_hf_to_gguf.py from.  This lets
    # us use a freshly built llama.cpp without the user having to add it to PATH.
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None and llama_cpp_dir:
        candidate = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            quantize_bin = str(candidate)
    if quantize_bin is None:
        raise FileNotFoundError(
            "'llama-quantize' not found on PATH or in --llama-cpp-dir/build/bin/. "
            "Build llama.cpp (cmake --build build --target llama-quantize) and "
            "either add the binary to PATH or pass --llama-cpp-dir."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [quantize_bin, str(fp16_path), str(output_path), quant_type]

    logger.info("  Running: %s", " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.stderr:
        for line in result.stderr.splitlines():
            logger.info("  [quantize] %s", line)

    if result.returncode != 0:
        logger.error("Quantization failed (exit %d)", result.returncode)
        raise subprocess.CalledProcessError(result.returncode, cmd)

    logger.info("  Quantization complete in %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Step 4 — Evaluate quality delta (optional)
# ---------------------------------------------------------------------------


def _load_baseline() -> dict | None:
    """Load the baseline eval report from results/eval_baseline.json, or None."""
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read baseline report: %s", exc)
    return None


def evaluate_quality(output_path: Path, dry_run: bool) -> None:
    """Run eval_persona.py and log pass/fail counts + delta vs baseline."""
    logger.info("Step 4 — Evaluating persona quality delta")
    logger.info("  model : %s", output_path)

    if dry_run:
        logger.info("  [dry-run] skipping evaluation")
        return

    if not EVAL_SCRIPT.exists():
        logger.warning("eval_persona.py not found at %s — skipping eval", EVAL_SCRIPT)
        return

    # Write eval output to a temporary file so we can parse it
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="eval_lumi_", delete=False
    ) as tmp:
        eval_output = Path(tmp.name)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--output",
        str(eval_output),
    ]

    logger.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info("  [eval] %s", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            logger.info("  [eval stderr] %s", line)

    if result.returncode != 0:
        logger.warning(
            "eval_persona.py exited with code %d — quality delta unavailable",
            result.returncode,
        )
        eval_output.unlink(missing_ok=True)
        return

    try:
        report = json.loads(eval_output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse eval report: %s", exc)
        eval_output.unlink(missing_ok=True)
        return
    finally:
        eval_output.unlink(missing_ok=True)

    summary = report.get("summary", {})
    passed = summary.get("passed", "?")
    failed = summary.get("failed", "?")
    pass_rate = summary.get("pass_rate", None)
    total_checks = summary.get("total_criteria_checks", "?")

    logger.info(
        "  Result: %s/%s criteria passed%s",
        passed,
        total_checks,
        f" ({pass_rate:.1%})" if isinstance(pass_rate, float) else "",
    )

    baseline = _load_baseline()
    if baseline:
        baseline_rate = baseline.get("summary", {}).get("pass_rate")
        if isinstance(baseline_rate, float) and isinstance(pass_rate, float):
            delta = pass_rate - baseline_rate
            sign = "+" if delta >= 0 else ""
            logger.info(
                "  Quality delta vs baseline: %s%.3f (baseline %.1f%% → now %.1f%%)",
                sign,
                delta,
                baseline_rate * 100,
                pass_rate * 100,
            )
        else:
            logger.info("  Baseline pass_rate unavailable — cannot compute delta")
    else:
        logger.info(
            "  No baseline found at %s — run eval_persona.py --output %s first",
            BASELINE_PATH,
            BASELINE_PATH,
        )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_merged_dir(merged_dir: Path, dry_run: bool) -> None:
    """Remove the temporary merged HF checkpoint directory."""
    logger.info("Cleanup — removing temporary merged directory: %s", merged_dir)
    if dry_run:
        logger.info("  [dry-run] skipping cleanup")
        return
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        logger.info("  Removed %s", merged_dir)
    else:
        logger.info("  Directory not found — nothing to remove")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full merge → convert → quantize → eval pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_path = output_dir / args.output_name

    # Derive an fp16 GGUF filename alongside the final output
    stem = Path(args.output_name).stem
    fp16_name = f"{stem}-fp16.gguf"
    fp16_path = output_dir / fp16_name

    # Temporary directory for the merged HF checkpoint
    merged_dir = Path(tempfile.mkdtemp(prefix="lumi_merged_"))

    logger.info("=== Lumi GGUF Export Pipeline ===")
    logger.info("  adapter dir  : %s", adapter_dir)
    logger.info("  base model   : %s", args.base_model)
    logger.info("  output       : %s", output_path)
    logger.info("  quant type   : %s", args.quant_type)
    logger.info("  dry run      : %s", args.dry_run)

    if not args.dry_run and not adapter_dir.exists():
        logger.error("Adapter directory not found: %s", adapter_dir)
        logger.error("Run scripts/train_lumi.py first.")
        sys.exit(1)

    try:
        # --- Step 1 ---
        merge_lora(
            base_model=args.base_model,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            dry_run=args.dry_run,
        )

        # --- Step 2 ---
        convert_to_gguf(
            merged_dir=merged_dir,
            fp16_path=fp16_path,
            llama_cpp_dir=args.llama_cpp_dir,
            dry_run=args.dry_run,
        )

        # --- Step 3 ---
        quantize_gguf(
            fp16_path=fp16_path,
            output_path=output_path,
            quant_type=args.quant_type,
            dry_run=args.dry_run,
            llama_cpp_dir=args.llama_cpp_dir,
        )

        # --- Step 4 ---
        if not args.skip_eval:
            evaluate_quality(output_path=output_path, dry_run=args.dry_run)
        else:
            logger.info("Step 4 — Skipped (--skip-eval)")

        # --- Cleanup ---
        cleanup_merged_dir(merged_dir, dry_run=args.dry_run)

        # Remove intermediate fp16 GGUF only after successful quantization
        if not args.dry_run and fp16_path.exists():
            logger.info("Cleanup — removing intermediate fp16 GGUF: %s", fp16_path)
            fp16_path.unlink()

        logger.info("=== Pipeline complete — output: %s ===", output_path)

    except (FileNotFoundError, subprocess.CalledProcessError, ImportError) as exc:
        logger.error("Pipeline failed: %s", exc)
        # Still attempt cleanup of the temp merged dir to avoid leaving gigabytes
        if merged_dir.exists() and not args.dry_run:
            logger.info("Cleaning up temporary merged dir after failure …")
            shutil.rmtree(merged_dir, ignore_errors=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter → convert to GGUF → quantize → eval.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="PEFT adapter directory produced by train_lumi.py",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model ID or local path for the base model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the final GGUF (created if absent)",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Filename for the final quantized GGUF",
    )
    parser.add_argument(
        "--quant-type",
        default=DEFAULT_QUANT_TYPE,
        help="llama.cpp quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0)",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=DEFAULT_LLAMA_CPP_DIR,
        help=(
            "Path to the llama.cpp repository root "
            "(used to locate convert_hf_to_gguf.py). "
            "Omit if 'llama-convert' is already on PATH."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the persona evaluation step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log all steps without executing anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
