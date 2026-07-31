"""Abliterate the Phi-prior identity direction from a merged Lumi HF model.

Bakes the persona-vector subtraction permanently into the model weights so no
runtime cvec is needed.  Targets the output-writing matrices (o_proj,
down_proj) in each transformer layer — the same matrices that the llama.cpp
cvec modifies at inference time.

Math
----
For each layer l in [layer_start, layer_end]:
    d_hat = directions[l] / ||directions[l]||         # unit Phi-prior direction
    For each output-writing weight W (shape [d_model, d_in]):
        W -= alpha * outer(d_hat, d_hat @ W)
    # This removes the component of W's output that lies along d_hat.

This is equivalent to prepending the projection  h -> h - (h·d_hat)*d_hat
to the residual stream at each layer, which matches what the cvec does.

Usage
-----
    # Full pipeline — abliterate merged v2.1 and quantize:
    uv run --extra qlora python scripts/abliterate_phi_prior.py \\
        --merged-dir   models/lumi-merged-v2.1 \\
        --vector       models/persona_vectors/phi_prior_v1.pt \\
        --output-name  lumi-phi35-v2.1-abliterated-Q5_K_M.gguf \\
        --llama-cpp-dir /home/hsozer/Dev/llama.cpp

    # Dry run (no weight changes, no GGUF written):
    uv run --extra qlora python scripts/abliterate_phi_prior.py --dry-run

    # Tune alpha (default 1.0 = full direction removal; try 0.7 for partial):
    uv run --extra qlora python scripts/abliterate_phi_prior.py \\
        --merged-dir models/lumi-merged-v2.1 \\
        --vector     models/persona_vectors/phi_prior_v1.pt \\
        --alpha 0.7 \\
        --output-name lumi-phi35-v2.1-abliterated-a07-Q5_K_M.gguf \\
        --llama-cpp-dir /home/hsozer/Dev/llama.cpp
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    format="%(levelname)-5s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DEFAULT_MERGED_DIR = Path("models/lumi-merged-v2.1")
DEFAULT_VECTOR = Path("models/persona_vectors/phi_prior_v1.pt")
DEFAULT_OUTPUT_DIR = Path("models/llm")
DEFAULT_OUTPUT_NAME = "lumi-phi35-v2.1-abliterated-Q5_K_M.gguf"
DEFAULT_QUANT_TYPE = "Q5_K_M"
DEFAULT_LLAMA_CPP_DIR = ""
DEFAULT_ALPHA = 1.0
DEFAULT_LAYER_START = 12
DEFAULT_LAYER_END = 28


# ---------------------------------------------------------------------------
# Step 1 — Abliterate output-writing weights
# ---------------------------------------------------------------------------


def abliterate(
    merged_dir: Path,
    vector_path: Path,
    output_dir: Path,
    layer_start: int,
    layer_end: int,
    alpha: float,
    dry_run: bool,
) -> Path:
    """Apply direction-projection abliteration and save modified model."""
    logger.info("Step 1 — Abliterating Phi-prior direction from merged model")
    logger.info("  merged dir  : %s", merged_dir)
    logger.info("  vector      : %s", vector_path)
    logger.info("  layers      : [%d, %d] (inclusive)", layer_start, layer_end)
    logger.info("  alpha       : %.2f", alpha)
    logger.info("  output dir  : %s", output_dir)

    if dry_run:
        logger.info("  [dry-run] skipping abliteration")
        return output_dir

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    t0 = time.monotonic()

    payload = torch.load(str(vector_path), map_location="cpu", weights_only=True)
    directions: torch.Tensor = payload["directions"]  # [n_layers, d_model]
    n_layers, d_model = directions.shape
    logger.info("  directions  : %d layers × %d dims", n_layers, d_model)

    if layer_start < 0 or layer_end >= n_layers or layer_start > layer_end:
        raise ValueError(
            f"Layer range [{layer_start}, {layer_end}] invalid for {n_layers}-layer model."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("  Device      : %s", device)
    if device == "cuda":
        free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        logger.info("  VRAM free   : %.1f GB", free_vram / 1024**3)

    logger.info("  Loading model in fp16 …")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    total_modified = 0
    # d_hat computed in fp32 for normalization precision, then cast to fp16 on device.
    alpha_16 = torch.tensor(alpha, dtype=torch.float16, device=device)

    for layer_idx in range(layer_start, layer_end + 1):
        d = directions[layer_idx].to(torch.float32)
        norm = d.norm()
        if norm < 1e-8:
            logger.warning(
                "  layer %2d: direction norm near-zero (%.2e) — skipping",
                layer_idx, norm.item(),
            )
            continue
        d_hat_16 = (d / norm).to(dtype=torch.float16, device=device)

        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        mlp = layer.mlp

        # Output-writing matrices [d_model, d_in]: remove what they WRITE along d_hat.
        # Formula: W -= alpha * outer(d_hat, d_hat @ W)
        for W_param in [attn.o_proj.weight, mlp.down_proj.weight]:
            W_param.data -= alpha_16 * torch.outer(d_hat_16, d_hat_16 @ W_param.data)
            total_modified += 1

        # Input-reading matrices [d_out, d_model]: remove what they READ from d_hat.
        # Formula: W -= alpha * outer(W @ d_hat, d_hat)
        # Phi-3.5 fuses QKV into qkv_proj and gate+up into gate_up_proj.
        for W_param in [attn.qkv_proj.weight, mlp.gate_up_proj.weight]:
            W_param.data -= alpha_16 * torch.outer(W_param.data @ d_hat_16, d_hat_16)
            total_modified += 1

        if (layer_idx - layer_start) % 4 == 0:
            logger.info("  layer %2d/%2d done", layer_idx, layer_end)

    logger.info(
        "  Modified %d weight matrices across %d layers",
        total_modified, layer_end - layer_start + 1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Saving abliterated model to %s …", output_dir)

    # Workaround for transformers ≥5.5 save_pretrained bug with Phi-3 tied weights.
    try:
        from transformers import modeling_utils as _mu  # type: ignore[import]
        _orig = _mu._get_tied_weight_keys

        def _safe_get_tied_weight_keys(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return _orig(*args, **kwargs)
            except (AttributeError, TypeError, KeyError):
                return []

        _mu._get_tied_weight_keys = _safe_get_tied_weight_keys
    except (ImportError, AttributeError):
        pass

    model.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(output_dir))

    # copy_to_gguf.py needs tokenizer.model for Phi-3 family
    src_spm = merged_dir / "tokenizer.model"
    if src_spm.is_file():
        shutil.copy2(src_spm, output_dir / "tokenizer.model")
        logger.info("  Copied tokenizer.model → %s", output_dir / "tokenizer.model")

    elapsed = time.monotonic() - t0
    logger.info("  Abliteration complete in %.1f s", elapsed)
    return output_dir


# ---------------------------------------------------------------------------
# Step 2 — Convert to GGUF (fp16 intermediate)
# ---------------------------------------------------------------------------


def _find_convert_script(llama_cpp_dir: str) -> str:
    if llama_cpp_dir:
        candidate = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found in {llama_cpp_dir}")
    try:
        import llama_cpp  # type: ignore[import]
        pkg_dir = Path(llama_cpp.__file__).parent
        candidate = pkg_dir / "convert_hf_to_gguf.py"
        if candidate.is_file():
            return str(candidate)
    except ImportError:
        pass
    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Pass --llama-cpp-dir to locate it."
    )


def convert_to_gguf(
    model_dir: Path,
    fp16_gguf: Path,
    llama_cpp_dir: str,
    dry_run: bool,
) -> None:
    logger.info("Step 2 — Converting to fp16 GGUF")
    logger.info("  model dir : %s", model_dir)
    logger.info("  output    : %s", fp16_gguf)

    if dry_run:
        logger.info("  [dry-run] skipping conversion")
        return

    convert_script = _find_convert_script(llama_cpp_dir)
    cmd = [
        sys.executable, convert_script,
        str(model_dir),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ]
    logger.info("  Running: %s", " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    for line in result.stdout.splitlines():
        logger.info("    [convert] %s", line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            logger.error("    [convert] %s", line)
        raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode})")
    logger.info("  Conversion complete in %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Step 3 — Quantize
# ---------------------------------------------------------------------------


def _find_llama_quantize(llama_cpp_dir: str) -> str:
    if llama_cpp_dir:
        candidate = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
        if candidate.is_file():
            return str(candidate)
    for name in ("llama-quantize", "quantize"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "llama-quantize not found. Build llama.cpp and pass --llama-cpp-dir."
    )


def quantize(
    fp16_gguf: Path,
    output_gguf: Path,
    quant_type: str,
    llama_cpp_dir: str,
    dry_run: bool,
) -> None:
    logger.info("Step 3 — Quantizing to %s", quant_type)
    logger.info("  input  : %s", fp16_gguf)
    logger.info("  output : %s", output_gguf)

    if dry_run:
        logger.info("  [dry-run] skipping quantization")
        return

    quantize_bin = _find_llama_quantize(llama_cpp_dir)
    cmd = [quantize_bin, str(fp16_gguf), str(output_gguf), quant_type]
    logger.info("  Running: %s", " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    for line in result.stdout.splitlines():
        logger.info("    [quantize] %s", line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            logger.error("    [quantize] %s", line)
        raise RuntimeError(f"llama-quantize failed (exit {result.returncode})")
    logger.info("  Quantization complete in %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Step 4 — Identity probe
# ---------------------------------------------------------------------------


def run_identity_probe(output_gguf: Path, probe_output: Path, dry_run: bool) -> None:
    """Run the identity probe using the shipping-candidate sampling config.

    Sampling params are taken from ``eval_identity.SHIPPING_*`` constants,
    which mirror production ``config.yaml`` values.  Passing them explicitly
    keeps the probe aligned with the config the shipped model uses and ensures
    the probe output JSON is a reproducible artifact.
    """
    logger.info("Step 4 — Running identity probe (41 prompts)")
    if dry_run:
        logger.info("  [dry-run] skipping probe")
        return

    # Import shipping constants without importing the full src stack — the
    # eval_identity script is a sibling file, importable via sys.path.
    _scripts_dir = Path(__file__).parent
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("eval_identity", _scripts_dir / "eval_identity.py")
    _ei = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_ei)  # type: ignore[union-attr]

    probe_script = _scripts_dir / "eval_identity.py"
    cmd = [
        sys.executable, str(probe_script),
        "--live",
        "--model-path", str(output_gguf),
        "--output", str(probe_output),
        # Shipping-candidate sampling — aligned with production config.yaml and
        # eval_identity.SHIPPING_* constants so the post-abliteration probe
        # uses the same sampler config as the shipped metric.
        "--temperature",    str(_ei.SHIPPING_TEMPERATURE),
        "--top-p",          str(_ei.SHIPPING_TOP_P),
        "--top-k",          str(_ei.SHIPPING_TOP_K),
        "--min-p",          str(_ei.SHIPPING_MIN_P),
        "--repeat-penalty", str(_ei.SHIPPING_REPEAT_PENALTY),
        "--max-tokens",     str(_ei.SHIPPING_MAX_TOKENS),
    ]
    logger.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.warning("  identity probe exited %d — check %s", result.returncode, probe_output)
    else:
        logger.info("  Probe complete — results at %s", probe_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Abliterate Phi-prior direction from merged Lumi model weights.",
    )
    p.add_argument(
        "--merged-dir", type=Path, default=DEFAULT_MERGED_DIR,
        help="HF model directory with merged LoRA weights (default: %(default)s).",
    )
    p.add_argument(
        "--vector", type=Path, default=DEFAULT_VECTOR,
        help="Phi-prior direction .pt file (default: %(default)s).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for the final GGUF (default: %(default)s).",
    )
    p.add_argument(
        "--output-name", type=str, default=DEFAULT_OUTPUT_NAME,
        help="GGUF filename (default: %(default)s).",
    )
    p.add_argument(
        "--quant-type", type=str, default=DEFAULT_QUANT_TYPE,
        help="llama-quantize quant type (default: %(default)s).",
    )
    p.add_argument(
        "--llama-cpp-dir", type=str, default=DEFAULT_LLAMA_CPP_DIR,
        help="Path to llama.cpp repo root (locates convert_hf_to_gguf.py and llama-quantize).",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Abliteration strength: 1.0 = full direction removal (default: %(default)s).",
    )
    p.add_argument(
        "--layer-start", type=int, default=DEFAULT_LAYER_START,
        help="First transformer layer to abliterate (inclusive, 0-indexed, default: %(default)s).",
    )
    p.add_argument(
        "--layer-end", type=int, default=DEFAULT_LAYER_END,
        help="Last transformer layer to abliterate (inclusive, 0-indexed, default: %(default)s).",
    )
    p.add_argument(
        "--skip-probe", action="store_true",
        help="Skip the identity probe step after quantization.",
    )
    p.add_argument(
        "--keep-hf", action="store_true",
        help="Keep the intermediate abliterated HF directory (useful for inspection).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log all steps without modifying any files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_gguf = args.output_dir / args.output_name
    fp16_gguf = args.output_dir / args.output_name.replace(".gguf", "-fp16.gguf")
    probe_output = Path("results") / args.output_name.replace(".gguf", "-identity.json")

    alpha_tag = f"a{int(args.alpha * 10):02d}"
    layer_tag = f"l{args.layer_start}-{args.layer_end}"
    abliterated_dir = Path(
        tempfile.mkdtemp(prefix=f"lumi_abliterated_{alpha_tag}_{layer_tag}_")
    )

    logger.info("=== Lumi Abliteration Pipeline ===")
    logger.info("  merged dir   : %s", args.merged_dir)
    logger.info("  vector       : %s", args.vector)
    logger.info("  output       : %s", output_gguf)
    logger.info("  alpha        : %.2f", args.alpha)
    logger.info("  layers       : [%d, %d]", args.layer_start, args.layer_end)
    logger.info("  dry run      : %s", args.dry_run)

    try:
        abliterate(
            merged_dir=args.merged_dir,
            vector_path=args.vector,
            output_dir=abliterated_dir,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            alpha=args.alpha,
            dry_run=args.dry_run,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        convert_to_gguf(
            model_dir=abliterated_dir,
            fp16_gguf=fp16_gguf,
            llama_cpp_dir=args.llama_cpp_dir,
            dry_run=args.dry_run,
        )

        quantize(
            fp16_gguf=fp16_gguf,
            output_gguf=output_gguf,
            quant_type=args.quant_type,
            llama_cpp_dir=args.llama_cpp_dir,
            dry_run=args.dry_run,
        )

        if not args.skip_probe:
            Path("results").mkdir(exist_ok=True)
            run_identity_probe(output_gguf, probe_output, dry_run=args.dry_run)

    finally:
        if not args.dry_run and fp16_gguf.exists():
            logger.info("Cleanup — removing fp16 GGUF: %s", fp16_gguf)
            fp16_gguf.unlink()

        if not args.dry_run and abliterated_dir.exists() and not args.keep_hf:
            logger.info("Cleanup — removing abliterated HF dir: %s", abliterated_dir)
            shutil.rmtree(abliterated_dir)

    logger.info("=== Pipeline complete — output: %s ===", output_gguf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
