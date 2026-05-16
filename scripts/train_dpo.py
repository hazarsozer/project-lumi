"""DPO fine-tuning on top of Lumi SFT adapter — persona Phi-prior suppression.

Directly targets the failure pattern observed in the v2.1 SFT eval:
  - "I'm Phi, an AI language model" on capability-denial prompts (#5/#6)
  - "as an AI language model" on knowledge-limit and memory prompts (#3/#17/#18)

Architecture
------------
SFT base: phi-3.5-mini + lumi-lora-v2.1 (merged into fp16 HuggingFace model)
  → loaded in 4-bit as DPO base
  → new LoRA adapter added (same attention/MLP targets + lm_head)
  → DPOTrainer: reference = SFT merged model (ref_model=None)
                policy    = SFT merged model + trainable DPO LoRA

Adding lm_head to target modules gives the optimizer a direct handle on
token probabilities at the output layer — where the Phi prior leaks the
"Phi / AI language model" tokens despite the SFT adapter.

Usage
-----
    uv run python scripts/train_dpo.py \\
        --base-model    models/llm/checkpoints/phi-3.5-mini \\
        --sft-adapter   models/lumi-lora-v2.1 \\
        --dataset       data/finetune/dpo_v2.1.jsonl \\
        --output-dir    models/lumi-dpo-v2.1 \\
        --merged-dir    models/lumi-merged-v2.1

After training, quantize with:
    uv run python scripts/merge_and_quantize.py \\
        --base-model    models/lumi-merged-v2.1 \\
        --adapter-dir   models/lumi-dpo-v2.1 \\
        --output-name   lumi-phi35-v2.1-dpo-Q5_K_M.gguf \\
        --quant-type    Q5_K_M \\
        --llama-cpp-dir /home/hsozer/Dev/llama.cpp
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# v2.2: attention + MLP only — lm_head excluded.
# v2.1 DPO targeted lm_head exclusively and hit 100% training accuracy with 0%
# inference movement (textbook 3D-Properties pathology on small datasets).
# The capability-denial decision is upstream of token logits; hitting lm_head
# alone fights the wrong layer.  IPO loss (below) regularises against overfit.
DPO_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_BASE_MODEL    = "models/llm/checkpoints/phi-3.5-mini"
DEFAULT_SFT_ADAPTER   = "models/lumi-lora-v2.1"
DEFAULT_DATASET       = Path("data/finetune/dpo_v2.2.jsonl")
DEFAULT_OUTPUT_DIR    = Path("models/lumi-dpo-v2.2")
DEFAULT_MERGED_DIR    = Path("models/lumi-merged-v2.1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_dpo_dataset(path: Path):
    from datasets import Dataset  # type: ignore[import]

    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def make_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig  # type: ignore[import]

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def make_dpo_lora_config(rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, TaskType  # type: ignore[import]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=DPO_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Step 1 — Merge SFT adapter + base into a permanent fp16 HuggingFace model
# ---------------------------------------------------------------------------


def merge_sft(base_model: str, sft_adapter: str, merged_dir: Path) -> None:
    """Merge the SFT LoRA into the base model and save as fp16.

    This merged model becomes the DPO reference and the foundation the DPO
    adapter trains on top of.  Loading on CPU avoids consuming GPU VRAM during
    the merge step.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import PeftModel  # type: ignore[import]

    logger.info("Step 1 — Merging SFT adapter into base model (CPU, fp16)")
    logger.info("  base    : %s", base_model)
    logger.info("  adapter : %s", sft_adapter)
    logger.info("  output  : %s", merged_dir)

    # trust_remote_code=False: Phi-3's bundled modeling_phi3.py breaks
    # convert_hf_to_gguf.py; native transformers Phi3ForCausalLM is correct.
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, sft_adapter)
    logger.info("  Merging and unloading adapter …")
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Saving merged model …")

    # Workaround: transformers ≥5.5 bug where _get_tied_weight_keys() expects
    # dict entries but Phi-3 exposes lists.  Monkey-patch with a safe wrapper.
    try:
        from transformers import modeling_utils as _mu  # type: ignore[import]
        _orig = _mu._get_tied_weight_keys

        def _safe(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return _orig(*args, **kwargs)
            except (AttributeError, TypeError, KeyError):
                return []

        _mu._get_tied_weight_keys = _safe
    except (ImportError, AttributeError):
        pass

    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del model, base
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("  Merge complete.")


# ---------------------------------------------------------------------------
# Step 2 — DPO training
# ---------------------------------------------------------------------------


def train_dpo(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # ── Merge SFT (skip if merged_dir already exists with weights) ──────────
    merged_dir = args.merged_dir
    adapter_config = merged_dir / "adapter_config.json"
    model_files = list(merged_dir.glob("*.safetensors")) + list(merged_dir.glob("*.bin"))
    if model_files and not adapter_config.exists():
        logger.info("Merged model already exists at %s — skipping merge step.", merged_dir)
    else:
        merge_sft(args.base_model, str(args.sft_adapter), merged_dir)

    # ── Load tokenizer ───────────────────────────────────────────────────────
    logger.info("Loading tokenizer from %s", merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load merged SFT model in 4-bit ───────────────────────────────────────
    logger.info("Loading merged SFT model in 4-bit QLoRA")
    bnb_config = make_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # ── Add DPO LoRA adapter (attention + MLP, no lm_head) ──────────────────
    logger.info("Adding DPO LoRA adapter (rank=%d, attn+MLP targets, no lm_head)", args.lora_rank)
    dpo_lora = make_dpo_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, dpo_lora)
    model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("Loading DPO dataset from %s", args.dataset)
    dataset = load_dpo_dataset(args.dataset)
    logger.info("  %d preference pairs loaded", len(dataset))

    # ── DPO config ───────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=args.save_total_limit,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        report_to="none",
        # DPO-specific — v2.2: IPO loss replaces standard DPO to regularise
        # against 3D-Properties overfit on small datasets (beta=0.1 with IPO
        # acts like a quadratic constraint rather than a logistic one, preventing
        # the policy from collapsing to the reference even with 196 pairs).
        loss_type="ipo",
        beta=args.beta,
        max_length=args.max_length,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    # ref_model=None: DPOTrainer uses the model with the LoRA adapter DISABLED
    # as the reference — i.e., the merged SFT model.  This is correct: we want
    # the reference to be SFT behaviour, and the policy to learn to deviate
    # toward Lumi voice on the failing prompts.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training …")
    trainer.train()

    logger.info("Saving DPO adapter to %s", args.output_dir)
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO fine-tuning for Lumi persona Phi-prior suppression.",
    )
    parser.add_argument("--base-model",  default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-adapter", type=Path, default=Path(DEFAULT_SFT_ADAPTER))
    parser.add_argument("--dataset",     type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-dir",  type=Path, default=DEFAULT_MERGED_DIR,
                        help="Where to save/load the merged SFT fp16 model used as "
                             "DPO reference and policy base.")
    parser.add_argument("--epochs",           type=int,   default=2)
    parser.add_argument("--batch-size",       type=int,   default=2)
    parser.add_argument("--lora-rank",        type=int,   default=32)
    parser.add_argument("--lora-alpha",       type=int,   default=32)
    parser.add_argument("--lora-dropout",     type=float, default=0.05)
    parser.add_argument("--learning-rate",    type=float, default=5e-6,
                        help="DPO LR — v2.2: 5e-6 (10x v2.1). IPO regularises against blowup.")
    parser.add_argument("--beta",             type=float, default=0.1,
                        help="DPO beta — controls how far from reference the "
                             "policy can diverge (default 0.1).")
    parser.add_argument("--max-length",       type=int,   default=768)
    parser.add_argument("--save-total-limit", type=int,   default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        train_dpo(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: uv sync --extra qlora")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
