"""DoRA fine-tuning on top of Lumi SFT adapter — Tier-2 persona Phi-prior suppression.

Tier 1.B (IPO DPO with standard LoRA, 196 pairs, attn+MLP targets) achieved zero
movement on capability-denial prompts (8% → 8% Lumi).  This script tries the same
pipeline with Weight-Decomposed LoRA (DoRA), which decomposes each weight matrix
into a magnitude vector and a direction matrix, allowing the optimizer to update
them independently.  If the Phi-prior is encoded primarily in the *direction* of
key weight matrices, DoRA may succeed where vanilla LoRA's coupled magnitude+direction
update could not.

DoRA is a drop-in replacement for LoRA in PEFT ≥0.9.0 via `use_dora=True`.
Trained adapters can be merged into the base model and exported to GGUF — same
production path as the LoRA pipeline.

Architecture
------------
Base: lumi-merged-v2.1 (fp16 HF model, SFT already baked in)
  → loaded in 4-bit QDoRA
  → new DoRA adapter (attn+MLP, no lm_head) — same targets as v2.2 LoRA
  → DPOTrainer: IPO loss, LR 5e-6, 2 epochs

Usage
-----
    uv run python scripts/train_dora.py

After training, quantize with:
    uv run python scripts/merge_and_quantize.py \\
        --base-model    models/lumi-merged-v2.1 \\
        --adapter-dir   models/lumi-dora-v2.2 \\
        --output-name   lumi-phi35-v2.2-dora-Q5_K_M.gguf \\
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

# DoRA Tier-2: same attention/MLP targets as v2.2 IPO-LoRA.
# lm_head excluded — capability-denial decision is upstream of token logits.
DORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_MERGED_DIR    = Path("models/lumi-merged-v2.1")
DEFAULT_DATASET       = Path("data/finetune/dpo_v2.2.jsonl")
DEFAULT_OUTPUT_DIR    = Path("models/lumi-dora-v2.2")


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


def make_dora_config(rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, TaskType  # type: ignore[import]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=DORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_dora(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    merged_dir = args.merged_dir

    # ── Load tokenizer ───────────────────────────────────────────────────────
    logger.info("Loading tokenizer from %s", merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load merged SFT model in 4-bit ───────────────────────────────────────
    logger.info("Loading merged SFT model in 4-bit QDoRA")
    bnb_config = make_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # ── Attach DoRA adapter ──────────────────────────────────────────────────
    logger.info("Adding DoRA adapter (rank=%d, attn+MLP, no lm_head)", args.lora_rank)
    dora_cfg = make_dora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, dora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("Loading DPO dataset from %s", args.dataset)
    dataset = load_dpo_dataset(args.dataset)
    logger.info("  %d preference pairs loaded", len(dataset))

    # ── DPO / IPO config ─────────────────────────────────────────────────────
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
        loss_type="ipo",
        beta=args.beta,
        max_length=args.max_length,
    )

    # ref_model=None: DPOTrainer uses merged SFT model (LoRA disabled) as reference.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DoRA training …")
    trainer.train()

    logger.info("Saving DoRA adapter to %s", args.output_dir)
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DoRA fine-tuning for Lumi persona Phi-prior suppression (Tier 2).",
    )
    parser.add_argument("--merged-dir",      type=Path, default=DEFAULT_MERGED_DIR,
                        help="Merged SFT fp16 model (DPO reference + policy base).")
    parser.add_argument("--dataset",         type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir",      type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs",          type=int,   default=2)
    parser.add_argument("--batch-size",      type=int,   default=2)
    parser.add_argument("--lora-rank",       type=int,   default=32)
    parser.add_argument("--lora-alpha",      type=int,   default=32)
    parser.add_argument("--lora-dropout",    type=float, default=0.05)
    parser.add_argument("--learning-rate",   type=float, default=5e-6)
    parser.add_argument("--beta",            type=float, default=0.1)
    parser.add_argument("--max-length",      type=int,   default=768)
    parser.add_argument("--save-total-limit",type=int,   default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        train_dora(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: uv sync --extra qlora")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
