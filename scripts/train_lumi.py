"""QLoRA fine-tuning pipeline for Project Lumi.

Fine-tunes a HuggingFace causal-LM with 4-bit quantisation (bitsandbytes)
and LoRA adapters (PEFT) using TRL's SFTTrainer on the synthetic ChatML
dataset produced by ``scripts/synth_dataset.py``.

Requirements
------------
    uv sync --extra qlora          # installs transformers, peft, trl, bitsandbytes, accelerate

Usage
-----
    # Quick smoke run (1 step, no GPU required if you have MPS/CPU):
    uv run python scripts/train_lumi.py --dry-run

    # Full training on RTX 4070 12 GB:
    uv run python scripts/train_lumi.py \\
        --base-model microsoft/Phi-3.5-mini-instruct \\
        --dataset    data/finetune/synthetic_v0.jsonl \\
        --output-dir models/lumi-lora-v1 \\
        --epochs 3 \\
        --batch-size 4

Output
------
A LoRA adapter directory at ``--output-dir`` that can be loaded with:

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, output_dir)

The directory can later be merged into the base weights for GGUF export with
``scripts/merge_and_quantize.py`` (Phase 10).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_DATASET = Path("data/finetune/synthetic_v0.jsonl")
DEFAULT_OUTPUT_DIR = Path("models/lumi-lora-v1")
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def messages_to_chatml(messages: list[dict]) -> str:
    """Convert a messages list to a ChatML-formatted string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def prepare_hf_dataset(records: list[dict]):
    """Convert JSONL records to a HuggingFace Dataset with a 'text' column."""
    from datasets import Dataset  # type: ignore[import]

    texts = [messages_to_chatml(r["messages"]) for r in records]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# QLoRA config
# ---------------------------------------------------------------------------

def make_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig  # type: ignore[import]

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def make_lora_config(rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, TaskType  # type: ignore[import]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore[import]
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    from trl import SFTTrainer  # type: ignore[import]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    logger.info("Loading dataset from %s", args.dataset)
    records = load_jsonl(args.dataset)
    logger.info("  %d examples loaded", len(records))
    dataset = prepare_hf_dataset(records)

    logger.info("Loading tokenizer for %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading base model (4-bit QLoRA)")
    bnb_config = make_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, make_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout))
    model.print_trainable_parameters()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        max_steps=1 if args.dry_run else -1,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=True,
    )

    logger.info("Starting training%s", " (dry run — 1 step)" if args.dry_run else "")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Lumi persona alignment.",
    )
    parser.add_argument("--base-model",   default=DEFAULT_BASE_MODEL,  help="HuggingFace model ID or local path")
    parser.add_argument("--dataset",      type=Path, default=DEFAULT_DATASET, help="JSONL training dataset")
    parser.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR, help="LoRA adapter output directory")
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-len",  type=int,   default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-rank",    type=int,   default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha",   type=int,   default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--dry-run",      action="store_true", help="Run 1 step only to verify setup")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run: uv run python scripts/synth_dataset.py")
        return 1

    try:
        train(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install QLoRA deps with: uv sync --extra qlora")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
