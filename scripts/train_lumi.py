"""QLoRA fine-tuning pipeline for Project Lumi.

Fine-tunes a HuggingFace causal-LM with 4-bit quantisation (bitsandbytes)
and LoRA adapters (PEFT) using TRL's SFTTrainer (≥1.0) on the synthetic
dataset produced by ``scripts/synth_dataset.py``.

Key changes in Ring 2 (2026-05-04):
  - Dataset format: ``{prompt, completion}`` columns (was ``{text}``) so TRL ≥1.0
    auto-detects prompt-completion shape and applies completion-only loss.
  - Training format: Uses ``tokenizer.apply_chat_template`` for Phi-3 native format
    instead of hand-rolled ChatML. Critical: training and inference must use the
    same format or token-level gibberish results.
  - Config migration: ``TrainingArguments`` → ``SFTConfig``, ``tokenizer=`` →
    ``processing_class=``, ``torch_dtype=`` → ``dtype=``, ``warmup_ratio`` →
    ``warmup_steps=10``.
  - New SFTConfig params: ``completion_only_loss=True``, ``packing=False``,
    ``gradient_checkpointing=True`` + ``gradient_checkpointing_kwargs``.
  - Learning rate: lowered ``2e-4`` → ``1e-4`` to stabilize completion-only loss.

Requirements
------------
    uv sync --extra qlora          # installs transformers, peft, trl, bitsandbytes, accelerate

Usage
-----
    # Quick smoke run (1 step, no GPU required if you have MPS/CPU):
    uv run python scripts/train_lumi.py --dry-run

    # Full training on RTX 4070 12 GB (reproducible Ring 2 command):
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    uv run python scripts/train_lumi.py \\
        --base-model models/llm/checkpoints/phi-3.5-mini \\
        --dataset    data/finetune/synthetic_v1.jsonl \\
        --output-dir models/lumi-lora-v1 \\
        --epochs 3 \\
        --batch-size 2

Output
------
A LoRA adapter directory at ``--output-dir`` containing:
  - adapter_config.json, adapter_model.safetensors (PEFT LoRA weights)
  - tokenizer.json, tokenizer_config.json, special_tokens_map.json
  - (tokenizer.model copied by merge_and_quantize.py if SentencePiece)

The directory can later be merged into the base weights for GGUF export with
``scripts/merge_and_quantize.py`` (Ring 2 onwards).
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


def messages_to_chatml(messages: list[dict], tokenizer=None) -> str:
    """Convert a messages list to a chat-formatted string.

    When *tokenizer* is provided, uses ``tokenizer.apply_chat_template`` so the
    training format matches the model's native template exactly.  This is
    critical: training in one format and serving in another produces token-level
    gibberish at inference time (the model has never seen the inference format).
    Falls back to legacy ChatML when tokenizer is None (only for tests).
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback: hand-rolled ChatML (only safe for ChatML-native models).
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def prepare_hf_dataset(records: list[dict], tokenizer):
    """Convert JSONL records to a HuggingFace prompt+completion Dataset.

    Each output record has two columns:
      - ``prompt``     : system + user turns formatted via ``apply_chat_template``
                         with ``add_generation_prompt=True`` (ends in
                         ``<|assistant|>\\n`` — exactly the inference prompt
                         shape from ``src/llm/prompt_engine.py``).
      - ``completion`` : assistant content followed by Phi-3 end markers
                         (``<|end|>\\n<|endoftext|>``).

    TRL ≥1.0's SFTTrainer auto-detects prompt+completion datasets and applies
    loss only over the completion tokens.  This is the fix for the gibberish
    we saw when the dataset had a single ``text`` column: in that mode TRL
    falls back to language-modeling loss (loss on every token), and with
    ``packing=True`` the model is trained to predict next-sample system
    prompts at packed boundaries — meaningless gradient that destroys
    generation behavior.
    """
    from datasets import Dataset  # type: ignore[import]

    prompts: list[str] = []
    completions: list[str] = []
    eos = tokenizer.eos_token  # "<|endoftext|>" for Phi-3.5
    for r in records:
        msgs = r["messages"]
        asst_idx = next(
            (i for i, m in enumerate(msgs) if m["role"] == "assistant"), None
        )
        if asst_idx is None:
            continue
        prompt = tokenizer.apply_chat_template(
            msgs[:asst_idx],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Match what apply_chat_template emits after the assistant turn:
        #   "{content}<|end|>\n<|endoftext|>"
        completion = msgs[asst_idx]["content"] + "<|end|>\n" + eos
        prompts.append(prompt)
        completions.append(completion)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


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
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    logger.info("Loading tokenizer for %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading dataset from %s", args.dataset)
    records = load_jsonl(args.dataset)
    logger.info("  %d examples loaded", len(records))
    # Pass the tokenizer so prepare_hf_dataset uses Phi-3's native chat template
    # (matches what prompt_engine.py produces at inference time).
    dataset = prepare_hf_dataset(records, tokenizer=tokenizer)

    logger.info("Loading base model (4-bit QLoRA)")
    bnb_config = make_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, make_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout))
    model.print_trainable_parameters()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # SFTConfig (trl ≥1.0) absorbs both TrainingArguments fields and SFT-specific
    # fields (dataset_text_field, max_length, packing) that previously lived as
    # SFTTrainer kwargs.
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        # Lowered from 2e-4 → 1e-4: with completion-only loss on a small
        # dataset the gradient signal is much more focused; 2e-4 risks
        # destabilising the LoRA in the first dozen steps.
        learning_rate=1e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        max_steps=1 if args.dry_run else -1,
        # Activation-memory savings: required to fit Phi-3.5-mini QLoRA on a
        # 12 GB consumer GPU.  Adds ~1.5–2× step time but ~3–4× memory headroom.
        # use_reentrant=False is required for PEFT compatibility.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT-specific fields:
        # * Dataset is prompt+completion (set up in prepare_hf_dataset); TRL
        #   auto-detects this shape and would set completion_only_loss=True
        #   even if we did not pass it explicitly — but we do, for clarity.
        # * packing=False: with prompt+completion + completion-only loss,
        #   packing across samples would still leak system+user tokens of
        #   the next sample into the loss-relevant span at each boundary.
        #   Trade ~1.5× wall time for correct gradient.
        max_length=args.max_seq_len,
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # was `tokenizer=` in trl <1.0
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
