# Training & LoRA Pipeline Codemap

**Last Updated:** 2026-05-04 (Ring 2 closure)
**Status:** Complete & tested (eval evidence: results/eval_*.json, both 75% on regex criteria)

## Overview

Project Lumi's persona alignment pipeline: dataset synthesis → fine-tuning → GGUF export → live evaluation.

## Architecture

```
Synthetic Dataset          TRL SFTTrainer (≥1.0)        Merge & Quantize        Evaluation Harness
(synthetic_v1.jsonl)  →    (4-bit QLoRA + gradient      (GGUF export via    →   (offline/live)
                            checkpointing)               llama.cpp)
                            ↓
                         models/lumi-lora-v1/
                         (gitignored, produced only)
                            ↓
                         models/lumi-phi35-v1-Q4_K_M.gguf
                         (gitignored, final artifact)
```

## Key Modules

| Module | File | Purpose | Entry Point |
|--------|------|---------|-------------|
| **Dataset prep** | `scripts/train_lumi.py` | Load JSONL, convert to prompt+completion format via `tokenizer.apply_chat_template` | `prepare_hf_dataset()` |
| **QLoRA training** | `scripts/train_lumi.py` | 4-bit QLoRA fine-tuning on Phi-3.5-mini, completion-only loss | `train()` |
| **LoRA merge** | `scripts/merge_and_quantize.py` | Merge LoRA weights into base, handle Phi-3 tokenizer quirks | `merge_lora()` |
| **GGUF conversion** | `scripts/merge_and_quantize.py` | Convert merged HF checkpoint to fp16 GGUF via llama.cpp | `convert_to_gguf()` |
| **GGUF quantization** | `scripts/merge_and_quantize.py` | Quantize to Q4_K_M (or user-specified type) | `quantize_gguf()` |
| **Persona evaluation** | `scripts/eval_persona.py` | Assess response quality on 23 prompts across 9 categories | `eval()` |

## Ring 2 Changes (Critical Fixes)

### 1. Dataset Format: Text → Prompt+Completion

**Before (gibberish):**
```python
# Single "text" column — TRL <1.0 applies language-modeling loss to entire sample
Dataset.from_dict({"text": [messages_to_chatml(msgs) for msgs in data]})
```

**After (correct):**
```python
# Two columns: prompt (context) + completion (target) — TRL ≥1.0 applies
# loss ONLY to completion tokens. Critical for small datasets.
def prepare_hf_dataset(records, tokenizer):
    prompts = []
    completions = []
    for r in records:
        msgs = r["messages"]
        asst_idx = ...  # index of first assistant turn
        prompt = tokenizer.apply_chat_template(msgs[:asst_idx], add_generation_prompt=True)
        completion = msgs[asst_idx]["content"] + "<|end|>\n" + tokenizer.eos_token
        prompts.append(prompt)
        completions.append(completion)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})
```

**Why:** With single "text" column, packing=True leaks system+user tokens of *next sample* into loss, teaching the model meaningless gradients. Prompt+completion prevents this.

### 2. Training Format: Hand-Rolled ChatML → Tokenizer's Native Format

**Before:**
```python
def messages_to_chatml(messages):
    # Hard-coded ChatML markers — may not match inference path
    parts = [f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" for msg in messages]
    return "\n".join(parts)
```

**After:**
```python
def messages_to_chatml(messages, tokenizer=None):
    if tokenizer and getattr(tokenizer, "chat_template", None):
        # Use Phi-3's native apply_chat_template — guarantees format matches inference
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback for tests; production always uses tokenizer.apply_chat_template
        ...
```

**Why:** Training in format X, serving in format Y = token-level gibberish. The tokenizer's `chat_template` is the source of truth.

### 3. TRL ≥1.0 Config: TrainingArguments → SFTConfig

**Migration table:**
| Old | New | Rationale |
|-----|-----|-----------|
| `TrainingArguments` | `SFTConfig` | SFT-specific config merged in TRL 1.0 |
| `tokenizer=...` | `processing_class=...` | Tokenizer now a "processing class" in transformers 5.x |
| `torch_dtype=torch.bfloat16` | `dtype=torch.bfloat16` | Simplified param name |
| `warmup_ratio=0.03` | `warmup_steps=10` | Fixed steps better for small datasets |
| `dataset_text_field='text'` | (removed) | Auto-detected from prompt+completion columns |
| (not set) | `completion_only_loss=True` | Explicit: apply loss only to completion span |
| (not set) | `packing=False` | Prevent boundary leakage with prompt+completion |

### 4. Dropout & Memory Savings

```python
training_args = SFTConfig(
    learning_rate=1e-4,           # lowered from 2e-4 (completion-only loss is sharper)
    gradient_checkpointing=True,  # ~1.5-2x slower but 3-4x less memory
    gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT compat
    # ...
)
```

### 5. Merge Pipeline: trust_remote_code=False + Tokenizer Copy

**Critical bug fix:**
```python
# WRONG — trust_remote_code=True:
tokenizer = AutoTokenizer.from_pretrained("phi-3.5-mini", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
# Phi-3.5's bundled modeling_phi3.py produces a state_dict that converts to CORRUPT GGUF
# (outputs "maggio maggio maggio..." regardless of quantization level)

# CORRECT:
tokenizer = AutoTokenizer.from_pretrained("phi-3.5-mini")  # no trust_remote_code
model = AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16, device_map="cpu")
# Native transformers Phi3ForCausalLM (4.39+) compatible; GGUF quantizes cleanly
```

**SentencePiece workaround:**
```python
# convert_hf_to_gguf.py requires tokenizer.model for Phi-3 family
# AutoTokenizer.save_pretrained only writes tokenizer.json, so copy the original:
src_spm = Path(base_model) / "tokenizer.model"
if src_spm.is_file():
    shutil.copy2(src_spm, merged_dir / "tokenizer.model")
```

### 6. Evaluation: Type-Safe Tool Call Checking

```python
# BEFORE — crashed on non-dict responses:
def criterion_tool_call_json_valid(response: str) -> bool:
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    return "tool" in data and "args" in data  # ← TypeError if data is int/str

# AFTER:
def criterion_tool_call_json_valid(response: str) -> bool:
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(data, dict) and "tool" in data and "args" in data  # ✓ safe
```

## External Dependencies

| Package | Version | Purpose | Ring 2 Usage |
|---------|---------|---------|-------------|
| `transformers` | 4.39+ | HF models & tokenizers | Native Phi3ForCausalLM, no trust_remote_code needed |
| `peft` | Latest | LoRA adapters | Merge weights, handle tied-weight monkey-patch |
| `trl` | 1.0+ | SFTTrainer/SFTConfig | Prompt+completion auto-detection, completion-only loss |
| `bitsandbytes` | Latest | 4-bit quantization | QLoRA inference-time memory savings |
| `torch` | 2.0+ | Deep learning | `torch.bfloat16`, gradient checkpointing |
| `datasets` | Latest | HF Datasets API | Load JSONL, build from_dict |
| `llama.cpp` | Latest (built locally) | GGUF conversion & quantization | convert_hf_to_gguf.py, llama-quantize binary |
| `gguf` | 0.10.0+ | GGUF utilities | (transitive from convert_hf_to_gguf.py) |

## CLI Interface

### `scripts/train_lumi.py`

```bash
usage: train_lumi.py [-h] [--base-model BASE_MODEL] [--dataset DATASET]
                     [--output-dir OUTPUT_DIR] [--epochs EPOCHS]
                     [--batch-size BATCH_SIZE] [--max-seq-len MAX_SEQ_LEN]
                     [--lora-rank LORA_RANK] [--lora-alpha LORA_ALPHA]
                     [--lora-dropout LORA_DROPOUT] [--dry-run]

Ring 2 reproducible command:
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train_lumi.py \
      --base-model models/llm/checkpoints/phi-3.5-mini \
      --dataset    data/finetune/synthetic_v1.jsonl \
      --output-dir models/lumi-lora-v1 \
      --epochs 3 --batch-size 2
```

### `scripts/merge_and_quantize.py`

```bash
usage: merge_and_quantize.py [-h] [--adapter-dir ADAPTER_DIR]
                             [--base-model BASE_MODEL]
                             [--output-dir OUTPUT_DIR]
                             [--output-name OUTPUT_NAME]
                             [--quant-type QUANT_TYPE]
                             [--llama-cpp-dir LLAMA_CPP_DIR]
                             [--skip-eval] [--dry-run]

Ring 2 reproducible command:
  uv run python scripts/merge_and_quantize.py \
      --adapter-dir   models/lumi-lora-v1 \
      --base-model    models/llm/checkpoints/phi-3.5-mini \
      --output-dir    models/llm \
      --output-name   lumi-phi35-v1-Q4_K_M.gguf \
      --quant-type    Q4_K_M \
      --llama-cpp-dir /home/hsozer/Dev/llama.cpp \
      --skip-eval
```

### `scripts/eval_persona.py`

```bash
usage: eval_persona.py [-h] [--output OUTPUT] [--live] [--dry-run]

# Offline (CI-safe, uses stub responses):
  uv run python scripts/eval_persona.py --output results/eval_baseline.json

# Live (against running Brain on ws://127.0.0.1:5556):
  uv run python scripts/eval_persona.py --output results/eval_lumi_v1.json --live
```

## Execution Flow

### Training Phase

1. Load tokenizer (Phi-3.5-mini)
2. Load JSONL dataset → parse messages
3. Convert to prompt+completion via `tokenizer.apply_chat_template`
4. Load base model in 4-bit QLoRA mode
5. Apply PEFT LoRA wrapper
6. Train for N epochs with SFTTrainer/SFTConfig
7. Save adapter + tokenizer → `models/lumi-lora-v1/`

**Output:** LoRA adapter ready for merge

### Merge Phase

1. Load base model (fp16, no trust_remote_code)
2. Attach PEFT adapter
3. Merge weights via `merge_and_unload()`
4. Monkey-patch `_get_tied_weight_keys` (transformers ≥5.5 workaround)
5. Save merged checkpoint → temp dir
6. Copy SentencePiece `tokenizer.model` if exists
7. (Ready for GGUF conversion)

**Output:** Merged HF checkpoint

### Quantization Phase

1. Call `convert_hf_to_gguf.py` → fp16 GGUF
2. Call `llama-quantize` → Q4_K_M GGUF (or other type)
3. Delete intermediate files
4. Optionally run eval harness

**Output:** Quantized GGUF at `models/llm/lumi-phi35-v1-Q4_K_M.gguf`

### Evaluation Phase

1. Load 23 prompts across 9 categories
2. Route each through either:
   - Stub responses (offline mode) — CI-safe, no model needed
   - Live WebSocket to Brain (live mode) — requires running Lumi process
3. Apply 13 criteria to each response
4. Aggregate results, compare to baseline if available
5. Write JSON results file

**Output:** Results file (results/eval_baseline.json, results/eval_lumi_v1.json, etc.)

## Data Files (Gitignored)

```
models/lumi-lora-v1/                    # LoRA adapter directory (produced by train_lumi.py)
  ├── adapter_config.json
  ├── adapter_model.safetensors
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── special_tokens_map.json

models/llm/lumi-phi35-v1-Q4_K_M.gguf   # Final GGUF artifact (produced by merge_and_quantize.py)
models/llm/tmp_merged_hf_*              # Intermediate (deleted after quantization)

results/eval_*.json                     # Evaluation output (produced by eval_persona.py)
```

All gitignored; produced by the pipeline, not committed.

## Testing & Verification

```bash
# Run all tests (including persona eval offline):
uv sync --extra dev
uv run pytest

# Live evaluation (requires running Brain):
cd /home/hsozer/Dev/Lumi
# In terminal 1: uv run python -m src.main
# In terminal 2: uv run python scripts/eval_persona.py --output results/eval_live.json --live

# View eval results:
cat results/eval_base_phi35.json
cat results/eval_lumi_v1.json
```

## Known Issues & Workarounds

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Bundled modeling_phi3.py** | "maggio maggio maggio..." output in GGUF | Drop `trust_remote_code=True`, use native Phi3ForCausalLM |
| **Transformers ≥5.5 tied_weight_keys** | TypeError during save_pretrained | Monkey-patch with signature-agnostic wrapper returning [] |
| **Missing tokenizer.model** | convert_hf_to_gguf.py fails on Phi-3 | Manually copy from base checkpoint |
| **eval_persona crashes on non-dict JSON** | TypeError when model returns raw number | Check isinstance(data, dict) before "tool" in data |

## Related Areas

- **Backend LLM inference:** `src/llm/model_loader.py`, `src/llm/prompt_engine.py` — load quantized GGUF, apply system+user prompts
- **Frontend state:** `app/src/state/useLumiState.ts` — display LLM responses in chat panel
- **RAG integration:** `src/rag/` — context-aware retrieval for long-answer categories

## Future Work (Beyond Ring 2)

- [ ] Streaming LoRA (DLoRA or similar) for multiple personae without retraining
- [ ] Per-category prompt engineering (e.g., separate prompts for tool-needing vs. honesty)
- [ ] Automated dataset expansion (synthetic_v2.jsonl with more categories)
- [ ] Quantization format comparison (GPTQ vs Q4_K_M vs Q5_K_M)
