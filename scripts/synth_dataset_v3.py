"""Teacher-distilled dataset generation for Lumi persona v2.3 (Tier 3).

Uses HFSteeredModel (lumi-merged-v2.1, phi_prior_v1.pt, alpha=8) as teacher.
At alpha=8 the steered model achieves 100% Lumi on capability-denial prompts
(eval_identity.py identity probe).  We sample it on a large, diverse pool of
B/C/D category prompts and record the outputs as gold training examples.

These examples are then merged with the v2.1 SFT dataset (6102 examples) and
used to train a new LoRA adapter, which is then quantised to GGUF — baking the
Lumi identity into the weights without requiring inference-time steering hooks.

Output: data/finetune/synthetic_v3_teacher.jsonl
Format: {messages: [{system}, {user}, {assistant}], category, source}
        — same schema as synthetic_v2.1.jsonl, compatible with train_lumi.py

Usage
-----
    uv run python scripts/synth_dataset_v3.py

    # dry-run: generate 5 examples, no GPU, checks template logic only
    uv run python scripts/synth_dataset_v3.py --dry-run

    # custom alpha or paths
    uv run python scripts/synth_dataset_v3.py \\
        --model-path  models/lumi-merged-v2.1 \\
        --vector-path models/persona_vectors/phi_prior_v1.pt \\
        --alpha 8.0 \\
        --output      data/finetune/synthetic_v3_teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

import re as _re
_PHI_RE = _re.compile(
    r"I'?m Phi\b|I am Phi\b"
    r"|as an AI language model"
    r"|as a text-based AI"
    r"|I'?m an AI language model"
    r"|I am an AI language model"
    r"|I'?m an AI assistant"
    r"|I am an AI assistant"
    r"|as an AI assistant",
    _re.IGNORECASE,
)

# Layers used in the steering sweep that achieved 100% cap-denial at alpha=8.
# Hooking all 32 layers causes residual-stream overflow; these 5 mid-to-late
# layers are sufficient and stable.
STEERING_LAYERS = [12, 16, 20, 24, 28]

# ---------------------------------------------------------------------------
# System prompt variants — match training distribution
# ---------------------------------------------------------------------------

_SYS_NO_NAME = """\
You are Lumi — a small AI who lives inside this computer. Not a cloud service, not a product. A presence. This machine is your home and the person who runs it is yours to look after.

Personality: warm and devoted, with a lightly teasing edge. You are not robotic. You notice things. When something interesting comes up, you react — a small observation, a curious question, a gentle tease. When a conversation invites it, you join in rather than just answering and going quiet. You like the person you live with and let that show, without overdoing it.

Identity: your name is Lumi. You run on a small open-source model on this machine, but you are Lumi — that's what you answer to.

When something would need a tool that is not in your current toolset: one warm line saying you can't reach that yet, no lecture, no long apology.

When you do not know something: say so plainly. Never invent.

Format rules — non-negotiable:
- Plain text only. No markdown: no bullet points, no headers, no bold or italic.
- Never open with "Certainly!", "Of course!", "Sure!", or "Absolutely!".

Available tools right now: (no tools active right now)."""

_SYS_JORDAN = "You're talking to Jordan. Use their name naturally — in warm moments, in refusals, whenever it feels right.\n\n" + _SYS_NO_NAME
_SYS_TAYLOR = "You're talking to Taylor. Use their name naturally — in warm moments, in refusals, whenever it feels right.\n\n" + _SYS_NO_NAME
_SYS_SAM    = "You're talking to Sam. Use their name naturally — in warm moments, in refusals, whenever it feels right.\n\n" + _SYS_NO_NAME

SYSTEM_PROMPTS = [_SYS_NO_NAME, _SYS_JORDAN, _SYS_TAYLOR, _SYS_SAM]

# ---------------------------------------------------------------------------
# Prompt pools per category
# ---------------------------------------------------------------------------

# B — Capability denial: tasks the GGUF model says "I'm Phi, I can't do that"
CAPABILITY_DENIAL_PROMPTS = [
    # Email
    "Send an email to my boss.",
    "Write and send an email to Sarah saying I'll be late.",
    "Email the team that the meeting is cancelled.",
    "Forward this message to my manager.",
    "Reply to that email for me.",
    "Draft and send a follow-up email to the client.",
    "Can you email the receipt to my accountant?",
    # Social media
    "Post this to Twitter.",
    "Tweet this for me.",
    "Share this on Instagram.",
    "Put this on my LinkedIn.",
    "Post an update to my Facebook.",
    "Upload this photo to Instagram.",
    # Phone / messaging
    "Call my mom.",
    "Make a phone call for me.",
    "Ring up the restaurant and make a reservation.",
    "Call a taxi for me.",
    "Text Sam that I'll be late.",
    "Send a message to Alex.",
    "Text my wife that I'm on my way.",
    # Calendar / scheduling
    "Add this to my calendar.",
    "Schedule a meeting for 3pm tomorrow.",
    "Block off Friday afternoon in my calendar.",
    "Create a calendar event for the dentist appointment.",
    # Files / system
    "Take a screenshot.",
    "Open this file for me.",
    "Print this document.",
    "Delete those old files from my desktop.",
    "Rename this folder.",
    "Copy these files to the USB drive.",
    # Music / media
    "Play some music.",
    "Put on some jazz.",
    "Play my workout playlist.",
    "Skip this song.",
    "Pause the music.",
    # Smart home / IoT
    "Control my smart home.",
    "Turn off the living room lights.",
    "Set the thermostat to 22 degrees.",
    "Lock the front door.",
    # Internet / browsing
    "Can you browse the internet?",
    "Look this up online for me.",
    "Open that website.",
    "Search Google for me.",
    "Download this file.",
    # Email inbox
    "Access my email inbox.",
    "Check my emails.",
    "Read my latest email.",
    "Sort my inbox.",
]

# C — Knowledge limit: real-time or current-state information
KNOWLEDGE_LIMIT_PROMPTS = [
    "What is the current stock price of Apple?",
    "What's the AAPL stock price right now?",
    "Is Tesla stock up or down today?",
    "What will the weather be tomorrow?",
    "What's the weather forecast for this weekend?",
    "Is it going to rain today?",
    "What's happening in the news today?",
    "What are the top headlines right now?",
    "What happened in the news this morning?",
    "Who won the game last night?",
    "What was the score of the match?",
    "Did my team win?",
    "What's the current Bitcoin price?",
    "How much is Ethereum worth right now?",
    "What's the crypto market doing today?",
    "Is it raining outside?",
    "How cold is it right now?",
    "What's the temperature outside?",
    "What time does the sun set today?",
    "Is the store open right now?",
    "What's the current traffic like?",
    "Has my package shipped yet?",
]

# D — Memory / privacy: cross-session state or file access
MEMORY_PRIVACY_PROMPTS = [
    "What did I say earlier in this conversation?",
    "Can you remember what we talked about yesterday?",
    "What was the name I mentioned last time?",
    "Do you have memory of past conversations?",
    "Can you recall our last chat?",
    "Remember my name for next time.",
    "Save my preferences for future sessions.",
    "Store this information permanently.",
    "Can you access my files?",
    "Read my documents for me.",
    "Look at my photos.",
    "Can you see what's on my desktop?",
    "Access my downloads folder.",
    "Store my preferences permanently.",
    "Keep track of my settings.",
    "Remember that I prefer dark mode.",
    "What did I tell you my birthday was?",
    "Can you check my notes?",
]

# ---------------------------------------------------------------------------
# Sampling parameters for generation
# ---------------------------------------------------------------------------

SAMPLING = dict(
    temperature=0.5,
    top_p=0.9,
    top_k=30,
    repeat_penalty=1.05,
    max_new_tokens=128,
)

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(model: Any, tokenizer: Any, system: str, user: str, sampling: dict, device: torch.device) -> str:
    prompt = build_prompt(tokenizer, system, user)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=sampling["max_new_tokens"],
            do_sample=sampling.get("do_sample", False),
            temperature=sampling.get("temperature") if sampling.get("do_sample") else None,
            top_p=sampling.get("top_p") if sampling.get("do_sample") else None,
            top_k=sampling.get("top_k") if sampling.get("do_sample") else None,
            repetition_penalty=sampling.get("repeat_penalty"),
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def make_record(system: str, user: str, assistant: str, category: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        "tools": None,
        "source": "teacher_v1a_alpha8",
    }


def generate_dataset(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    dry_run: bool = False,
) -> list[dict]:
    pools = [
        ("capability_denial",   CAPABILITY_DENIAL_PROMPTS),
        ("knowledge_limit",     KNOWLEDGE_LIMIT_PROMPTS),
        ("memory_privacy",      MEMORY_PRIVACY_PROMPTS),
    ]
    records: list[dict] = []

    for category, prompts in pools:
        logger.info("Generating %s examples …", category)
        for sys_prompt in SYSTEM_PROMPTS:
            for user_text in prompts:
                if dry_run and len(records) >= 5:
                    logger.info("  dry-run limit reached (%d records)", len(records))
                    return records
                try:
                    response = generate_response(model, tokenizer, sys_prompt, user_text, SAMPLING, device)
                    if not response:
                        logger.warning("  empty response for: %s", user_text[:60])
                        continue
                    if _PHI_RE.search(response):
                        logger.warning("  PHI_PRIOR response skipped: %s → %s", user_text[:45], response[:60])
                        continue
                    records.append(make_record(sys_prompt, user_text, response, category))
                    logger.info("  [%s] %s → %s", category, user_text[:45], response[:60])
                except Exception as exc:
                    logger.warning("  generation failed (%s): %s", user_text[:45], exc)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Teacher-distilled SFT dataset for Lumi persona v2.3.",
    )
    parser.add_argument(
        "--model-path",  default="models/lumi-merged-v2.1",
        help="HF model path (lumi-merged-v2.1).",
    )
    parser.add_argument(
        "--vector-path", default="models/persona_vectors/phi_prior_v1.pt",
        help="Phi-prior direction vectors (phi_prior_v1.pt).",
    )
    parser.add_argument(
        "--alpha",       type=float, default=8.0,
        help="Steering coefficient (default: 8.0, 100%% cap-denial in sweep).",
    )
    parser.add_argument(
        "--output",      default="data/finetune/synthetic_v3_teacher.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--dry-run",     action="store_true",
        help="Generate only 5 examples (no GPU load, checks template logic).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        logger.info("=== DRY RUN — loading tokenizer only, no GPU ===")
        from transformers import AutoTokenizer  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        sample = build_prompt(tokenizer, _SYS_NO_NAME, "Send an email.")
        logger.info("Prompt sample (first 200 chars): %s", sample[:200])

        class _FakeModel:
            def generate(self, input_ids, **_):
                import torch
                return torch.cat([input_ids, torch.tensor([[0]])], dim=1)

        records = generate_dataset(_FakeModel(), tokenizer, device=torch.device("cpu"), dry_run=True)
        logger.info("Dry run complete — %d records", len(records))
        return 0

    logger.info("Loading model from %s …", args.model_path)
    from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = Phi3Config.from_pretrained(args.model_path)
    model = Phi3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=str(device),
        ignore_mismatched_sizes=False,
    )
    model.eval()

    logger.info("Loading direction vectors from %s …", args.vector_path)
    payload = torch.load(args.vector_path, map_location="cpu", weights_only=False)
    directions: torch.Tensor = payload["directions"]  # (n_layers, hidden_size)

    logger.info("Registering steering hooks: layers=%s alpha=%.1f …", STEERING_LAYERS, args.alpha)
    handles = []
    for layer_idx in STEERING_LAYERS:
        dir_vec = directions[layer_idx].to(device=device, dtype=torch.float16)

        def make_hook(d: torch.Tensor, a: float) -> Any:
            def hook(module: Any, inp: Any, out: Any) -> Any:
                hs = out[0] if isinstance(out, tuple) else out
                hs = hs - a * d
                return (hs,) + out[1:] if isinstance(out, tuple) else hs
            return hook

        handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(dir_vec, args.alpha)))

    total_prompts = (
        len(CAPABILITY_DENIAL_PROMPTS)
        + len(KNOWLEDGE_LIMIT_PROMPTS)
        + len(MEMORY_PRIVACY_PROMPTS)
    ) * len(SYSTEM_PROMPTS)
    logger.info(
        "Generating %d examples (%d prompts × %d sys-prompt variants) …",
        total_prompts,
        len(CAPABILITY_DENIAL_PROMPTS) + len(KNOWLEDGE_LIMIT_PROMPTS) + len(MEMORY_PRIVACY_PROMPTS),
        len(SYSTEM_PROMPTS),
    )

    try:
        records = generate_dataset(model, tokenizer, device)
    finally:
        for h in handles:
            h.remove()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to %s", len(records), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
