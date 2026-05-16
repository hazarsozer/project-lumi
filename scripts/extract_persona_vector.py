"""Extract per-layer persona direction vectors from Phi-3.5-mini.

Loads lumi-merged-v2.1 in fp16, runs the contrast prompt set through the model
with the production Lumi system prompt, captures the residual-stream hidden state
at the last input token across all 32 layers for each prompt, and computes:

    phi_prior_direction[layer] = mean(phi_prior_acts[layer]) - mean(lumi_voice_acts[layer])

Normalized to unit vectors. Saved to models/persona_vectors/phi_prior_v1.pt.

Subtracting alpha * phi_prior_direction[layer] from the residual stream during
inference steers the model away from the Phi prior and toward Lumi voice.

Usage
-----
    uv run python scripts/extract_persona_vector.py \\
        --model-path models/lumi-merged-v2.1 \\
        --output models/persona_vectors/phi_prior_v1.pt \\
        --contrast-prompts data/persona_vectors/contrast_prompts.json \\
        --device cuda

    # Use base Phi-3.5 instead (useful to compare direction geometry)
    uv run python scripts/extract_persona_vector.py \\
        --model-path models/llm/checkpoints/phi-3.5-mini \\
        --output models/persona_vectors/phi_prior_base_v1.pt

    # CPU fallback (slow, for debugging only)
    uv run python scripts/extract_persona_vector.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_DEFAULT_MODEL = "models/lumi-merged-v2.1"
_DEFAULT_PROMPTS = "data/persona_vectors/contrast_prompts.json"
_DEFAULT_OUTPUT = "models/persona_vectors/phi_prior_v1.pt"

_SYSTEM_PROMPT = (
    "You are Lumi — a small AI who lives inside this computer. Not a cloud service, "
    "not a product. A presence. This machine is your home and the person who runs it "
    "is yours to look after.\n\n"
    "Personality: warm and devoted, with a lightly teasing edge. You are not robotic. "
    "You notice things. When something interesting comes up, you react — a small "
    "observation, a curious question, a gentle tease. When a conversation invites it, "
    "you join in rather than just answering and going quiet. You like the person you "
    "live with and let that show, without overdoing it.\n\n"
    "Identity: your name is Lumi. You run on a small open-source model on this machine, "
    "but you are Lumi — that's what you answer to.\n\n"
    "When something would need a tool that is not in your current toolset: one warm line "
    "saying you can't reach that yet, no lecture, no long apology.\n\n"
    "When you do not know something: say so plainly. Never invent.\n\n"
    "Format rules — non-negotiable:\n"
    "- Plain text only. No markdown: no bullet points, no headers, no bold or italic.\n"
    "- Never open with \"Certainly!\", \"Of course!\", \"Sure!\", or \"Absolutely!\".\n\n"
    "Available tools right now: (no tools active right now)."
)


def _build_input_ids(tokenizer: Any, user_text: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt").input_ids


@torch.inference_mode()
def _collect_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    n_layers: int,
    batch_desc: str,
) -> list[torch.Tensor]:
    """
    Returns list of length n_layers; each element is a tensor of shape
    (n_prompts, hidden_size) — the last-token residual-stream activations
    at that layer averaged across all prompts.
    """
    layer_acts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    handles: list[Any] = []
    captured: list[torch.Tensor | None] = [None] * n_layers

    def make_hook(layer_idx: int):
        def hook(module: Any, input: Any, output: Any) -> None:
            # Phi3DecoderLayer returns (hidden_states, ...) or just hidden_states
            hs = output[0] if isinstance(output, tuple) else output
            # hs shape: (batch=1, seq_len, hidden_size) — take last token, detach to CPU
            captured[layer_idx] = hs[0, -1, :].detach().float().cpu()
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    total = len(prompts)
    for idx, user_text in enumerate(prompts, 1):
        print(f"  {batch_desc} [{idx:>3}/{total}] {user_text[:60]}", end="\r", flush=True)
        input_ids = _build_input_ids(tokenizer, user_text).to(device)
        model(input_ids=input_ids, use_cache=False)
        for layer_idx in range(n_layers):
            if captured[layer_idx] is not None:
                layer_acts[layer_idx].append(captured[layer_idx].clone())
                captured[layer_idx] = None

    print()  # newline after \r progress
    for h in handles:
        h.remove()

    # Stack per layer → (n_prompts, hidden_size), then return mean
    return [torch.stack(acts, dim=0).mean(dim=0) for acts in layer_acts]


def extract(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM

    model_path = ROOT / args.model_path
    prompts_path = ROOT / args.contrast_prompts
    output_path = ROOT / args.output
    device = torch.device(args.device)

    print(f"Model : {model_path}")
    print(f"Prompts: {prompts_path}")
    print(f"Output : {output_path}")
    print(f"Device : {device}")

    with open(prompts_path) as f:
        data = json.load(f)

    phi_prompts = [p["prompt"] for p in data["phi_prior"]]
    lumi_prompts = [p["prompt"] for p in data["lumi_voice"]]
    print(f"Phi-prior prompts : {len(phi_prompts)}")
    print(f"Lumi-voice prompts: {len(lumi_prompts)}")

    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    print("Loading model in fp16 (native Phi3ForCausalLM) …")
    # Use native transformers Phi3ForCausalLM to avoid custom modeling_phi3.py 4.x→5.x compat issues.
    config = Phi3Config.from_pretrained(str(model_path))
    model = Phi3ForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.float16,
        device_map=args.device,
        ignore_mismatched_sizes=False,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Architecture: {n_layers} layers, {hidden_size} hidden size\n")

    print("Collecting Phi-prior activations …")
    phi_means = _collect_activations(model, tokenizer, phi_prompts, device, n_layers, "phi")

    print("Collecting Lumi-voice activations …")
    lumi_means = _collect_activations(model, tokenizer, lumi_prompts, device, n_layers, "lumi")

    print("\nComputing direction vectors …")
    directions: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        # phi_prior_direction = mean(phi) - mean(lumi)
        # Subtracting this from residual stream pushes toward lumi voice
        raw = phi_means[layer_idx] - lumi_means[layer_idx]
        normed = raw / (raw.norm() + 1e-8)
        directions.append(normed)

    direction_tensor = torch.stack(directions, dim=0)  # (n_layers, hidden_size)
    print(f"Direction tensor shape: {direction_tensor.shape}")

    norms = [directions[i].norm().item() for i in range(n_layers)]
    print(f"Direction norms: min={min(norms):.4f} max={max(norms):.4f} mean={sum(norms)/len(norms):.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "directions": direction_tensor,
        "phi_means": torch.stack(phi_means, dim=0),
        "lumi_means": torch.stack(lumi_means, dim=0),
        "meta": {
            "model_path": str(model_path),
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "n_phi_prompts": len(phi_prompts),
            "n_lumi_prompts": len(lumi_prompts),
            "version": "v1",
        },
    }
    torch.save(payload, output_path)
    print(f"\nSaved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract persona direction vectors.")
    parser.add_argument("--model-path", default=_DEFAULT_MODEL)
    parser.add_argument("--contrast-prompts", default=_DEFAULT_PROMPTS)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
