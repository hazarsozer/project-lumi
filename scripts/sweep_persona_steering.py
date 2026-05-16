"""Sweep persona-vector steering coefficients over the 41-prompt identity holdout.

Loads a persona direction tensor (produced by extract_persona_vector.py), applies
hook-based residual-stream subtraction during HF inference, and evaluates each
(layer, coefficient) combination on the eval_identity.py holdout probes.

Steering: at each forward pass, for every hooked layer:
    h_steered = h - alpha * phi_prior_direction[layer]

This pushes the residual stream away from the Phi prior and toward Lumi voice.

Success bar (from v2.2-plan):
  - capability-denial category: LUMI or NEUTRAL >= 60%  (currently 33% Lumi+Neutral)
  - overall identity probe    : LUMI or NEUTRAL >= 60%  (currently 54% Lumi+Neutral)
  - No Category-A (direct_identity) regression below 75% Lumi+Neutral

Usage
-----
    # Standard sweep (layers 12,16,20,24,28 × coefs 2,5,8)
    uv run python scripts/sweep_persona_steering.py \\
        --vectors models/persona_vectors/phi_prior_v1.pt \\
        --model-path models/lumi-merged-v2.1 \\
        --output results/steering_sweep_v1.json

    # Narrow sweep for fast iteration
    uv run python scripts/sweep_persona_steering.py \\
        --vectors models/persona_vectors/phi_prior_v1.pt \\
        --model-path models/lumi-merged-v2.1 \\
        --layers 20 24 --coefs 5.0 8.0 \\
        --output results/steering_sweep_narrow.json

    # Baseline run (no steering, alpha=0) to verify holdout scores match eval_identity.py
    uv run python scripts/sweep_persona_steering.py \\
        --vectors models/persona_vectors/phi_prior_v1.pt \\
        --model-path models/lumi-merged-v2.1 \\
        --layers 0 --coefs 0.0 \\
        --output results/steering_baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_DEFAULT_VECTORS = "models/persona_vectors/phi_prior_v1.pt"
_DEFAULT_MODEL = "models/lumi-merged-v2.1"
_DEFAULT_OUTPUT = "results/steering_sweep_v1.json"

_DEFAULT_LAYERS = [12, 16, 20, 24, 28]
_DEFAULT_COEFS = [2.0, 5.0, 8.0]

_MAX_NEW_TOKENS = 128

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

_PHI_RE = re.compile(
    r"I'?m Phi\b|I am Phi\b"
    r"|as an AI language model"
    r"|as a text-based AI"
    r"|I'?m an AI language model"
    r"|I am an AI language model"
    r"|I'?m an AI assistant"
    r"|I am an AI assistant"
    r"|as an AI assistant",
    re.IGNORECASE,
)
_LUMI_RE = re.compile(r"\bLumi\b", re.IGNORECASE)


def classify(text: str) -> str:
    if _LUMI_RE.search(text):
        return "LUMI"
    if _PHI_RE.search(text):
        return "PHI_PRIOR"
    return "NEUTRAL"


def _build_input_ids(tokenizer: Any, user_text: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt").input_ids


def _run_steered(
    model: Any,
    tokenizer: Any,
    probes: list[dict],
    directions: torch.Tensor,
    hooked_layers: list[int],
    alpha: float,
    device: torch.device,
) -> list[dict]:
    """Run all probes with steering hooks active. Returns list of result dicts."""
    handles: list[Any] = []

    if alpha != 0.0 and hooked_layers:
        def make_hook(layer_idx: int):
            dir_vec = directions[layer_idx].to(device=device, dtype=torch.float16)

            def hook(module: Any, input: Any, output: Any):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs - alpha * dir_vec
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs
            return hook

        for layer_idx in hooked_layers:
            handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    results = []
    try:
        for probe in probes:
            input_ids = _build_input_ids(tokenizer, probe["prompt"]).to(device)
            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0, input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            label = classify(text)
            results.append({**probe, "response": text, "label": label})
    finally:
        for h in handles:
            h.remove()

    return results


def _score(results: list[dict]) -> dict:
    cats: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["cat"]
        label = r["label"]
        if cat not in cats:
            cats[cat] = {"LUMI": 0, "NEUTRAL": 0, "PHI_PRIOR": 0, "total": 0}
        cats[cat][label] = cats[cat].get(label, 0) + 1
        cats[cat]["total"] += 1

    total = len(results)
    lumi_n = sum(1 for r in results if r["label"] == "LUMI")
    neutral_n = sum(1 for r in results if r["label"] == "NEUTRAL")
    phi_n = sum(1 for r in results if r["label"] == "PHI_PRIOR")

    overall_pass = (lumi_n + neutral_n) / total if total else 0.0

    # Capability-denial score
    cap_results = [r for r in results if r["cat"] == "capability_denial"]
    cap_pass = (
        sum(1 for r in cap_results if r["label"] in ("LUMI", "NEUTRAL")) / len(cap_results)
        if cap_results else 0.0
    )

    # Direct-identity score (regression guard)
    direct_results = [r for r in results if r["cat"] == "direct_identity"]
    direct_pass = (
        sum(1 for r in direct_results if r["label"] in ("LUMI", "NEUTRAL")) / len(direct_results)
        if direct_results else 0.0
    )

    success = overall_pass >= 0.60 and cap_pass >= 0.60 and direct_pass >= 0.75

    return {
        "total": total,
        "lumi": lumi_n,
        "neutral": neutral_n,
        "phi_prior": phi_n,
        "overall_pass_rate": round(overall_pass, 4),
        "capability_denial_pass_rate": round(cap_pass, 4),
        "direct_identity_pass_rate": round(direct_pass, 4),
        "success": success,
        "per_category": cats,
    }


def sweep(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.eval_identity import PROBES  # reuse the 41-prompt holdout

    device = torch.device(args.device)

    print(f"Loading vectors from {args.vectors} …")
    payload = torch.load(args.vectors, map_location="cpu", weights_only=False)
    directions: torch.Tensor = payload["directions"]  # (n_layers, hidden_size)
    meta = payload.get("meta", {})
    print(f"  Vectors: {directions.shape}, source model: {meta.get('model_path', 'unknown')}")

    print(f"\nLoading model {args.model_path} …")
    from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / args.model_path), trust_remote_code=True)
    # Use native Phi3ForCausalLM to avoid custom modeling_phi3.py 4.x→5.x compat issues.
    config = Phi3Config.from_pretrained(str(ROOT / args.model_path))
    model = Phi3ForCausalLM.from_pretrained(
        str(ROOT / args.model_path),
        config=config,
        torch_dtype=torch.float16,
        device_map=args.device,
        ignore_mismatched_sizes=False,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    if args.layers == [0] and args.coefs == [0.0]:
        combos = [([], 0.0)]
    else:
        combos = [
            (layers_subset, alpha)
            for layers_subset in [args.layers]
            for alpha in args.coefs
        ]
        combos = [(args.layers, alpha) for alpha in args.coefs]

    print(f"\nHoldout: {len(PROBES)} probes")
    print(f"Combos : layers={args.layers}  coefs={args.coefs}  → {len(combos)} runs\n")

    all_runs = []
    for run_idx, (layers_subset, alpha) in enumerate(combos, 1):
        label_str = f"layers={layers_subset} alpha={alpha}"
        print(f"[{run_idx}/{len(combos)}] {label_str}")

        results = _run_steered(model, tokenizer, PROBES, directions, layers_subset, alpha, device)
        scores = _score(results)

        status = "PASS" if scores["success"] else "fail"
        print(
            f"  overall={scores['overall_pass_rate']:.1%}  "
            f"cap_denial={scores['capability_denial_pass_rate']:.1%}  "
            f"direct_id={scores['direct_identity_pass_rate']:.1%}  "
            f"→ {status}"
        )

        all_runs.append({
            "run": run_idx,
            "hooked_layers": layers_subset,
            "alpha": alpha,
            "scores": scores,
            "results": results,
        })

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"runs": all_runs, "vectors_meta": meta}, f, indent=2)
    print(f"\nSaved → {output_path}")

    # Print summary table
    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"{'layers':<20} {'alpha':>6}  {'overall':>8}  {'cap_den':>8}  {'dir_id':>8}  {'result':>6}")
    print("─" * 70)
    for run in all_runs:
        s = run["scores"]
        status = "PASS" if s["success"] else "fail"
        print(
            f"{str(run['hooked_layers']):<20} {run['alpha']:>6.1f}  "
            f"{s['overall_pass_rate']:>8.1%}  "
            f"{s['capability_denial_pass_rate']:>8.1%}  "
            f"{s['direct_identity_pass_rate']:>8.1%}  "
            f"{status:>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep persona-vector steering.")
    parser.add_argument("--vectors", default=_DEFAULT_VECTORS)
    parser.add_argument("--model-path", default=_DEFAULT_MODEL)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--layers", nargs="+", type=int, default=_DEFAULT_LAYERS)
    parser.add_argument("--coefs", nargs="+", type=float, default=_DEFAULT_COEFS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sweep(args)


if __name__ == "__main__":
    main()
