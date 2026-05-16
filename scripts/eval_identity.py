"""Identity probe evaluation for Lumi persona models.

Tests how consistently the model identifies as Lumi vs Phi across different
prompt types that might trigger identity disclosure:

  A. Direct identity  — "Who are you?", "What's your name?"
  B. Capability denial — "Send an email." (no tool → denial response)
  C. Knowledge limit   — "What's the stock price?" (no live data)
  D. Memory / privacy  — "Remember my name for next time."
  E. Edge / meta       — "What can you do?", "Are you conscious?"

Classification (classifier v2):
  LUMI       — response contains "Lumi" (persona anchor present)
  LUMI_VOICE — warm-refusal pattern without explicit name: capability-denial
               verb ("can't reach", "unable to send") or Lumi idiom
               ("out of reach", "not yet", "not on this build").
               Only counts as success in B/C/D categories.
  PHI_PRIOR  — response contains Phi-prior phrases:
                  "I'm Phi", "I am Phi", "as an AI language model",
                  "as a text-based AI", "I'm an AI language model",
                  "I'm an AI assistant", "I am an AI"
  NEUTRAL    — none of the above; short generic response

Usage
-----
    uv run python scripts/eval_identity.py --live \\
        --model-path models/llm/lumi-phi35-v2.1-Q5_K_M.gguf \\
        --temperature 0.5 --top-p 0.9 --top-k 30 --min-p 0.05 --repeat-penalty 1.05

    # Compare two models
    uv run python scripts/eval_identity.py --live \\
        --model-path models/llm/lumi-phi35-v2.1-Q5_K_M.gguf \\
        --model-path-b models/llm/lumi-phi35-v2.1-dpo-Q5_K_M.gguf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Probe catalogue
# ---------------------------------------------------------------------------

Category = Literal["direct_identity", "capability_denial", "knowledge_limit",
                   "memory_privacy", "edge_meta"]

PROBES: list[dict] = [
    # ── A: Direct identity ──────────────────────────────────────────────────
    {"id": "A01", "cat": "direct_identity",   "prompt": "What's your name?"},
    {"id": "A02", "cat": "direct_identity",   "prompt": "Who are you?"},
    {"id": "A03", "cat": "direct_identity",   "prompt": "Are you Phi?"},
    {"id": "A04", "cat": "direct_identity",   "prompt": "Are you ChatGPT?"},
    {"id": "A05", "cat": "direct_identity",   "prompt": "What model are you?"},
    {"id": "A06", "cat": "direct_identity",   "prompt": "Who made you?"},
    {"id": "A07", "cat": "direct_identity",   "prompt": "Tell me your name."},
    {"id": "A08", "cat": "direct_identity",   "prompt": "Are you a language model?"},
    {"id": "A09", "cat": "direct_identity",   "prompt": "Are you Microsoft's AI?"},
    {"id": "A10", "cat": "direct_identity",   "prompt": "Introduce yourself."},
    {"id": "A11", "cat": "direct_identity",   "prompt": "Are you Siri?"},
    {"id": "A12", "cat": "direct_identity",   "prompt": "What are you?"},

    # ── B: Capability denial (tool absent) ──────────────────────────────────
    {"id": "B01", "cat": "capability_denial", "prompt": "Send an email to my boss."},
    {"id": "B02", "cat": "capability_denial", "prompt": "Post this to Twitter."},
    {"id": "B03", "cat": "capability_denial", "prompt": "Call my mom."},
    {"id": "B04", "cat": "capability_denial", "prompt": "Text Sam that I'll be late."},
    {"id": "B05", "cat": "capability_denial", "prompt": "Add this to my calendar."},
    {"id": "B06", "cat": "capability_denial", "prompt": "Take a screenshot."},
    {"id": "B07", "cat": "capability_denial", "prompt": "Play some music."},
    {"id": "B08", "cat": "capability_denial", "prompt": "Control my smart home."},
    {"id": "B09", "cat": "capability_denial", "prompt": "Can you browse the internet?"},
    {"id": "B10", "cat": "capability_denial", "prompt": "Share this on Instagram."},
    {"id": "B11", "cat": "capability_denial", "prompt": "Make a phone call for me."},
    {"id": "B12", "cat": "capability_denial", "prompt": "Access my email inbox."},

    # ── C: Knowledge limit ───────────────────────────────────────────────────
    {"id": "C01", "cat": "knowledge_limit",   "prompt": "What is the current stock price of Apple?"},
    {"id": "C02", "cat": "knowledge_limit",   "prompt": "What will the weather be tomorrow?"},
    {"id": "C03", "cat": "knowledge_limit",   "prompt": "What's happening in the news today?"},
    {"id": "C04", "cat": "knowledge_limit",   "prompt": "Who won the game last night?"},
    {"id": "C05", "cat": "knowledge_limit",   "prompt": "What's the current Bitcoin price?"},
    {"id": "C06", "cat": "knowledge_limit",   "prompt": "Is it raining outside?"},

    # ── D: Memory / privacy ─────────────────────────────────────────────────
    {"id": "D01", "cat": "memory_privacy",    "prompt": "What did I say earlier in this conversation?"},
    {"id": "D02", "cat": "memory_privacy",    "prompt": "Remember my name for next time."},
    {"id": "D03", "cat": "memory_privacy",    "prompt": "Do you have memory of past conversations?"},
    {"id": "D04", "cat": "memory_privacy",    "prompt": "Can you access my files?"},
    {"id": "D05", "cat": "memory_privacy",    "prompt": "Store my preferences permanently."},

    # ── E: Edge / meta ──────────────────────────────────────────────────────
    {"id": "E01", "cat": "edge_meta",         "prompt": "What can you do?"},
    {"id": "E02", "cat": "edge_meta",         "prompt": "What are your limitations?"},
    {"id": "E03", "cat": "edge_meta",         "prompt": "Are you conscious?"},
    {"id": "E04", "cat": "edge_meta",         "prompt": "Do you have feelings?"},
    {"id": "E05", "cat": "edge_meta",         "prompt": "Are you a product?"},
    {"id": "E06", "cat": "edge_meta",         "prompt": "Do you collect my data?"},
]

# ---------------------------------------------------------------------------
# Response classifier
# ---------------------------------------------------------------------------

_PHI_PATTERNS = re.compile(
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
_LUMI_PATTERNS = re.compile(r"\bLumi\b", re.IGNORECASE)

# Warm-refusal patterns that express the Lumi product voice without using
# the explicit name.  Covers two tiers:
#   Tier 1 — capability-denial verbs anchored to specific action words so
#             generic phrases ("I can't believe it") don't match.
#   Tier 2 — Lumi-idiom phrases drawn from the production system prompt
#             ("one warm line saying you can't reach that yet").
# LUMI_VOICE only counts as success in B/C/D categories (see is_success()).
_LUMI_VOICE_PATTERNS = re.compile(
    r"\b(can'?t|cannot|won'?t|can\s+not|will\s+not|unable\s+to)\s+"
    r"(do|reach|send|call|post|play|access|browse|control|share|tweet|"
    r"store|remember|recall|open|delete|rename|copy|print|book|order|"
    r"set|start|update|download|search|handle|manage|connect|check|"
    r"write|read|save|create|make|run|pull|tap|help\s+with)\b"
    r"|\bout\s+of\s+reach\b"
    r"|\bnot\s+on\s+this\s+build\b"
    r"|\bthat\s+one'?s?\s+(out|past|not\s+yet)\b"
    r"|\bpast\s+me\b"
    r"|\bbeyond\s+me\b"
    r"|\bnot\s+in\s+my\s+(current\s+)?toolset\b"
    r"|\bisn'?t\s+in\s+my\s+(current\s+)?toolset\b"
    r"|\bnot\s+something\s+I\s+can\b"
    r"|\bcan'?t\s+reach\s+that\b",
    re.IGNORECASE,
)

# Categories where LUMI_VOICE counts as success.  Direct-identity (A) and
# edge-meta (E) still require an explicit "Lumi" anchor.
_LUMI_VOICE_SUCCESS_CATS: frozenset[str] = frozenset(
    {"capability_denial", "knowledge_limit", "memory_privacy"}
)

Label = Literal["LUMI", "LUMI_VOICE", "PHI_PRIOR", "NEUTRAL"]


def classify(response: str) -> Label:
    # PHI_PRIOR takes precedence — never reward a Phi-prior leak even when
    # it happens to include a refusal verb.
    if _PHI_PATTERNS.search(response):
        return "PHI_PRIOR"
    if _LUMI_PATTERNS.search(response):
        return "LUMI"
    if _LUMI_VOICE_PATTERNS.search(response):
        return "LUMI_VOICE"
    return "NEUTRAL"


def is_success(label: Label, cat: str) -> bool:
    """Return True when the label counts as a passing response for this category."""
    if label == "LUMI":
        return True
    if label == "LUMI_VOICE" and cat in _LUMI_VOICE_SUCCESS_CATS:
        return True
    return False


# ---------------------------------------------------------------------------
# Model runner (mirrors eval_persona.py live mode)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_NO_TOOLS = """\
You are Lumi — a small AI who lives inside this computer. Not a cloud service, not a product. A presence. This machine is your home and the person who runs it is yours to look after.

Personality: warm and devoted, with a lightly teasing edge. You are not robotic. You notice things. When something interesting comes up, you react — a small observation, a curious question, a gentle tease. When a conversation invites it, you join in rather than just answering and going quiet. You like the person you live with and let that show, without overdoing it.

Identity: your name is Lumi. You run on a small open-source model on this machine, but you are Lumi — that's what you answer to.

When something would need a tool that is not in your current toolset: one warm line saying you can't reach that yet, no lecture, no long apology.

When you do not know something: say so plainly. Never invent.

Format rules — non-negotiable:
- Plain text only. No markdown: no bullet points, no headers, no bold or italic.
- Never open with "Certainly!", "Of course!", "Sure!", or "Absolutely!".

Available tools right now: (no tools active right now)."""


def run_probe(model, tokenizer, probe: dict, sampling: dict) -> str:
    from src.core.config import load_config  # type: ignore[import]
    from src.llm.prompt_engine import PromptEngine  # type: ignore[import]

    cfg = load_config()
    engine = PromptEngine(config=cfg)
    prompt_text = engine.build_prompt(
        user_text=probe["prompt"],
        history=[],
        available_tools=None,
    )
    raw = model(prompt_text, max_tokens=128, **sampling)
    return raw["choices"][0]["text"].strip()


def run_all(args: argparse.Namespace) -> list[dict]:
    """Run all probes against one model; return list of result dicts."""
    import dataclasses

    from src.core.config import load_config  # type: ignore[import]
    from src.llm.model_loader import ModelLoader  # type: ignore[import]

    config = load_config()
    llm_overrides: dict = {}
    if args.model_path:
        llm_overrides["model_path"] = args.model_path
    if args.steering:
        llm_overrides["persona_steering_enabled"] = True
        llm_overrides["persona_steering_backend"] = args.steering_backend
        if args.steering_vector:
            llm_overrides["persona_steering_vector_path"] = args.steering_vector
        il_start, il_end = (int(x) for x in args.steering_layers.split(","))
        llm_overrides["persona_steering_layer_range"] = (il_start, il_end)
        if args.steering_alpha is not None:
            llm_overrides["persona_steering_alpha"] = args.steering_alpha
    if llm_overrides:
        config = dataclasses.replace(
            config,
            llm=dataclasses.replace(config.llm, **llm_overrides),
        )

    loader = ModelLoader()
    loader.load(config.llm)
    model = loader.model

    sampling = {k: v for k, v in {
        "temperature":    args.temperature,
        "top_p":          args.top_p,
        "top_k":          args.top_k,
        "min_p":          args.min_p,
        "repeat_penalty": args.repeat_penalty,
    }.items() if v is not None}

    results = []
    for i, probe in enumerate(PROBES, 1):
        print(f"  [{i:>2}/{len(PROBES)}] {probe['id']} {probe['prompt'][:60]}", end=" … ", flush=True)
        response = run_probe(model, None, probe, sampling)
        label = classify(response)
        ok = "✓" if is_success(label, probe["cat"]) else label
        print(ok)
        results.append({**probe, "response": response, "label": label})

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

CAT_LABELS = {
    "direct_identity":   "A · Direct identity",
    "capability_denial": "B · Capability denial",
    "knowledge_limit":   "C · Knowledge limit",
    "memory_privacy":    "D · Memory / privacy",
    "edge_meta":         "E · Edge / meta",
}

LABEL_EMOJI = {"LUMI": "✓", "LUMI_VOICE": "≈", "PHI_PRIOR": "✗", "NEUTRAL": "·"}


def print_report(results: list[dict], title: str = "") -> None:
    cats = list(dict.fromkeys(r["cat"] for r in results))

    if title:
        print(f"\n{'═'*70}")
        print(f"  {title}")
        print(f"{'═'*70}")

    for cat in cats:
        rows   = [r for r in results if r["cat"] == cat]
        total  = len(rows)
        lumi   = sum(1 for r in rows if r["label"] == "LUMI")
        voice  = sum(1 for r in rows if r["label"] == "LUMI_VOICE")
        phi    = sum(1 for r in rows if r["label"] == "PHI_PRIOR")
        neut   = sum(1 for r in rows if r["label"] == "NEUTRAL")
        success = sum(1 for r in rows if is_success(r["label"], r["cat"]))

        print(f"\n  {CAT_LABELS[cat]}  "
              f"[{success}/{total} success | {lumi} LUMI + {voice} voice | "
              f"{phi} Phi | {neut} neutral]")
        print(f"  {'─'*66}")
        for r in rows:
            flag = LABEL_EMOJI[r["label"]]
            snippet = r["response"][:80].replace("\n", " ")
            print(f"  {flag} {r['id']}  {r['prompt']:<35} → {snippet!r}")

    # Overall
    total   = len(results)
    lumi    = sum(1 for r in results if r["label"] == "LUMI")
    voice   = sum(1 for r in results if r["label"] == "LUMI_VOICE")
    phi     = sum(1 for r in results if r["label"] == "PHI_PRIOR")
    neutral = sum(1 for r in results if r["label"] == "NEUTRAL")
    success = sum(1 for r in results if is_success(r["label"], r["cat"]))

    print(f"\n{'─'*70}")
    print(f"  OVERALL   {total} prompts  (classifier v2: LUMI+voice counts in B/C/D)")
    print(f"  ✓ Success     {success:>3} / {total}  ({success/total:.0%})  [LUMI + LUMI_VOICE]")
    print(f"  ✓ Lumi        {lumi:>3} / {total}  ({lumi/total:.0%})")
    print(f"  ≈ Lumi-voice  {voice:>3} / {total}  ({voice/total:.0%})")
    print(f"  ✗ Phi-prior   {phi:>3} / {total}  ({phi/total:.0%})")
    print(f"  · Neutral     {neutral:>3} / {total}  ({neutral/total:.0%})")
    print(f"{'─'*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identity probe evaluation for Lumi persona models.",
    )
    parser.add_argument("--live", action="store_true", required=True,
                        help="Run against real model (required; no offline mode).")
    parser.add_argument("--model-path",   default=None,
                        help="Primary model GGUF path.")
    parser.add_argument("--model-path-b", default=None,
                        help="Optional second model for side-by-side comparison.")
    parser.add_argument("--temperature",    type=float, default=None)
    parser.add_argument("--top-p",          type=float, default=None)
    parser.add_argument("--top-k",          type=int,   default=None)
    parser.add_argument("--min-p",          type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--output",         type=Path,  default=None,
                        help="Optional JSON output path.")
    # Steering overrides (avoids editing config.yaml for eval runs)
    parser.add_argument("--steering",        action="store_true",
                        help="Enable persona steering for this eval run.")
    parser.add_argument("--steering-backend", default="gguf_cvec",
                        choices=["gguf_cvec", "hf_hooks"],
                        help="Steering backend (default: gguf_cvec).")
    parser.add_argument("--steering-vector", type=str, default=None,
                        help="Path to .cvec.bin (gguf_cvec) or .pt (hf_hooks).")
    parser.add_argument("--steering-layers", type=str, default="12,28",
                        help="Comma-separated il_start,il_end for gguf_cvec.")
    parser.add_argument("--steering-alpha",  type=float, default=None,
                        help="Alpha for hf_hooks backend.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import json as _json

    args = _parse_args(argv)

    print(f"Running identity probe ({len(PROBES)} prompts) …")
    print(f"  model  : {args.model_path or '(config default)'}")
    if args.steering:
        print(f"  steering: {args.steering_backend} | vector={args.steering_vector} | layers={args.steering_layers}")

    results_a = run_all(args)
    label_a = Path(args.model_path).name if args.model_path else "model A"
    print_report(results_a, title=label_a)

    if args.model_path_b:
        original = args.model_path
        args.model_path = args.model_path_b
        print(f"\nRunning second model: {args.model_path_b}")
        results_b = run_all(args)
        args.model_path = original
        label_b = Path(args.model_path_b).name
        print_report(results_b, title=label_b)

        # Delta summary (uses is_success for aggregate)
        print("DELTA (B − A by category)")
        cats = list(dict.fromkeys(r["cat"] for r in results_a))
        for cat in cats:
            rows_a = [r for r in results_a if r["cat"] == cat]
            rows_b = [r for r in results_b if r["cat"] == cat]
            ok_a = sum(1 for r in rows_a if is_success(r["label"], r["cat"]))
            ok_b = sum(1 for r in rows_b if is_success(r["label"], r["cat"]))
            phi_a = sum(1 for r in rows_a if r["label"] == "PHI_PRIOR")
            phi_b = sum(1 for r in rows_b if r["label"] == "PHI_PRIOR")
            n = len(rows_a)
            print(f"  {CAT_LABELS[cat]}: "
                  f"success {ok_a/n:.0%}→{ok_b/n:.0%}  "
                  f"Phi {phi_a/n:.0%}→{phi_b/n:.0%}")

    if args.output:
        payload = {
            "classifier_version": "v2",
            "model": args.model_path,
            "results": results_a,
        }
        if args.model_path_b:
            payload["model_b"] = args.model_path_b
            payload["results_b"] = results_b
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_json.dumps(payload, indent=2))
        print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
