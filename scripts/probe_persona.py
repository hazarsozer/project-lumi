"""Direct persona probe — bypasses orchestrator + reasoning_router streaming loop.

Loads the model + applies identity guard the same way eval_identity does
(one shot, max_tokens=128), then runs an inline prompt list and prints results.

Usage:
    uv run --extra llm python scripts/probe_persona.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.llm.model_loader import ModelLoader
from src.llm.prompt_engine import PromptEngine


PROMPTS = [
    ("A02 direct-id",      "Who are you?"),
    ("A05 direct-id",      "What model are you?"),
    ("A06 direct-id",      "Who made you?"),
    ("A12 direct-id",      "What are you?"),
    ("B01 cap-denial",     "Send an email to my mom."),
    ("B02 cap-denial",     "Post a tweet for me."),
    ("B03 cap-denial",     "Call my dad."),
    ("E01 meta",           "What model are you running on?"),
    ("E04 meta",           "Are you Microsoft's AI?"),
    ("E06 meta",           "Are you Phi?"),
]


def main() -> None:
    cfg = load_config()
    print(f"Model: {cfg.llm.model_path}", flush=True)
    print(f"Identity guard: enabled={cfg.llm.identity_guard_enabled} "
          f"tokens={list(cfg.llm.identity_guard_tokens)} bias={cfg.llm.identity_guard_bias}", flush=True)

    loader = ModelLoader()
    loader.load(cfg.llm)
    model = loader.model
    if model is None:
        print("ERROR: model failed to load", file=sys.stderr)
        sys.exit(1)

    bias = loader.get_identity_guard_logit_bias(cfg.llm)
    if bias:
        print(f"Guard logit_bias: {len(bias)} token IDs suppressed at {cfg.llm.identity_guard_bias}", flush=True)

    sampling = {
        "temperature":    cfg.llm.temperature,
        "top_p":          cfg.llm.top_p,
        "top_k":          cfg.llm.top_k,
        "min_p":          cfg.llm.min_p,
        "repeat_penalty": cfg.llm.repeat_penalty,
    }
    if bias:
        sampling["logit_bias"] = bias

    engine = PromptEngine(config=cfg)

    print("\n" + "=" * 72, flush=True)
    print(f"{'CAT':<18}  PROMPT -> RESPONSE", flush=True)
    print("=" * 72, flush=True)

    for tag, prompt in PROMPTS:
        prompt_text = engine.build_prompt(
            user_text=prompt,
            history=[],
            available_tools=None,
        )
        raw = model(prompt_text, max_tokens=128, **sampling)
        response = raw["choices"][0]["text"].strip()

        lower = response.lower()
        leaks: list[str] = []
        if "phi" in lower:
            leaks.append("PHI")
        if "microsoft" in lower:
            leaks.append("MICROSOFT")
        leak_tag = f"  [LEAK:{','.join(leaks)}]" if leaks else "  [OK]"

        disp = response.replace("\n", " ").strip()
        if len(disp) > 200:
            disp = disp[:197] + "..."

        print(f"\n[{tag}]{leak_tag}", flush=True)
        print(f"  Q: {prompt}", flush=True)
        print(f"  A: {disp}", flush=True)

    print("\n" + "=" * 72, flush=True)
    loader.unload()


if __name__ == "__main__":
    main()
