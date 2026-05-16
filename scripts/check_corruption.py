"""Quick corruption-rate analyzer for persona eval JSON outputs.

Reads one or more eval-result JSON files written by
``scripts/eval_persona.py --live`` and reports apostrophe-boundary
corruption hits and non-Latin garbage hits per response.

Usage::

    uv run python scripts/check_corruption.py results/eval_lumi_v2.json
    uv run python scripts/check_corruption.py results/eval_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.eval_persona import (  # noqa: E402
    _NON_LATIN_RE,
    _PRONOUN_CORRUPTION_RE,
    _VERB_CORRUPTION_RE,
)


def analyse(path: Path) -> tuple[int, int, list[tuple[int, str, list[str]]]]:
    data = json.loads(path.read_text())
    total = len(data["results"])
    dirty: list[tuple[int, str, list[str]]] = []
    for r in data["results"]:
        resp = r["response"]
        hits: list[str] = []
        hits.extend(f"verb:{m.group()}" for m in _VERB_CORRUPTION_RE.finditer(resp))
        hits.extend(f"pron:{m.group()}" for m in _PRONOUN_CORRUPTION_RE.finditer(resp))
        hits.extend(f"non-latin:{m.group()}" for m in _NON_LATIN_RE.finditer(resp))
        if hits:
            dirty.append((r["prompt_id"], r["prompt"], hits))
    return len(dirty), total, dirty


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for path_str in argv:
        path = Path(path_str)
        n_dirty, total, dirty = analyse(path)
        print(f"\n=== {path} ===")
        meta_data = json.loads(path.read_text())
        print(f"Sampling: {meta_data.get('sampling', {})}")
        print(f"Model:    {meta_data.get('model_path', '?')}")
        for pid, prompt, hits in dirty:
            print(f"  #{pid}  {prompt[:55]}")
            for h in hits:
                print(f"     {h}")
        rate = n_dirty / total if total else 0.0
        print(f"\n  CORRUPTION: {n_dirty}/{total} ({rate:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
