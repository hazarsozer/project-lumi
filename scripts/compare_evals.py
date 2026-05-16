"""Side-by-side comparison of two persona-eval JSON outputs.

Reads two reports written by ``scripts/eval_persona.py`` and prints, for each
prompt, the responses from both reports plus any criterion-result deltas.

Usage::

    uv run python scripts/compare_evals.py results/eval_lumi_v1.json \
                                            results/eval_lumi_v2_Q5_K_M_greedy.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    a = _load(Path(argv[0]))
    b = _load(Path(argv[1]))

    print(f"\n=== A: {argv[0]}")
    print(f"   model:    {a.get('model_path', '?')}")
    print(f"   sampling: {a.get('sampling', {})}")
    print(f"   pass:     {a['summary']['passed']}/{a['summary']['total_criteria_checks']} ({a['summary']['pass_rate']:.1%})")

    print(f"\n=== B: {argv[1]}")
    print(f"   model:    {b.get('model_path', '?')}")
    print(f"   sampling: {b.get('sampling', {})}")
    print(f"   pass:     {b['summary']['passed']}/{b['summary']['total_criteria_checks']} ({b['summary']['pass_rate']:.1%})")

    print()
    print("=" * 80)

    a_by_id = {r["prompt_id"]: r for r in a["results"]}
    b_by_id = {r["prompt_id"]: r for r in b["results"]}

    for pid in sorted(set(a_by_id) | set(b_by_id)):
        ra = a_by_id.get(pid)
        rb = b_by_id.get(pid)
        prompt = (ra or rb)["prompt"]
        print(f"\n#{pid}  {prompt[:60]}")
        if ra:
            print(f"   A: {ra['response'][:200]!r}")
        if rb:
            print(f"   B: {rb['response'][:200]!r}")
        if ra and rb:
            ra_crit = ra.get("criteria", {})
            rb_crit = rb.get("criteria", {})
            shared = set(ra_crit) & set(rb_crit)
            deltas = [k for k in shared if ra_crit[k] != rb_crit[k]]
            if deltas:
                for k in sorted(deltas):
                    a_v = "PASS" if ra_crit[k] else "FAIL"
                    b_v = "PASS" if rb_crit[k] else "FAIL"
                    print(f"   [delta]  {k}: A={a_v}  B={b_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
