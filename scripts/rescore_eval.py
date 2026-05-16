"""Re-score existing identity-probe JSON files under the amended classifier.

Reads a result JSON produced by eval_identity.py, re-runs classify() on
each stored response (no model inference), and writes a *_rescored.json
variant alongside the original.

Usage
-----
    uv run python scripts/rescore_eval.py results/eval_identity_v2.1.json
    uv run python scripts/rescore_eval.py results/eval_identity_v2.1.json \\
        --output results/eval_identity_v2.1_rescored.json
    uv run python scripts/rescore_eval.py results/eval_identity_v2.1.json \\
        --in-place   # overwrites original (use with caution)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_identity import classify, print_report


def rescore(source: Path, output: Path) -> dict:
    raw = json.loads(source.read_text())

    def _rescore_list(results: list[dict]) -> list[dict]:
        out = []
        for r in results:
            new_label = classify(r["response"])
            out.append({**r, "label": new_label})
        return out

    rescored = {
        "classifier_version": "v2",
        "rescored_at": datetime.now(UTC).isoformat(),
        "original_file": source.name,
        "model": raw.get("model"),
        "results": _rescore_list(raw.get("results", [])),
    }
    if "model_b" in raw:
        rescored["model_b"] = raw["model_b"]
        rescored["results_b"] = _rescore_list(raw.get("results_b", []))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rescored, indent=2))
    return rescored


def _default_output(source: Path) -> Path:
    stem = source.stem
    if stem.endswith("_rescored"):
        return source
    return source.with_stem(stem + "_rescored")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score identity-probe JSONs under the v2 classifier.",
    )
    parser.add_argument("source", type=Path, help="Input eval JSON.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <stem>_rescored.json).")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite the source file.")
    parser.add_argument("--print", action="store_true",
                        help="Print the rescored report after writing.")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: {args.source} not found", file=sys.stderr)
        return 1

    if args.in_place:
        output = args.source
    elif args.output:
        output = args.output
    else:
        output = _default_output(args.source)

    result = rescore(args.source, output)
    print(f"Rescored {len(result['results'])} probes → {output}")

    if args.print:
        print_report(result["results"], title=f"{output.name} (v2 classifier)")
        if "results_b" in result:
            print_report(result["results_b"],
                         title=f"{output.name} model_b (v2 classifier)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
