"""Merge ``synthetic_v4.jsonl`` with preserved ``synthetic_v2.1.jsonl``
records into the final ``synthetic_v2.4.jsonl`` training corpus.

Logic (per v2.4-plan §"Records to PRESERVE / DROP")
---------------------------------------------------
1. Load v4 (Gemini-distilled warm-refusal records; legacy
   ``synthetic_v4_claude`` records are also accepted for back-compat) and v2.1
   (the existing broad SFT corpus).
2. Drop v2.1 records where ``category == "conditional_no_tool"`` — these are
   the target-blind cap-denial records replaced by v4. (The script-quota
   was 1500; the actual emitted count in ``synthetic_v2.1.jsonl`` is 1289 —
   see v2.4-plan §"Records to DROP from v2.1" for the reconciliation.)
3. Validate every record against the shared schema:
     - ``messages`` is a 3-tuple of system / user / assistant.
     - ``category`` is a string.
     - ``tools`` is ``None`` or a list of strings.
     - ``source`` is a string. v4 records must carry
       ``source ∈ {"synthetic_v4", "synthetic_v4_claude"}``; v2.1 records may
       carry any non-empty source string.
4. Shuffle (seeded — default seed 42 — for reproducibility).
5. Write merged JSONL to ``--output``.
6. Print summary stats (counts per category, totals, % cap-denial).

Usage
-----
::

    uv run python scripts/merge_v2.4_corpus.py
    uv run python scripts/merge_v2.4_corpus.py --dry-run
    uv run python scripts/merge_v2.4_corpus.py --v4-input data/finetune/synthetic_v4.jsonl \\
        --v2.1-input data/finetune/synthetic_v2.1.jsonl \\
        --output data/finetune/synthetic_v2.4.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DROP_CATEGORIES_FROM_V21: frozenset[str] = frozenset({"conditional_no_tool"})
REQUIRED_ROLES: tuple[str, ...] = ("system", "user", "assistant")
CAP_DENIAL_CATEGORIES: frozenset[str] = frozenset({"capability_denial"})

# v4 records may carry either the current Gemini source tag or the legacy
# Claude one (any fixtures still on disk pre-pivot). v2.1 records pass through
# the generic non-empty-string check without source-tag policing.
V4_ALLOWED_SOURCES: frozenset[str] = frozenset(
    {"synthetic_v4", "synthetic_v4_claude"}
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_record(
    record: Any,
    *,
    source_label: str,
    allowed_sources: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Return (is_valid, reason). Pure (does not mutate input).

    When ``allowed_sources`` is provided, the record's ``source`` field must
    match one of the entries (this is how v4 records are policed against the
    Gemini / legacy-Claude tag allowlist). When it is ``None``, ``source``
    only needs to be a non-empty string (v2.1 path).
    """
    if not isinstance(record, dict):
        return False, f"{source_label}: record is not a dict"

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return False, f"{source_label}: messages must be a 3-element list"
    for idx, (msg, expected_role) in enumerate(zip(messages, REQUIRED_ROLES)):
        if not isinstance(msg, dict):
            return False, f"{source_label}: messages[{idx}] is not a dict"
        if msg.get("role") != expected_role:
            return (
                False,
                f"{source_label}: messages[{idx}] role={msg.get('role')!r} expected={expected_role!r}",
            )
        if not isinstance(msg.get("content"), str):
            return (
                False,
                f"{source_label}: messages[{idx}] content is not a string",
            )

    category = record.get("category")
    if not isinstance(category, str) or not category:
        return False, f"{source_label}: category is not a non-empty string"

    tools = record.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            return False, f"{source_label}: tools must be null or list[str]"
        if not all(isinstance(t, str) for t in tools):
            return False, f"{source_label}: tools must be null or list[str]"

    source = record.get("source")
    if not isinstance(source, str) or not source:
        return False, f"{source_label}: source is not a non-empty string"
    if allowed_sources is not None and source not in allowed_sources:
        return (
            False,
            f"{source_label}: source={source!r} not in allowed set "
            f"{sorted(allowed_sources)!r}",
        )

    return True, ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def merge_corpora(
    v4_records: list[dict[str, Any]],
    v21_records: list[dict[str, Any]],
    *,
    drop_categories: frozenset[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Return a new shuffled list. Inputs are not mutated."""
    # Validate v4 (source tag must be in V4_ALLOWED_SOURCES — current Gemini
    # tag or legacy Claude tag).
    for idx, rec in enumerate(v4_records):
        ok, reason = validate_record(
            rec,
            source_label=f"v4[{idx}]",
            allowed_sources=V4_ALLOWED_SOURCES,
        )
        if not ok:
            raise ValueError(reason)

    # Validate + filter v2.1
    kept_v21: list[dict[str, Any]] = []
    for idx, rec in enumerate(v21_records):
        ok, reason = validate_record(rec, source_label=f"v2.1[{idx}]")
        if not ok:
            raise ValueError(reason)
        if rec["category"] in drop_categories:
            continue
        kept_v21.append(rec)

    # Build merged list as a fresh sequence (no mutation of inputs).
    merged: list[dict[str, Any]] = [*v4_records, *kept_v21]

    # Shuffle deterministically.
    rng = random.Random(seed)
    indices = list(range(len(merged)))
    rng.shuffle(indices)
    shuffled = [merged[i] for i in indices]
    return shuffled


# ---------------------------------------------------------------------------
# Public entry point (for programmatic use; what tests call)
# ---------------------------------------------------------------------------


def merge_corpus(
    *,
    v4_path: Path,
    v21_path: Path,
    output_path: Path,
    seed: int = 42,
    dry_run: bool = False,
    drop_categories: frozenset[str] = DROP_CATEGORIES_FROM_V21,
) -> list[dict[str, Any]]:
    """High-level merge entry point.

    Loads both JSONL files, drops ``conditional_no_tool`` (or whatever set is
    passed via ``drop_categories``) from the v2.1 side, validates schema on
    everything, shuffles deterministically with ``seed``, and writes to
    ``output_path`` unless ``dry_run`` is True. Returns the merged list.

    Keyword-only so call sites are unambiguous at the merge site.
    """
    v4 = load_jsonl(v4_path)
    v21 = load_jsonl(v21_path)
    merged = merge_corpora(
        v4,
        v21,
        drop_categories=drop_categories,
        seed=seed,
    )
    if not dry_run:
        write_jsonl(output_path, merged)
    return merged


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def summarize(label: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(r["category"] for r in records)
    total = len(records)
    cap_denial = sum(counts[c] for c in CAP_DENIAL_CATEGORIES)
    pct = (100.0 * cap_denial / total) if total else 0.0
    logger.info("%s — %d records", label, total)
    for cat, cnt in counts.most_common():
        logger.info("  %-32s %d", cat, cnt)
    logger.info("  %% capability_denial: %.1f%%", pct)
    return {
        "label": label,
        "total": total,
        "counts": dict(counts),
        "pct_capability_denial": pct,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge v4 + v2.1 (minus conditional_no_tool) → synthetic_v2.4.jsonl",
    )
    parser.add_argument(
        "--v4-input",
        type=Path,
        default=Path("data/finetune/synthetic_v4.jsonl"),
        help="Path to Gemini-generated v4 JSONL "
        "(legacy synthetic_v4_claude.jsonl also accepted by source-tag policy).",
    )
    parser.add_argument(
        "--v2.1-input",
        dest="v21_input",
        type=Path,
        default=Path("data/finetune/synthetic_v2.1.jsonl"),
        help="Path to v2.1 SFT JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/finetune/synthetic_v2.4.jsonl"),
        help="Output path for merged corpus.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count records and print stats; do not write the output file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(message)s",
    )

    try:
        v4 = load_jsonl(args.v4_input)
        v21 = load_jsonl(args.v21_input)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Load failure: %s", exc)
        return 2

    summarize("v4 (teacher)", v4)
    summarize("v2.1 (raw)", v21)

    try:
        merged = merge_corpora(
            v4,
            v21,
            drop_categories=DROP_CATEGORIES_FROM_V21,
            seed=args.seed,
        )
    except ValueError as exc:
        logger.error("Merge failure: %s", exc)
        return 3

    summary = summarize("merged v2.4", merged)

    if args.dry_run:
        logger.info("Dry run — not writing %s", args.output)
        return 0

    n_written = write_jsonl(args.output, merged)
    logger.info("Wrote %d records to %s", n_written, args.output)
    logger.info("Final cap-denial coverage: %.1f%%", summary["pct_capability_denial"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
