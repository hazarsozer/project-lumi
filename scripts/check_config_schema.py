#!/usr/bin/env python3
"""
CI script to detect drift between FIELD_META and actual LumiConfig fields.

Ensures the Settings UI schema stays in sync with the actual config structure.

Run from repo root:
    uv run python scripts/check_config_schema.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Add repo root to path so imports work
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.core.config_schema import FIELD_META
from src.core.config import (
    LumiConfig,
    AudioConfig,
    ScribeConfig,
    LLMConfig,
    TTSConfig,
    IPCConfig,
    ToolsConfig,
    VisionConfig,
    RAGConfig,
    PersonaConfig,
)


# Fields intentionally excluded from FIELD_META (not user-facing in Settings UI)
INTENTIONALLY_EXCLUDED: frozenset[str] = frozenset({"llm.kv_cache_quant"})

# Map section names to their dataclass types
SECTION_TYPES: dict[str, type] = {
    "audio": AudioConfig,
    "scribe": ScribeConfig,
    "llm": LLMConfig,
    "tts": TTSConfig,
    "ipc": IPCConfig,
    "tools": ToolsConfig,
    "vision": VisionConfig,
    "rag": RAGConfig,
    "persona": PersonaConfig,
}

# Top-level scalar fields in LumiConfig (not part of a section)
TOP_LEVEL_SCALARS: frozenset[str] = frozenset({"edition", "log_level", "json_logs"})


def build_config_keys() -> frozenset[str]:
    """Extract all dotted-path keys from LumiConfig dataclass structure."""
    keys = set()

    # Top-level scalar fields
    keys.update(TOP_LEVEL_SCALARS)

    # Section fields: "section.field"
    for section_name, section_type in SECTION_TYPES.items():
        for field in dataclasses.fields(section_type):
            keys.add(f"{section_name}.{field.name}")

    return frozenset(keys)


def build_schema_keys() -> frozenset[str]:
    """Extract all keys currently in FIELD_META."""
    return frozenset(FIELD_META.keys())


def main() -> int:
    """Check for drift and return exit code."""
    config_keys = build_config_keys()
    schema_keys = build_schema_keys()

    # Calculate drift
    missing_from_schema = config_keys - INTENTIONALLY_EXCLUDED - schema_keys
    stale_in_schema = schema_keys - config_keys

    # Report results
    if not missing_from_schema and not stale_in_schema:
        total_fields = len(config_keys - INTENTIONALLY_EXCLUDED)
        print(
            f"✓ No schema drift detected. FIELD_META covers all {total_fields} user-facing fields."
        )
        return 0
    else:
        print("✗ Schema drift detected!")
        if missing_from_schema:
            print(f"\nMissing from FIELD_META ({len(missing_from_schema)}):")
            for key in sorted(missing_from_schema):
                print(f"  - {key}")
        if stale_in_schema:
            print(f"\nStale in FIELD_META ({len(stale_in_schema)}):")
            for key in sorted(stale_in_schema):
                print(f"  - {key}")
        return 1


if __name__ == "__main__":
    exit(main())
