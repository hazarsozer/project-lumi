"""
Atomic YAML writer for LumiConfig (Wave S0).

Converts a frozen ``LumiConfig`` dataclass back to a nested plain-dict
structure, then writes it to disk via PyYAML.  An atomic
write-then-rename pattern guarantees that a concurrent reader never sees
a partial file: the data is written to a ``.tmp`` sibling, fsync'd, and
then ``os.replace()``'d over the target path in a single system call.

Before overwriting the target, an existing file is copied to ``<path>.bak``
so the user can recover the previous config if needed.

Optional-field exclusions
--------------------------
- ``llm.kv_cache_quant``      — omitted when ``None``
- ``persona.system_prompt``   — omitted when ``None``
- ``scribe.initial_prompt``   — omitted when ``None``
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.core.config import LumiConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class ConfigWriteError(Exception):
    """Raised when the config YAML cannot be written to disk."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Fields that should be omitted from the YAML output when their value is None.
# Keyed as "<section>.<field>" or "<field>" for top-level scalars.
_OMIT_WHEN_NONE: frozenset[str] = frozenset(
    [
        "llm.kv_cache_quant",
        "persona.system_prompt",
        "scribe.initial_prompt",
    ]
)


def _config_to_dict(config: LumiConfig) -> dict[str, Any]:
    """Convert a frozen ``LumiConfig`` to a plain nested dict for YAML output.

    The conversion:
    - Recursively replaces frozen sub-dataclasses with plain dicts.
    - Converts ``tuple`` values (e.g. ``allowed_tools``) to ``list``.
    - Omits fields listed in ``_OMIT_WHEN_NONE`` when their value is ``None``.

    Args:
        config: The config object to serialize.

    Returns:
        A plain nested ``dict`` ready to be serialized to YAML.
    """
    out: dict[str, Any] = {}

    # Top-level scalar fields on LumiConfig.
    top_scalar_fields = {"edition", "log_level", "json_logs"}
    for f in dataclasses.fields(config):
        if f.name in top_scalar_fields:
            out[f.name] = getattr(config, f.name)

    # Sub-section dataclasses — iterate in the order they appear on LumiConfig.
    section_names = [
        f.name for f in dataclasses.fields(config) if f.name not in top_scalar_fields
    ]

    for section_name in section_names:
        sub_cfg = getattr(config, section_name)
        section_dict: dict[str, Any] = {}
        for sub_field in dataclasses.fields(sub_cfg):
            dotted_key = f"{section_name}.{sub_field.name}"
            value = getattr(sub_cfg, sub_field.name)

            # Omit None-valued optional fields.
            if value is None and dotted_key in _OMIT_WHEN_NONE:
                continue

            # Convert tuple → list so PyYAML writes a YAML sequence.
            if isinstance(value, tuple):
                value = list(value)

            section_dict[sub_field.name] = value

        out[section_name] = section_dict

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_config(config: LumiConfig, path: str = "config.yaml") -> None:
    """Atomically write a ``LumiConfig`` instance to a YAML file.

    Steps:
    1. Convert ``config`` to a plain nested dict.
    2. If the target file already exists, copy it to ``<path>.bak``.
    3. Write the dict to ``<path>.tmp`` via PyYAML.
    4. ``fsync`` the file descriptor to flush kernel buffers.
    5. ``os.replace()`` the tmp file over the target path.

    Args:
        config: The config object to persist.
        path:   Target YAML file path.  Relative paths are resolved from the
                current working directory.

    Raises:
        ConfigWriteError: If any IO operation fails.
    """
    target = Path(path)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    bak_path = target.with_suffix(target.suffix + ".bak")

    data = _config_to_dict(config)

    # ------------------------------------------------------------------
    # Step 1: Back up the existing file.
    # ------------------------------------------------------------------
    if target.exists():
        try:
            shutil.copy2(target, bak_path)
            logger.debug("Config backup written to '%s'.", bak_path)
        except OSError as exc:
            raise ConfigWriteError(
                f"Failed to create config backup at '{bak_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Step 2: Write to .tmp using PyYAML.
    # ------------------------------------------------------------------
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            yaml.dump(
                data,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
            fh.flush()
            os.fsync(fh.fileno())
        logger.debug("Config temp file written to '%s'.", tmp_path)
    except OSError as exc:
        # Clean up orphaned tmp file on error.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ConfigWriteError(
            f"Failed to write config temp file at '{tmp_path}': {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Step 3: Atomic rename.
    # ------------------------------------------------------------------
    try:
        os.replace(tmp_path, target)
        logger.info("Config saved to '%s'.", target)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ConfigWriteError(
            f"Failed to replace config file '{target}' with tmp: {exc}"
        ) from exc
