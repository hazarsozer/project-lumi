"""
Tests for ipc.address loopback-only enforcement (GitHub Issue #7 / CR-06).

Covers:
- _validate_ipc_address: loopback addresses accepted
- _validate_ipc_address: non-loopback rejected by default
- _validate_ipc_address: non-loopback allowed when allow_non_loopback=True
- load_config: 0.0.0.0 in config dict raises ValueError
- load_config: localhost accepted
- load_config: allow_non_loopback=True allows 0.0.0.0
- ConfigManager.apply: setting ipc.address to 0.0.0.0 is rejected (default)
- ConfigManager.apply: setting ipc.address to loopback is accepted
- ConfigManager.apply: setting ipc.address to non-loopback accepted when
  allow_non_loopback=True already in config
- ConfigManager.apply: setting ipc.address to non-loopback accepted when
  allow_non_loopback=True is in the SAME batch
- FIELD_META: ipc.allow_non_loopback is control=="toggle", NOT "path"
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src.core.config import (
    IPCConfig,
    LumiConfig,
    _validate_ipc_address,
    load_config,
)
from src.core.config_runtime import ConfigManager
from src.core.config_schema import FIELD_META


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_manager() -> ConfigManager:
    """ConfigManager seeded with the all-defaults LumiConfig."""
    return ConfigManager(LumiConfig())


@pytest.fixture
def non_loopback_allowed_manager() -> ConfigManager:
    """ConfigManager where ipc.allow_non_loopback is already True."""
    ipc = IPCConfig(allow_non_loopback=True)
    return ConfigManager(LumiConfig(ipc=ipc))


# ---------------------------------------------------------------------------
# _validate_ipc_address — unit tests (pure function)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "localhost", "::1"],
)
def test_loopback_addresses_accepted(address: str) -> None:
    """Loopback addresses must not raise regardless of allow_non_loopback."""
    # Default (opt-in flag off)
    _validate_ipc_address(address, allow_non_loopback=False)
    # And with the flag on for good measure
    _validate_ipc_address(address, allow_non_loopback=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "address",
    ["0.0.0.0", "192.168.1.1", "10.0.0.1", "0:0:0:0:0:0:0:0", "example.com"],
)
def test_non_loopback_rejected_by_default(address: str) -> None:
    """Non-loopback addresses must raise ValueError when opt-in is False."""
    with pytest.raises(ValueError, match="not a loopback address"):
        _validate_ipc_address(address, allow_non_loopback=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "address",
    ["0.0.0.0", "192.168.1.1", "10.0.0.1"],
)
def test_non_loopback_allowed_with_opt_in(address: str) -> None:
    """Non-loopback addresses must NOT raise when allow_non_loopback=True."""
    # Should not raise — just logs a warning.
    _validate_ipc_address(address, allow_non_loopback=True)


@pytest.mark.unit
def test_error_message_mentions_opt_in_flag() -> None:
    """The ValueError message must hint at the opt-in flag name."""
    with pytest.raises(ValueError, match="allow_non_loopback"):
        _validate_ipc_address("0.0.0.0", allow_non_loopback=False)


# ---------------------------------------------------------------------------
# load_config — startup / hot-reload path (RED → GREEN regression)
# ---------------------------------------------------------------------------


def _make_ipc_yaml(**ipc_overrides: Any) -> str:
    """Return minimal YAML content with the given ipc section overrides."""
    ipc_block: dict[str, Any] = {
        "enabled": False,
        "address": "127.0.0.1",
        "port": 5556,
        "token_path": "~/.lumi/ipc_token",
        "allow_non_loopback": False,
        **ipc_overrides,
    }
    return yaml.dump({"ipc": ipc_block})


@pytest.mark.unit
def test_load_config_rejects_non_loopback_by_default(tmp_path: Path) -> None:
    """load_config must raise ValueError when ipc.address is 0.0.0.0 (default)."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_make_ipc_yaml(address="0.0.0.0"))

    with pytest.raises(ValueError, match="not a loopback address"):
        load_config(str(cfg_file))


@pytest.mark.unit
@pytest.mark.parametrize("address", ["127.0.0.1", "localhost", "::1"])
def test_load_config_accepts_loopback_addresses(tmp_path: Path, address: str) -> None:
    """load_config must succeed for any loopback address."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_make_ipc_yaml(address=address))
    cfg = load_config(str(cfg_file))
    assert cfg.ipc.address == address


@pytest.mark.unit
def test_load_config_allows_non_loopback_with_opt_in(tmp_path: Path) -> None:
    """load_config must succeed when 0.0.0.0 and allow_non_loopback=true."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        _make_ipc_yaml(address="0.0.0.0", allow_non_loopback=True)
    )
    cfg = load_config(str(cfg_file))
    assert cfg.ipc.address == "0.0.0.0"
    assert cfg.ipc.allow_non_loopback is True


# ---------------------------------------------------------------------------
# ConfigManager.apply — runtime change path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_non_loopback_address_rejected_by_default(
    default_manager: ConfigManager,
) -> None:
    """Applying ipc.address=0.0.0.0 must return an error (default config)."""
    result = default_manager.apply({"ipc.address": "0.0.0.0"})

    assert "ipc.address" in result.errors
    assert result.applied_live == []
    assert result.pending_restart == []
    # Config must not have changed.
    assert default_manager.current.ipc.address == "127.0.0.1"


@pytest.mark.unit
@pytest.mark.parametrize("address", ["127.0.0.1", "localhost", "::1"])
def test_apply_loopback_address_accepted(
    default_manager: ConfigManager, address: str
) -> None:
    """Applying a loopback ipc.address must succeed without errors."""
    result = default_manager.apply({"ipc.address": address})

    assert "ipc.address" not in result.errors
    assert "ipc.address" in result.pending_restart  # restart_required=True


@pytest.mark.unit
def test_apply_non_loopback_allowed_when_config_already_has_opt_in(
    non_loopback_allowed_manager: ConfigManager,
) -> None:
    """0.0.0.0 must be accepted when current config has allow_non_loopback=True."""
    result = non_loopback_allowed_manager.apply({"ipc.address": "0.0.0.0"})

    assert "ipc.address" not in result.errors
    assert "ipc.address" in result.pending_restart


@pytest.mark.unit
def test_apply_non_loopback_allowed_when_same_batch_enables_opt_in(
    default_manager: ConfigManager,
) -> None:
    """Sending ipc.allow_non_loopback=True and ipc.address=0.0.0.0 in a
    single batch must succeed — the effective opt-in from the batch is used."""
    result = default_manager.apply(
        {"ipc.address": "0.0.0.0", "ipc.allow_non_loopback": True}
    )

    assert "ipc.address" not in result.errors
    assert result.errors == {}


@pytest.mark.unit
def test_apply_non_loopback_error_message_hints_opt_in(
    default_manager: ConfigManager,
) -> None:
    """The rejection error message must mention the opt-in flag."""
    result = default_manager.apply({"ipc.address": "0.0.0.0"})

    error_msg = result.errors.get("ipc.address", "")
    assert "allow_non_loopback" in error_msg


# ---------------------------------------------------------------------------
# FIELD_META: ipc.allow_non_loopback must NOT be a "path" control
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ipc_allow_non_loopback_is_toggle_not_path() -> None:
    """ipc.allow_non_loopback must use control='toggle', NOT 'path'.

    The event_bridge derives its _WIRE_BLOCKED_PATH_KEYS set from
    FIELD_META entries with control=='path'.  A bool flag must never be
    marked as 'path' or it would be mistakenly added to that block-set.
    """
    meta = FIELD_META["ipc.allow_non_loopback"]
    assert meta["control"] == "toggle"
    assert meta["control"] != "path"


@pytest.mark.unit
def test_ipc_allow_non_loopback_in_field_meta() -> None:
    """ipc.allow_non_loopback must be present in FIELD_META."""
    assert "ipc.allow_non_loopback" in FIELD_META


@pytest.mark.unit
def test_ipc_allow_non_loopback_is_restart_required() -> None:
    """ipc.allow_non_loopback must require a restart (same as ipc.address)."""
    assert FIELD_META["ipc.allow_non_loopback"]["restart_required"] is True
