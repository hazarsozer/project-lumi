"""
Tests for src/core/config_writer.py (Wave S0).

Covers:
- write_config() produces a valid YAML that round-trips through load_config()
- Atomic write: tmp file is renamed to target (target exists, tmp gone)
- .bak file is created when the target file already exists
- None-valued optional fields (kv_cache_quant, persona.system_prompt,
  initial_prompt) are omitted from YAML output
- ConfigWriteError is raised when os.replace raises OSError
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import LumiConfig, LLMConfig, PersonaConfig, ScribeConfig
from src.core.config_writer import ConfigWriteError, write_config


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_round_trips_through_load_config(tmp_path: Path) -> None:
    """write_config() + load_config() must produce an equivalent config."""
    from src.core.config import load_config

    original = LumiConfig(
        edition="pro",
        log_level="DEBUG",
        json_logs=True,
    )
    target = tmp_path / "config.yaml"
    write_config(original, str(target))

    loaded = load_config(str(target))

    assert loaded.edition == "pro"
    assert loaded.log_level == "DEBUG"
    assert loaded.json_logs is True
    # Sub-sections should match defaults since we didn't override them.
    assert loaded.audio.sample_rate == original.audio.sample_rate
    assert loaded.llm.temperature == pytest.approx(original.llm.temperature)
    assert loaded.ipc.port == original.ipc.port


@pytest.mark.unit
def test_write_config_preserves_sub_section_values(tmp_path: Path) -> None:
    """Modified sub-section values must survive the round-trip."""
    from src.core.config import load_config, AudioConfig

    config = LumiConfig(
        audio=AudioConfig(sensitivity=0.42, vad_threshold=0.33),
    )
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    loaded = load_config(str(target))
    assert loaded.audio.sensitivity == pytest.approx(0.42)
    assert loaded.audio.vad_threshold == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_target_exists_tmp_is_gone(tmp_path: Path) -> None:
    """After write_config, target must exist and the .tmp file must be gone."""
    target = tmp_path / "config.yaml"
    tmp_file = tmp_path / "config.yaml.tmp"

    write_config(LumiConfig(), str(target))

    assert target.exists(), "Target file must exist after write_config."
    assert not tmp_file.exists(), ".tmp file must be cleaned up after rename."


@pytest.mark.unit
def test_write_config_target_is_valid_yaml(tmp_path: Path) -> None:
    """The written file must be parseable as YAML."""
    import yaml

    target = tmp_path / "config.yaml"
    write_config(LumiConfig(), str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert "edition" in raw
    assert "audio" in raw


# ---------------------------------------------------------------------------
# Backup (.bak)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_creates_bak_when_target_exists(tmp_path: Path) -> None:
    """A .bak file must be created when the target file already exists."""
    target = tmp_path / "config.yaml"
    bak = tmp_path / "config.yaml.bak"

    # Write an initial file.
    target.write_text("edition: light\n", encoding="utf-8")
    assert not bak.exists()

    write_config(LumiConfig(edition="pro"), str(target))

    assert bak.exists(), ".bak file must exist after overwriting an existing target."
    # The .bak should contain the OLD content.
    assert "light" in bak.read_text(encoding="utf-8")


@pytest.mark.unit
def test_write_config_no_bak_when_target_absent(tmp_path: Path) -> None:
    """No .bak file should be created when the target does not yet exist."""
    target = tmp_path / "config.yaml"
    bak = tmp_path / "config.yaml.bak"

    write_config(LumiConfig(), str(target))

    assert not bak.exists()


# ---------------------------------------------------------------------------
# Omission of None-valued optional fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_omits_kv_cache_quant_when_none(tmp_path: Path) -> None:
    """kv_cache_quant must not appear in the YAML when it is None."""
    import yaml

    config = LumiConfig(llm=LLMConfig(kv_cache_quant=None))
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "kv_cache_quant" not in raw.get("llm", {})


@pytest.mark.unit
def test_write_config_includes_kv_cache_quant_when_set(tmp_path: Path) -> None:
    """kv_cache_quant must appear in the YAML when it has a value."""
    import yaml

    config = LumiConfig(llm=LLMConfig(kv_cache_quant="turbo3"))
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert raw["llm"]["kv_cache_quant"] == "turbo3"


@pytest.mark.unit
def test_write_config_omits_persona_system_prompt_when_none(tmp_path: Path) -> None:
    """persona.system_prompt must not appear in the YAML when it is None."""
    import yaml

    config = LumiConfig(persona=PersonaConfig(system_prompt=None))
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    # Either the key is absent or the persona section is absent entirely.
    persona_section = raw.get("persona", {})
    assert "system_prompt" not in persona_section


@pytest.mark.unit
def test_write_config_includes_persona_system_prompt_when_set(
    tmp_path: Path,
) -> None:
    """persona.system_prompt must appear in the YAML when it has a value."""
    import yaml

    config = LumiConfig(persona=PersonaConfig(system_prompt="You are Lumi."))
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert raw["persona"]["system_prompt"] == "You are Lumi."


@pytest.mark.unit
def test_write_config_omits_scribe_initial_prompt_when_none(
    tmp_path: Path,
) -> None:
    """scribe.initial_prompt must not appear in the YAML when it is None."""
    import yaml

    config = LumiConfig(scribe=ScribeConfig(initial_prompt=None))
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    scribe_section = raw.get("scribe", {})
    assert "initial_prompt" not in scribe_section


@pytest.mark.unit
def test_write_config_includes_scribe_initial_prompt_when_set(
    tmp_path: Path,
) -> None:
    """scribe.initial_prompt must appear in the YAML when it has a value."""
    import yaml

    config = LumiConfig(scribe=ScribeConfig(initial_prompt="Lumi, Firefox"))
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert raw["scribe"]["initial_prompt"] == "Lumi, Firefox"


# ---------------------------------------------------------------------------
# ConfigWriteError on IO failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_raises_config_write_error_on_replace_failure(
    tmp_path: Path,
) -> None:
    """ConfigWriteError must be raised when os.replace raises OSError."""
    target = tmp_path / "config.yaml"

    with patch("os.replace", side_effect=OSError("disk full")):
        with pytest.raises(ConfigWriteError, match="disk full"):
            write_config(LumiConfig(), str(target))


@pytest.mark.unit
def test_config_write_error_is_exception_subclass() -> None:
    """ConfigWriteError must be a subclass of Exception."""
    assert issubclass(ConfigWriteError, Exception)


# ---------------------------------------------------------------------------
# allowed_tools tuple → list round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_allowed_tools_round_trips_as_list(tmp_path: Path) -> None:
    """ToolsConfig.allowed_tools (tuple) must be written as a YAML list and
    load back correctly as a tuple through load_config."""
    import yaml
    from src.core.config import load_config, ToolsConfig

    config = LumiConfig(
        tools=ToolsConfig(allowed_tools=("launch_app", "clipboard"))
    )
    target = tmp_path / "config.yaml"
    write_config(config, str(target))

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(raw["tools"]["allowed_tools"], list)

    # Round-trip via load_config.
    loaded = load_config(str(target))
    assert isinstance(loaded.tools.allowed_tools, tuple)
    assert loaded.tools.allowed_tools == ("launch_app", "clipboard")
