"""
Tests for scripts/setup_wizard.py (Wave J0).

All tests use tmp_path for file operations so nothing is written to the real
repo tree.  builtins.input is mocked wherever the wizard would prompt
interactively.

Test coverage:
- test_creates_missing_dirs            — wizard creates models/llm/, data/, eval_results/
- test_model_found_skips_prompt        — model present → exits 0, no prompt
- test_model_missing_yes_flag_exits_1  — model absent + --yes → exits 1
- test_config_writeback                — new model path written back to config.yaml
- test_noninteractive_known_good_config — --yes + model present → exits 0
- test_read_model_path_fallback        — missing config → default path returned
- test_read_model_path_from_llm_section — correct llm.model_path parsed
- test_write_model_path_creates_file   — write-back creates config if absent
- test_write_model_path_replaces_existing — write-back updates existing key
- test_prompt_yes_mode_returns_default  — _prompt returns default when yes=True
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal config.yaml in tmp_path
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, model_path: str = "models/llm/phi-3.5-mini.gguf") -> Path:
    """Write a minimal config.yaml under tmp_path and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            # Lumi config
            edition: standard
            llm:
              model_path: {model_path}
            """
        ),
        encoding="utf-8",
    )
    return cfg


# ---------------------------------------------------------------------------
# test_creates_missing_dirs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_creates_missing_dirs(tmp_path: Path) -> None:
    """Wizard creates models/llm/, data/, and eval_results/ when absent."""
    from scripts.setup_wizard import run_setup

    # Create a config that points at a model file that actually exists.
    model_file = tmp_path / "models" / "llm" / "model.gguf"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()

    cfg = _make_config(tmp_path, model_path="models/llm/model.gguf")

    # Run from tmp_path so relative paths resolve there.
    import os

    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        rc = run_setup(yes=True, config_path=str(cfg))
    finally:
        os.chdir(orig)

    assert rc == 0
    assert (tmp_path / "models" / "llm").is_dir()
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "eval_results").is_dir()


# ---------------------------------------------------------------------------
# test_model_found_skips_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_found_skips_prompt(tmp_path: Path) -> None:
    """When the model file exists, the wizard exits 0 without asking for input."""
    from scripts.setup_wizard import run_setup

    model_file = tmp_path / "models" / "llm" / "model.gguf"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()

    cfg = _make_config(tmp_path, model_path="models/llm/model.gguf")

    import os

    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        with patch("builtins.input") as mock_input:
            rc = run_setup(yes=False, config_path=str(cfg))
    finally:
        os.chdir(orig)

    assert rc == 0
    # input() must NOT have been called because the model was already present.
    mock_input.assert_not_called()


# ---------------------------------------------------------------------------
# test_model_missing_yes_flag_exits_1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_missing_yes_flag_exits_1(tmp_path: Path) -> None:
    """When model is missing and --yes is set, wizard exits 1 (can't auto-download)."""
    from scripts.setup_wizard import run_setup

    # Model file does NOT exist.
    cfg = _make_config(tmp_path, model_path="models/llm/nonexistent.gguf")

    import os

    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        rc = run_setup(yes=True, config_path=str(cfg))
    finally:
        os.chdir(orig)

    assert rc == 1


# ---------------------------------------------------------------------------
# test_config_writeback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_writeback(tmp_path: Path) -> None:
    """When user provides a new model path, config.yaml is updated."""
    from scripts.setup_wizard import run_setup

    # The new model file exists at a different location.
    new_model = tmp_path / "custom" / "my_model.gguf"
    new_model.parent.mkdir(parents=True, exist_ok=True)
    new_model.touch()

    # Config points at a non-existent path so the wizard prompts.
    cfg = _make_config(tmp_path, model_path="models/llm/missing.gguf")

    import os

    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Simulate user entering the path to the new model file.
        with patch("builtins.input", return_value=str(new_model)):
            rc = run_setup(yes=False, config_path=str(cfg))
    finally:
        os.chdir(orig)

    assert rc == 0
    updated = cfg.read_text(encoding="utf-8")
    assert str(new_model) in updated


# ---------------------------------------------------------------------------
# test_noninteractive_known_good_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_noninteractive_known_good_config(tmp_path: Path) -> None:
    """With --yes and a valid model present, wizard exits 0."""
    from scripts.setup_wizard import run_setup

    model_file = tmp_path / "models" / "llm" / "phi-3.5-mini.gguf"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()

    cfg = _make_config(tmp_path, model_path="models/llm/phi-3.5-mini.gguf")

    import os

    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        rc = run_setup(yes=True, config_path=str(cfg))
    finally:
        os.chdir(orig)

    assert rc == 0


# ---------------------------------------------------------------------------
# test_read_model_path_fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_model_path_fallback(tmp_path: Path) -> None:
    """When config.yaml is absent, _read_model_path_from_config returns the default."""
    from scripts.setup_wizard import _read_model_path_from_config

    missing_cfg = str(tmp_path / "nonexistent.yaml")
    result = _read_model_path_from_config(missing_cfg)
    assert result == "models/llm/phi-3.5-mini.gguf"


# ---------------------------------------------------------------------------
# test_read_model_path_from_llm_section
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_model_path_from_llm_section(tmp_path: Path) -> None:
    """_read_model_path_from_config returns the value under the llm section."""
    from scripts.setup_wizard import _read_model_path_from_config

    cfg = _make_config(tmp_path, model_path="models/llm/custom.gguf")
    result = _read_model_path_from_config(str(cfg))
    assert result == "models/llm/custom.gguf"


# ---------------------------------------------------------------------------
# test_write_model_path_creates_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_model_path_creates_file(tmp_path: Path) -> None:
    """_write_model_path_to_config creates config.yaml when it does not exist."""
    from scripts.setup_wizard import _write_model_path_to_config

    cfg_path = str(tmp_path / "config.yaml")
    _write_model_path_to_config(cfg_path, "models/llm/new.gguf")

    content = Path(cfg_path).read_text(encoding="utf-8")
    assert "models/llm/new.gguf" in content


# ---------------------------------------------------------------------------
# test_write_model_path_replaces_existing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_model_path_replaces_existing(tmp_path: Path) -> None:
    """_write_model_path_to_config updates model_path in an existing config."""
    from scripts.setup_wizard import _write_model_path_to_config

    cfg = _make_config(tmp_path, model_path="models/llm/old.gguf")
    _write_model_path_to_config(str(cfg), "models/llm/new.gguf")

    content = cfg.read_text(encoding="utf-8")
    assert "models/llm/new.gguf" in content
    # The old path must no longer appear.
    assert "old.gguf" not in content


# ---------------------------------------------------------------------------
# test_prompt_yes_mode_returns_default
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prompt_yes_mode_returns_default() -> None:
    """_prompt returns the default immediately when yes=True."""
    from scripts.setup_wizard import _prompt

    result = _prompt("Choose path", default="/tmp/model.gguf", yes=True)
    assert result == "/tmp/model.gguf"
