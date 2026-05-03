"""Unit tests for scripts/synth_dataset.py.

These tests run the script as a subprocess into a temporary output path so
they do not depend on, or clobber, the committed dataset at
``data/finetune/synthetic_v0.jsonl``.  No live LLM or network is required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "synth_dataset.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the script once per module into a temp path and return the file."""
    out = tmp_path_factory.mktemp("finetune") / "synthetic_test.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--count", "1000"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"synth_dataset.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return out


@pytest.fixture(scope="module")
def records(generated_dataset: Path) -> list[dict]:
    """Parse the generated JSONL into a list of records (once per module)."""
    return [json.loads(line) for line in generated_dataset.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_file_created(generated_dataset: Path) -> None:
    """The script creates a non-empty JSONL file at the requested path."""
    assert generated_dataset.exists()
    assert generated_dataset.stat().st_size > 0


@pytest.mark.unit
def test_line_count_in_range(generated_dataset: Path) -> None:
    """The generated file has between 1000 and 1200 lines inclusive."""
    lines = generated_dataset.read_text().splitlines()
    assert 1000 <= len(lines) <= 1200, f"unexpected line count: {len(lines)}"


@pytest.mark.unit
def test_all_lines_valid_json(generated_dataset: Path) -> None:
    """Every line of the JSONL parses as a JSON object."""
    for i, line in enumerate(generated_dataset.read_text().splitlines(), start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"line {i} is not valid JSON: {exc}\n{line[:200]}")
        assert isinstance(obj, dict), f"line {i} did not decode to a dict"


@pytest.mark.unit
def test_required_keys_present(records: list[dict]) -> None:
    """Each record has the three required top-level keys."""
    required = {"messages", "category", "source"}
    for i, rec in enumerate(records):
        missing = required - rec.keys()
        assert not missing, f"record {i} is missing keys: {missing}"
        assert rec["source"] == "synthetic_v1"


@pytest.mark.unit
def test_six_categories(records: list[dict]) -> None:
    """At least 6 distinct category labels appear in the dataset."""
    categories = {r["category"] for r in records}
    assert len(categories) >= 6, f"only {len(categories)} distinct categories: {categories}"


@pytest.mark.unit
def test_chatML_structure(records: list[dict]) -> None:
    """Each record's messages list follows the system/user/assistant order."""
    for i, rec in enumerate(records):
        msgs = rec["messages"]
        assert isinstance(msgs, list), f"record {i} messages is not a list"
        assert len(msgs) == 3, f"record {i} has {len(msgs)} messages, expected 3"

        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], (
            f"record {i} has role order {roles}, expected system/user/assistant"
        )

        for m in msgs:
            assert "content" in m, f"record {i} message missing 'content'"
            assert isinstance(m["content"], str), f"record {i} content not str"

        # System prompt must be non-trivial — otherwise the persona reference
        # was lost somewhere.
        assert len(msgs[0]["content"]) > 100
        # User and assistant turns must be non-empty.
        assert msgs[1]["content"].strip()
        assert msgs[2]["content"].strip()
