"""
Unit tests for scripts/smoke_live.py stage logic.

These tests verify the script's control-flow behaviour without loading any real
models.  All model-loading functions and subprocess calls are mocked at the
import boundary.

Marking: @pytest.mark.unit — no hardware, no model files, no network.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so `scripts` is importable.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — import the module under test
# ---------------------------------------------------------------------------

import importlib
import types


def _load_smoke_live():
    """Import scripts.smoke_live as a module (handles the scripts/ directory)."""
    scripts_dir = _PROJECT_ROOT / "scripts"
    spec = importlib.util.spec_from_file_location(
        "smoke_live", scripts_dir / "smoke_live.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so dataclass decorator can resolve
    # __module__ — required for Python 3.12+ dataclass machinery.
    sys.modules["smoke_live"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. test_smoke_stage_times_out
#    A stage function that sleeps past its threshold must return SmokeResult.FAIL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_smoke_stage_times_out():
    """A stage whose operation exceeds its threshold must return FAIL."""
    smoke = _load_smoke_live()

    def _slow_fn():
        time.sleep(0.2)  # will exceed a 50 ms threshold
        return "result"

    result = smoke.run_stage(
        name="TestStage",
        fn=_slow_fn,
        threshold_ms=50,
        validate=lambda r: r is not None,
    )

    assert result.status == "FAIL", f"Expected FAIL, got {result.status!r}"
    assert "timeout" in result.message.lower(), (
        f"Expected 'timeout' in message, got {result.message!r}"
    )


# ---------------------------------------------------------------------------
# 2. test_smoke_skip_flag_skips_stage
#    --skip-llm must cause the LLM stage to return SKIP
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_smoke_skip_flag_skips_stage():
    """run_stage with skip=True must return SKIP without calling fn."""
    smoke = _load_smoke_live()

    called = []

    def _fn():
        called.append(True)
        return "value"

    result = smoke.run_stage(
        name="LLM",
        fn=_fn,
        threshold_ms=10_000,
        validate=lambda r: bool(r),
        skip=True,
    )

    assert result.status == "SKIP"
    assert called == [], "fn must not be called when skip=True"


# ---------------------------------------------------------------------------
# 3. test_smoke_exit_code_zero_on_all_pass
#    When all 4 stage functions are mocked to return passing values, exit code 0
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_smoke_exit_code_zero_on_all_pass():
    """main() must return exit code 0 when every stage passes."""
    smoke = _load_smoke_live()

    # Patch each individual stage runner to return a PASS result.
    pass_result = smoke.SmokeResult(name="X", status="PASS", elapsed_ms=1, message="ok")

    with (
        patch.object(smoke, "stage_stt", return_value=pass_result),
        patch.object(smoke, "stage_llm", return_value=pass_result),
        patch.object(smoke, "stage_tts", return_value=pass_result),
        patch.object(smoke, "stage_rag", return_value=pass_result),
    ):
        code = smoke.main([""])  # empty args → no skip flags

    assert code == 0, f"Expected exit code 0, got {code}"


# ---------------------------------------------------------------------------
# 4. test_smoke_exit_code_one_on_any_fail
#    If any stage returns FAIL, exit code must be 1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_smoke_exit_code_one_on_any_fail():
    """main() must return exit code 1 when at least one stage fails."""
    smoke = _load_smoke_live()

    pass_result = smoke.SmokeResult(name="X", status="PASS", elapsed_ms=1, message="ok")
    fail_result = smoke.SmokeResult(
        name="LLM", status="FAIL", elapsed_ms=100, message="timeout after 10000ms"
    )

    with (
        patch.object(smoke, "stage_stt", return_value=pass_result),
        patch.object(smoke, "stage_llm", return_value=fail_result),
        patch.object(smoke, "stage_tts", return_value=pass_result),
        patch.object(smoke, "stage_rag", return_value=pass_result),
    ):
        code = smoke.main([""])

    assert code == 1, f"Expected exit code 1, got {code}"


# ---------------------------------------------------------------------------
# 5. test_doctor_live_flag_invokes_smoke
#    --live flag in doctor.py must call subprocess.run with smoke_live.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_doctor_live_flag_invokes_smoke():
    """doctor.main(['--live']) must invoke smoke_live.py via subprocess.run."""
    # Load doctor module
    doctor_path = _PROJECT_ROOT / "scripts" / "doctor.py"
    spec = importlib.util.spec_from_file_location("doctor", doctor_path)
    doctor = importlib.util.module_from_spec(spec)

    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="[PASS] STT 10ms\n"))

    # We need to patch sys.argv and all the I/O before exec_module.
    with (
        patch("subprocess.run", mock_run),
        patch("sys.argv", ["doctor.py", "--live"]),
        patch("builtins.print"),           # suppress output during test
        patch("sounddevice.query_devices", return_value=[{"max_input_channels": 1}]),
    ):
        try:
            spec.loader.exec_module(doctor)
            doctor.main()
        except SystemExit:
            pass  # main() may call sys.exit; that's fine

    # Verify subprocess.run was called with smoke_live.py somewhere in args.
    assert mock_run.called, "subprocess.run was never called"
    call_args = mock_run.call_args
    cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
    cmd_str = " ".join(str(a) for a in cmd)
    assert "smoke_live" in cmd_str, (
        f"Expected 'smoke_live' in subprocess command, got: {cmd_str!r}"
    )
