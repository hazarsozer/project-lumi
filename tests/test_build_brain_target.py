"""
Tests for the host-triple detection logic in scripts/build_brain.sh.

Strategy: invoke the script's ``--print-target`` mode in a subprocess,
faking ``uname`` via a tiny wrapper script placed earlier on PATH so we
can probe every supported mapping without touching real hardware.
The tests do NOT run a real PyInstaller / Tauri build.
"""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "scripts" / "build_brain.sh"


def _fake_uname_env(tmp_path: Path, uname_s: str, uname_m: str) -> dict[str, str]:
    """
    Return a modified env dict where a tiny wrapper script shadows ``uname``
    on PATH so that ``uname -s`` / ``uname -m`` return the supplied values.
    """
    wrapper = tmp_path / "uname"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [ "${{1:-}}" = "-s" ]; then
                echo "{uname_s}"
            elif [ "${{1:-}}" = "-m" ]; then
                echo "{uname_m}"
            else
                /usr/bin/uname "$@"
            fi
            """
        )
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    # Strip any override so detection runs
    env.pop("LUMI_TARGET_TRIPLE", None)
    return env


def _run_print_target(env: dict[str, str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        ["bash", str(SCRIPT), "--print-target"],
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Mapping tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uname_s, uname_m, expected_triple",
    [
        ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
        ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
        ("Linux", "arm64", "aarch64-unknown-linux-gnu"),
        ("Darwin", "arm64", "aarch64-apple-darwin"),
        ("Darwin", "aarch64", "aarch64-apple-darwin"),
        ("Darwin", "x86_64", "x86_64-apple-darwin"),
    ],
)
def test_detection_mapping(
    tmp_path: Path, uname_s: str, uname_m: str, expected_triple: str
) -> None:
    env = _fake_uname_env(tmp_path, uname_s, uname_m)
    result = _run_print_target(env)
    assert result.returncode == 0, (
        f"Expected exit 0 for {uname_s}/{uname_m}, got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    # Last line of stdout is the raw triple (after the "[build_brain]" info line)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    assert lines[-1] == expected_triple, (
        f"Expected '{expected_triple}', got '{lines[-1]}'\nfull stdout:\n{result.stdout}"
    )


@pytest.mark.parametrize(
    "uname_s, uname_m",
    [
        ("Linux", "riscv64"),
        ("Darwin", "i386"),
        ("Windows_NT", "x86_64"),
    ],
)
def test_unsupported_host_exits_nonzero(
    tmp_path: Path, uname_s: str, uname_m: str
) -> None:
    env = _fake_uname_env(tmp_path, uname_s, uname_m)
    result = _run_print_target(env)
    assert result.returncode != 0, (
        f"Expected non-zero exit for unsupported {uname_s}/{uname_m}, got 0"
    )
    assert "ERROR" in result.stderr, (
        f"Expected ERROR message in stderr, got:\n{result.stderr}"
    )


def test_env_override_bypasses_detection(tmp_path: Path) -> None:
    """LUMI_TARGET_TRIPLE override must be honoured verbatim."""
    override = "aarch64-apple-darwin"
    env = os.environ.copy()
    env["LUMI_TARGET_TRIPLE"] = override
    result = _run_print_target(env)
    assert result.returncode == 0
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    assert lines[-1] == override, (
        f"Expected override triple '{override}', got '{lines[-1]}'\n"
        f"full stdout:\n{result.stdout}"
    )


def test_this_host_resolves_to_linux_x86_64() -> None:
    """On the CI / dev box (Linux x86_64) the real detection must give the expected triple."""
    env = os.environ.copy()
    env.pop("LUMI_TARGET_TRIPLE", None)
    result = _run_print_target(env)
    # Only assert if we actually are on Linux x86_64; skip otherwise.
    import platform

    machine = platform.machine()
    system = platform.system()
    if system == "Linux" and machine == "x86_64":
        assert result.returncode == 0
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        assert lines[-1] == "x86_64-unknown-linux-gnu"
    else:
        pytest.skip(f"Host is {system}/{machine}, not Linux/x86_64 — skip native assertion")
