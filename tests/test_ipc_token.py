"""
Tests for src.core.ipc_token — IPC bearer-token generation and verification.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from src.core.ipc_token import generate_and_write_token, read_token, verify_token


# ---------------------------------------------------------------------------
# generate_and_write_token
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_creates_file(tmp_path: Path) -> None:
    """generate_and_write_token must create the token file."""
    token_path = tmp_path / "ipc_token"
    generate_and_write_token(token_path)
    assert token_path.exists()


@pytest.mark.unit
def test_generate_returns_hex_string(tmp_path: Path) -> None:
    """Returned token must be a 64-character lowercase hex string."""
    token_path = tmp_path / "ipc_token"
    token = generate_and_write_token(token_path)
    assert len(token) == 64
    assert all(c in "0123456789abcdef" for c in token)


@pytest.mark.unit
def test_generate_file_content_matches_return(tmp_path: Path) -> None:
    """File contents must match the returned token (stripped of newline)."""
    token_path = tmp_path / "ipc_token"
    token = generate_and_write_token(token_path)
    assert token_path.read_text(encoding="utf-8").strip() == token


@pytest.mark.unit
def test_generate_file_permissions(tmp_path: Path) -> None:
    """Token file must be chmod 0600 (owner read-write only)."""
    token_path = tmp_path / "ipc_token"
    generate_and_write_token(token_path)
    mode = token_path.stat().st_mode & 0o777
    assert mode == 0o600


@pytest.mark.unit
def test_generate_creates_parent_dir(tmp_path: Path) -> None:
    """generate_and_write_token must create missing parent directories."""
    token_path = tmp_path / "nested" / "dir" / "ipc_token"
    generate_and_write_token(token_path)
    assert token_path.exists()


@pytest.mark.unit
def test_generate_produces_unique_tokens(tmp_path: Path) -> None:
    """Two consecutive calls must produce different tokens."""
    t1 = generate_and_write_token(tmp_path / "t1")
    t2 = generate_and_write_token(tmp_path / "t2")
    assert t1 != t2


# ---------------------------------------------------------------------------
# read_token
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_token_returns_content(tmp_path: Path) -> None:
    """read_token must return the token string from the file."""
    token_path = tmp_path / "ipc_token"
    token = generate_and_write_token(token_path)
    assert read_token(token_path) == token


@pytest.mark.unit
def test_read_token_missing_file_returns_none(tmp_path: Path) -> None:
    """read_token must return None when the file does not exist."""
    assert read_token(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_token_correct(tmp_path: Path) -> None:
    """verify_token must return True when tokens match."""
    token = generate_and_write_token(tmp_path / "ipc_token")
    assert verify_token(token, token) is True


@pytest.mark.unit
def test_verify_token_wrong(tmp_path: Path) -> None:
    """verify_token must return False when tokens differ."""
    token = generate_and_write_token(tmp_path / "ipc_token")
    assert verify_token("wrong_token", token) is False


@pytest.mark.unit
def test_verify_token_empty_provided() -> None:
    """verify_token must return False when provided token is empty."""
    assert verify_token("", "someexpectedtoken") is False
