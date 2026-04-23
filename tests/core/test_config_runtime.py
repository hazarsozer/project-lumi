"""
Tests for src/core/config_runtime.py (Wave S0).

Covers:
- apply() with a hot field → appears in applied_live, not pending_restart
- apply() with a restart-required field → appears in pending_restart, not applied_live
- apply() with an unknown key → appears in errors, config unchanged
- apply() with a value out of range → appears in errors
- apply() is thread-safe: 10 concurrent threads, no exceptions, valid config
- Observer is notified after a hot-field apply
- Observer is NOT notified after a restart-required-only apply
- persist=True calls write_config (mocked)
- current returns the latest config after apply
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import LumiConfig
from src.core.config_runtime import ConfigManager, ConfigObserver, ConfigUpdateResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> LumiConfig:
    return LumiConfig()


@pytest.fixture
def manager(default_config: LumiConfig) -> ConfigManager:
    return ConfigManager(default_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockObserver:
    """Minimal ConfigObserver implementation for test assertions."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_config: LumiConfig | None = None

    def reconfigure(self, new_config: LumiConfig) -> None:
        self.call_count += 1
        self.last_config = new_config


# ---------------------------------------------------------------------------
# Basic apply() behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_hot_field_in_applied_live(manager: ConfigManager) -> None:
    """A hot-reloadable field must appear in applied_live, not pending_restart."""
    result = manager.apply({"audio.sensitivity": 0.5})

    assert "audio.sensitivity" in result.applied_live
    assert "audio.sensitivity" not in result.pending_restart
    assert result.errors == {}


@pytest.mark.unit
def test_apply_restart_required_field_in_pending_restart(
    manager: ConfigManager,
) -> None:
    """A restart-required field must appear in pending_restart, not applied_live."""
    result = manager.apply({"llm.model_path": "models/llm/other.gguf"})

    assert "llm.model_path" in result.pending_restart
    assert "llm.model_path" not in result.applied_live
    assert result.errors == {}


@pytest.mark.unit
def test_apply_top_level_hot_field(manager: ConfigManager) -> None:
    """Top-level hot field (log_level) appears in applied_live."""
    result = manager.apply({"log_level": "DEBUG"})

    assert "log_level" in result.applied_live
    assert result.errors == {}


@pytest.mark.unit
def test_apply_top_level_restart_field(manager: ConfigManager) -> None:
    """Top-level restart field (edition) appears in pending_restart."""
    result = manager.apply({"edition": "pro"})

    assert "edition" in result.pending_restart
    assert result.errors == {}


@pytest.mark.unit
def test_current_reflects_applied_change(manager: ConfigManager) -> None:
    """current property must return the updated config after apply()."""
    original_sensitivity = manager.current.audio.sensitivity
    new_sensitivity = original_sensitivity + 0.1

    manager.apply({"audio.sensitivity": new_sensitivity})

    assert manager.current.audio.sensitivity == pytest.approx(new_sensitivity)


@pytest.mark.unit
def test_current_unchanged_after_restart_field_apply(
    manager: ConfigManager,
) -> None:
    """Restart-required changes still update the in-memory config."""
    manager.apply({"llm.model_path": "/new/path/model.gguf"})
    # The in-memory config IS updated; a restart just re-reads from the
    # file, so the persisted value is what matters at startup.
    assert manager.current.llm.model_path == "/new/path/model.gguf"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_unknown_key_returns_error(manager: ConfigManager) -> None:
    """An unknown key must appear in errors and config must remain unchanged."""
    original = manager.current
    result = manager.apply({"totally.unknown.key": 42})

    assert "totally.unknown.key" in result.errors
    assert result.applied_live == []
    assert result.pending_restart == []
    # Config must not have changed.
    assert manager.current is original


@pytest.mark.unit
def test_apply_out_of_range_value_returns_error(manager: ConfigManager) -> None:
    """A value outside the allowed range must appear in errors."""
    original = manager.current
    result = manager.apply({"audio.sensitivity": 2.0})  # max is 1.0

    assert "audio.sensitivity" in result.errors
    assert "below" not in result.errors.get("audio.sensitivity", "")
    assert "exceeds" in result.errors.get("audio.sensitivity", "")
    assert result.applied_live == []
    assert manager.current is original


@pytest.mark.unit
def test_apply_below_range_value_returns_error(manager: ConfigManager) -> None:
    """A value below the allowed minimum must appear in errors."""
    result = manager.apply({"audio.silence_timeout_s": 0.0})  # min is 0.1

    assert "audio.silence_timeout_s" in result.errors


@pytest.mark.unit
def test_apply_wrong_type_toggle_returns_error(manager: ConfigManager) -> None:
    """A non-boolean value for a toggle field must produce an error."""
    result = manager.apply({"tts.enabled": "yes"})

    assert "tts.enabled" in result.errors


@pytest.mark.unit
def test_apply_invalid_select_option_returns_error(manager: ConfigManager) -> None:
    """A value not in the allowed options for a select field must produce an error."""
    result = manager.apply({"edition": "ultra"})

    assert "edition" in result.errors


@pytest.mark.unit
def test_apply_mixed_valid_invalid_all_rejected(manager: ConfigManager) -> None:
    """If any key has an error, the entire batch must be rejected."""
    original = manager.current
    result = manager.apply(
        {
            "audio.sensitivity": 0.5,  # valid hot field
            "totally.fake.key": "bad",  # invalid
        }
    )

    assert result.errors  # at least one error
    assert result.applied_live == []
    assert result.pending_restart == []
    # Config must not have changed.
    assert manager.current.audio.sensitivity == original.audio.sensitivity


# ---------------------------------------------------------------------------
# Observer notifications
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_observer_notified_after_hot_field_change(manager: ConfigManager) -> None:
    """Observer must be called when a hot-reloadable field changes."""
    obs = _MockObserver()
    manager.register_observer("test_obs", obs)

    manager.apply({"audio.sensitivity": 0.3})

    assert obs.call_count == 1
    assert obs.last_config is not None
    assert obs.last_config.audio.sensitivity == pytest.approx(0.3)


@pytest.mark.unit
def test_observer_not_notified_for_unchanged_hot_field(
    manager: ConfigManager,
) -> None:
    """Observer must NOT be called when the hot field value does not change."""
    obs = _MockObserver()
    manager.register_observer("test_obs", obs)

    # Apply the same value that's already set.
    current_sensitivity = manager.current.audio.sensitivity
    manager.apply({"audio.sensitivity": current_sensitivity})

    assert obs.call_count == 0


@pytest.mark.unit
def test_observer_not_notified_for_restart_required_only(
    manager: ConfigManager,
) -> None:
    """Observer must NOT be called when only restart-required fields change."""
    obs = _MockObserver()
    manager.register_observer("test_obs", obs)

    manager.apply({"llm.model_path": "models/llm/new.gguf"})

    assert obs.call_count == 0


@pytest.mark.unit
def test_multiple_observers_all_notified(manager: ConfigManager) -> None:
    """All registered observers must be notified on a hot change."""
    obs_a = _MockObserver()
    obs_b = _MockObserver()
    manager.register_observer("a", obs_a)
    manager.register_observer("b", obs_b)

    manager.apply({"audio.vad_threshold": 0.6})

    assert obs_a.call_count == 1
    assert obs_b.call_count == 1


# ---------------------------------------------------------------------------
# persist=True
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_persist_true_calls_write_config(manager: ConfigManager) -> None:
    """persist=True must invoke write_config with the updated config."""
    with patch(
        "src.core.config_runtime.write_config", autospec=True
    ) as mock_write:
        manager.apply({"audio.sensitivity": 0.4}, persist=True)

    mock_write.assert_called_once()
    called_config = mock_write.call_args[0][0]
    assert isinstance(called_config, LumiConfig)
    assert called_config.audio.sensitivity == pytest.approx(0.4)


@pytest.mark.unit
def test_persist_false_does_not_call_write_config(manager: ConfigManager) -> None:
    """persist=False must not invoke write_config."""
    with patch(
        "src.core.config_runtime.write_config", autospec=True
    ) as mock_write:
        manager.apply({"audio.sensitivity": 0.4}, persist=False)

    mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_thread_safe_concurrent_hot_fields(
    default_config: LumiConfig,
) -> None:
    """10 threads calling apply() concurrently must not raise and result in a valid config."""
    manager = ConfigManager(default_config)
    errors_seen: list[Exception] = []
    barrier = threading.Barrier(10)

    def worker(sensitivity_value: float) -> None:
        barrier.wait()  # All threads start at the same moment.
        try:
            manager.apply({"audio.sensitivity": sensitivity_value})
        except Exception as exc:
            errors_seen.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i / 10.0,), daemon=True)
        for i in range(1, 11)  # 0.1, 0.2, ..., 1.0
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors_seen == [], f"Thread(s) raised exceptions: {errors_seen}"

    # The final config must be a valid LumiConfig with sensitivity in [0.0, 1.0].
    final = manager.current
    assert isinstance(final, LumiConfig)
    assert 0.0 <= final.audio.sensitivity <= 1.0


# ---------------------------------------------------------------------------
# Multiselect (allowed_tools)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_multiselect_valid(manager: ConfigManager) -> None:
    """A valid multiselect value (list of allowed options) must be accepted."""
    result = manager.apply({"tools.allowed_tools": ["launch_app", "clipboard"]})

    assert "tools.allowed_tools" in result.applied_live
    assert result.errors == {}
    # stored as a tuple in ToolsConfig
    assert manager.current.tools.allowed_tools == ("launch_app", "clipboard")


@pytest.mark.unit
def test_apply_multiselect_invalid_option_returns_error(
    manager: ConfigManager,
) -> None:
    """A multiselect value with an unknown option must produce an error."""
    result = manager.apply({"tools.allowed_tools": ["launch_app", "rm_rf"]})

    assert "tools.allowed_tools" in result.errors


# ---------------------------------------------------------------------------
# ConfigUpdateResult dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_update_result_fields() -> None:
    """ConfigUpdateResult must expose the three required fields."""
    r = ConfigUpdateResult(
        applied_live=["a"],
        pending_restart=["b"],
        errors={"c": "oops"},
    )
    assert r.applied_live == ["a"]
    assert r.pending_restart == ["b"]
    assert r.errors == {"c": "oops"}
