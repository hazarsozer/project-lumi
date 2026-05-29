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
# Stale-read window (CR-13 atomicity)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_stale_read_during_concurrent_reload(default_config: LumiConfig) -> None:
    """A reader must never see the new config while observers have not yet completed.

    This test is DETERMINISTIC by construction using a gating observer:

    1. A ``_GatingObserver`` is the only registered observer.  Its
       ``reconfigure()`` method sets ``observer_entered`` then blocks on
       ``observer_gate`` (held by the test).  This keeps the notification
       round open for an arbitrarily long time.

    2. ``apply()`` is run in a background thread.  The background thread
       will commit the new config (Phase 2) and then call the observer
       (Phase 4), blocking inside the observer's ``reconfigure()``.

    3. The main test thread waits until ``observer_entered`` is set,
       confirming that the notification round is in progress (Phase 4 started
       but not finished).  It then calls ``manager.current``.

    4. Pre-fix: ``current()`` returns immediately with the new config value
       (the commit happened before Phase 4); ``saw_new_value`` is True.

    5. Post-fix: ``current()`` blocks until the notification round completes,
       so it only returns after the gate is released; the main thread releases
       the gate AFTER sampling, so ``current()`` would have to wait → the main
       thread would not reach the ``saw_new_value`` assignment until after the
       notification is done.

    The key invariant asserted: if the reader saw the NEW value AND the
    observer was still inside ``reconfigure()`` at the time of the read,
    that's a stale-read violation.
    """
    import time

    NEW_SENSITIVITY = 0.777

    manager = ConfigManager(default_config)
    assert manager.current.audio.sensitivity != pytest.approx(NEW_SENSITIVITY)

    # observer_entered: set as soon as reconfigure() is called.
    observer_entered = threading.Event()
    # observer_gate: controls when the observer is allowed to finish.
    observer_gate = threading.Event()
    # observer_done: set when reconfigure() returns.
    observer_done = threading.Event()

    class _GatingObserver:
        """Observer that signals entry, then blocks until released."""

        def reconfigure(self, new_config: LumiConfig) -> None:
            observer_entered.set()
            observer_gate.wait(timeout=10.0)
            observer_done.set()

    manager.register_observer("gate", _GatingObserver())

    # Run apply() in a background thread.
    apply_thread = threading.Thread(
        target=lambda: manager.apply({"audio.sensitivity": NEW_SENSITIVITY}),
        daemon=True,
    )
    apply_thread.start()

    # Wait until the notification round has started (observer entered reconfigure).
    assert observer_entered.wait(timeout=5.0), (
        "Observer never entered reconfigure() — apply_thread may have stalled."
    )

    # At this moment: Phase 4 is in progress (observer is blocked inside
    # reconfigure()).  The new config was committed in Phase 2 (before Phase 4).
    #
    # Pre-fix: current() returns new_config immediately → saw_new_value = True.
    # Post-fix: current() blocks until notification round completes.
    #           We have NOT released observer_gate yet, so current() must block
    #           here waiting.  Release the gate AFTER the sampling to let
    #           apply_thread finish; then current() will return.

    # Use a reader thread so we can enforce a timeout.
    reader_result: list[LumiConfig] = []

    def _reader() -> None:
        reader_result.append(manager.current)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # Give the reader a moment to either return (pre-fix) or block (post-fix).
    reader_thread.join(timeout=0.2)

    if reader_thread.is_alive():
        # Post-fix: current() is blocking — the reader hasn't returned yet.
        # This is the correct behaviour. Release the gate and let everything finish.
        observer_gate.set()
        reader_thread.join(timeout=5.0)
        apply_thread.join(timeout=5.0)
        observer_done.wait(timeout=5.0)
        # The reader must have received the new config (it eventually unblocked).
        assert len(reader_result) == 1
        assert reader_result[0].audio.sensitivity == pytest.approx(NEW_SENSITIVITY)
        # Test passes — current() blocked during the notification round.
        return

    # Pre-fix path: current() returned immediately (reader_thread finished within 0.2s).
    observer_gate.set()
    apply_thread.join(timeout=5.0)
    observer_done.wait(timeout=5.0)

    assert len(reader_result) == 1
    saw_new_value = reader_result[0].audio.sensitivity == pytest.approx(NEW_SENSITIVITY)

    assert not saw_new_value, (
        "Stale-read window detected: manager.current returned the NEW config "
        "value while the observer notification was still in progress. "
        "(GitHub issue #14: current() must not return until the notify round "
        "is complete.)"
    )


@pytest.mark.unit
def test_config_version_increments_on_hot_change(manager: ConfigManager) -> None:
    """config_version must increment each time a hot-reloadable field actually changes."""
    v0 = manager.config_version
    manager.apply({"audio.sensitivity": 0.3})
    v1 = manager.config_version
    assert v1 == v0 + 1, "version should increment after a hot change"


@pytest.mark.unit
def test_config_version_does_not_increment_on_no_change(
    manager: ConfigManager,
) -> None:
    """config_version must NOT increment when the applied value is unchanged."""
    manager.apply({"audio.sensitivity": 0.3})
    v_after_first = manager.config_version
    # Apply the same value again.
    manager.apply({"audio.sensitivity": 0.3})
    assert manager.config_version == v_after_first


@pytest.mark.unit
def test_config_version_does_not_increment_on_restart_only_change(
    manager: ConfigManager,
) -> None:
    """config_version must NOT increment when only restart-required fields change
    (those changes don't trigger observers so there's no notify window to guard)."""
    v0 = manager.config_version
    manager.apply({"llm.model_path": "models/llm/new.gguf"})
    assert manager.config_version == v0


# ---------------------------------------------------------------------------
# finally-guard: observer exception must NOT wedge config readers (CR-13 fix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_observer_exception_does_not_wedge_readers(
    default_config: LumiConfig,
) -> None:
    """Readers must not be permanently blocked when an observer raises Exception.

    This tests the Phase 4 try/finally guard introduced to fix the HIGH bug in
    CR-13: if the observer loop exits abnormally (via Exception), the
    finally-block must still advance _notify_done_epoch to match
    _notify_epoch and broadcast _notify_cond so that any concurrent
    manager.current call is released promptly.

    Strategy
    --------
    1. Register an observer that always raises ``RuntimeError``.
    2. Register a gating observer FIRST so we can confirm that Phase 4 has
       started (and an exception has propagated) before we call current().
       Actually simpler: call apply() synchronously — the per-observer
       try/except catches the RuntimeError, so apply() finishes normally.
       After apply() returns, _notify_epoch and _notify_done_epoch must be
       equal (round closed) and current() must return promptly (no wedge).

    Additionally verifies: a concurrent reader started WHILE the observer is
    running (via a gating observer followed by a raising observer) is still
    released after the raising observer, not permanently stalled.

    What is tested
    --------------
    - After apply() with a raising observer, ``_notify_done_epoch == _notify_epoch``.
    - ``manager.current`` returns (not blocked) and holds the updated value.
    - A concurrent reader thread spawned while Phase 4 is in progress is
      also unblocked (the gating observer releases before the raising one
      is called, but the finally fires unconditionally regardless of which
      observer raises and from which position in the loop).
    """
    import time

    NEW_SENSITIVITY = 0.444
    manager = ConfigManager(default_config)

    # Observer that raises unconditionally.
    class _RaisingObserver:
        def reconfigure(self, new_config: LumiConfig) -> None:
            raise RuntimeError("Simulated observer failure")

    # Gating observer: blocks until released, THEN the raising observer fires.
    observer_entered = threading.Event()
    observer_gate = threading.Event()

    class _GatingObserver:
        def reconfigure(self, new_config: LumiConfig) -> None:
            observer_entered.set()
            observer_gate.wait(timeout=10.0)

    # Register gating first so it runs first; raising runs second.
    manager.register_observer("gate", _GatingObserver())
    manager.register_observer("raise", _RaisingObserver())

    apply_done = threading.Event()

    def _apply() -> None:
        manager.apply({"audio.sensitivity": NEW_SENSITIVITY})
        apply_done.set()

    apply_thread = threading.Thread(target=_apply, daemon=True)
    apply_thread.start()

    # Wait until the gating observer has started (Phase 4 is in progress).
    assert observer_entered.wait(timeout=5.0), "Gating observer never entered"

    # Spawn a concurrent reader while Phase 4 is in progress.
    reader_result: list[LumiConfig] = []

    def _reader() -> None:
        reader_result.append(manager.current)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # Give the reader a moment — it should be blocking (notification round open).
    reader_thread.join(timeout=0.1)
    assert reader_thread.is_alive(), (
        "Reader returned before notification round closed — stale-read violation"
    )

    # Release the gating observer → raising observer fires → finally closes epoch.
    observer_gate.set()

    # Both threads must complete promptly.
    apply_thread.join(timeout=5.0)
    reader_thread.join(timeout=5.0)
    assert not apply_thread.is_alive(), "apply() thread did not finish"
    assert not reader_thread.is_alive(), "reader thread is still blocked (wedged)"

    # Epoch must be closed: _notify_done_epoch must equal _notify_epoch.
    assert manager._notify_done_epoch == manager._notify_epoch, (
        f"Epoch not closed: _notify_epoch={manager._notify_epoch}, "
        f"_notify_done_epoch={manager._notify_done_epoch}"
    )

    # Reader must have received the updated config.
    assert len(reader_result) == 1
    assert reader_result[0].audio.sensitivity == pytest.approx(NEW_SENSITIVITY)


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
