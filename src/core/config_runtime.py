"""
Runtime config manager for Project Lumi (Wave S0).

``ConfigManager`` is the single point of truth for the running configuration.
It accepts dotted-path changes, validates them, classifies them as live-apply
or restart-required (using ``FIELD_META`` from ``config_schema.py``), notifies
registered observers, and optionally persists the new config to disk.

Thread safety
-------------
All mutations go through a ``threading.RLock``.  Observers are called while
the lock is NOT held so that observer code can safely call back into the
manager without deadlocking.

Frozen-dataclass invariant
--------------------------
``LumiConfig`` and its sub-dataclasses are frozen.  Updates always use
``dataclasses.replace()`` — never in-place mutation.

Observer protocol
-----------------
Only observers are notified for fields that are *hot-reloadable*
(``restart_required=False`` in FIELD_META) and whose value actually changed.
Observers are NOT called when only restart-required fields change, because
those changes only take effect after the process restarts.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Protocol

from src.core.config import (
    AudioConfig,
    IPCConfig,
    LLMConfig,
    LumiConfig,
    PersonaConfig,
    RAGConfig,
    ScribeConfig,
    ToolsConfig,
    TTSConfig,
    VisionConfig,
    _validate_ipc_address,
)
from src.core.config_schema import FIELD_META
from src.core.config_writer import write_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observer protocol
# ---------------------------------------------------------------------------


class ConfigObserver(Protocol):
    """Interface for objects that want live config-change notifications."""

    def reconfigure(self, new_config: LumiConfig) -> None:
        """Called after a hot-reloadable field has been applied.

        Args:
            new_config: The fully updated ``LumiConfig`` instance.
        """
        ...


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ConfigUpdateResult:
    """Summary returned by ``ConfigManager.apply()``."""

    applied_live: list[str]
    """Field keys applied immediately without requiring a restart."""

    pending_restart: list[str]
    """Field keys whose changes will only take effect after a restart."""

    errors: dict[str, str]
    """Field key → human-readable error message for any rejected change."""


# ---------------------------------------------------------------------------
# Section routing
# ---------------------------------------------------------------------------

# Maps section prefix → (sub-config class, attribute name on LumiConfig)
_SECTION_MAP: dict[str, tuple[type, str]] = {
    "audio": (AudioConfig, "audio"),
    "scribe": (ScribeConfig, "scribe"),
    "llm": (LLMConfig, "llm"),
    "tts": (TTSConfig, "tts"),
    "ipc": (IPCConfig, "ipc"),
    "tools": (ToolsConfig, "tools"),
    "vision": (VisionConfig, "vision"),
    "rag": (RAGConfig, "rag"),
    "persona": (PersonaConfig, "persona"),
}

# Top-level scalar fields that live directly on LumiConfig (no sub-section).
_TOP_LEVEL_SCALARS: frozenset[str] = frozenset(["edition", "log_level", "json_logs"])

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _coerce_value(key: str, value: Any, meta: dict[str, Any]) -> tuple[Any, str | None]:
    """Attempt to coerce and range-check ``value`` according to ``meta``.

    Returns:
        ``(coerced_value, None)`` on success.
        ``(None, error_message)`` on failure.
    """
    control = meta.get("control", "")

    # ---- Type coercion ---------------------------------------------------
    if control == "toggle":
        if not isinstance(value, bool):
            return None, (
                f"Field '{key}' expects a boolean (true/false), "
                f"got {type(value).__name__!r}."
            )
    elif control in ("slider", "number"):
        if not isinstance(value, (int, float)):
            return None, (
                f"Field '{key}' expects a numeric value, "
                f"got {type(value).__name__!r}."
            )
        # Reject non-finite floats (NaN, ±Inf) before any range check.
        import math

        if isinstance(value, float) and not math.isfinite(value):
            return None, (f"Field '{key}' value {value!r} is not a finite number.")
        # Hard absolute ceiling: no config value should ever exceed 1 billion.
        # This prevents a compromised client from passing absurdly large integers
        # (e.g. 2**53) that cause downstream consumers to attempt impossible
        # allocations (e.g. audio buffer of 10^15 frames).
        _ABS_MAX = 1_000_000_000
        if value > _ABS_MAX:
            return None, (
                f"Field '{key}' value {value!r} exceeds the absolute "
                f"maximum of {_ABS_MAX!r}."
            )
        # Range check
        if "min" in meta and value < meta["min"]:
            return None, (
                f"Field '{key}' value {value!r} is below the minimum "
                f"{meta['min']!r}."
            )
        if "max" in meta and value > meta["max"]:
            return None, (
                f"Field '{key}' value {value!r} exceeds the maximum "
                f"{meta['max']!r}."
            )
    elif control == "select":
        options = meta.get("options", [])
        if value not in options:
            return None, (
                f"Field '{key}' value {value!r} is not one of the "
                f"allowed options: {options!r}."
            )
    elif control == "multiselect":
        options = meta.get("options", [])
        # Accept list or tuple.
        if not isinstance(value, (list, tuple)):
            return None, (
                f"Field '{key}' expects a list of values, "
                f"got {type(value).__name__!r}."
            )
        invalid = [v for v in value if v not in options]
        if invalid:
            return None, (
                f"Field '{key}' contains unknown option(s): {invalid!r}. "
                f"Allowed: {options!r}."
            )
        # ToolsConfig stores allowed_tools as a tuple.
        value = tuple(value)
    elif control in ("text", "path"):
        if value is not None and not isinstance(value, str):
            return None, (
                f"Field '{key}' expects a string, " f"got {type(value).__name__!r}."
            )

    return value, None


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Thread-safe runtime config manager.

    Usage::

        manager = ConfigManager(load_config())
        manager.register_observer("llm", my_llm_engine)
        result = manager.apply({"audio.sensitivity": 0.6}, persist=False)

    Atomicity guarantee (CR-13)
    ----------------------------
    ``current`` never returns a config snapshot while a notification round is
    in progress.  Concretely: ``_notify_epoch`` is incremented (under
    ``_lock``) right before the notification loop begins; ``_notify_done_epoch``
    is incremented and ``_notify_cond`` is broadcast after the last observer
    returns.  Any reader that finds ``_notify_epoch != _notify_done_epoch``
    waits on ``_notify_cond`` before returning the config.

    This eliminates the window between the config commit (end of Phase 2) and
    the end of the observer notification loop (end of Phase 4) that was
    identified in GitHub issue #14.

    Deadlock freedom
    ----------------
    - ``_lock`` (RLock) is NEVER held while observers are called (snapshot
      under lock, iterate outside).
    - ``_notify_cond`` uses its own internal lock, separate from ``_lock``.
      It is acquired only in ``current`` (readers) and at the very end of
      Phase 4 (after all locks on ``_config`` have been released).
    - Observers that call back into ``ConfigManager`` (e.g. reading
      ``current``) will wait on ``_notify_cond`` until the notification
      round completes — they will NOT deadlock because ``_lock`` is not held
      during that wait, and Phase 4 will eventually release ``_notify_cond``.
    """

    def __init__(self, config: LumiConfig) -> None:
        self._config: LumiConfig = config
        self._lock: threading.RLock = threading.RLock()
        self._observers: dict[str, ConfigObserver] = {}
        # Notification-round tracking for CR-13 atomicity.
        self._notify_epoch: int = 0       # incremented when a round BEGINS
        self._notify_done_epoch: int = 0  # incremented when a round ENDS
        self._notify_cond: threading.Condition = threading.Condition()
        # Public monotonic version counter (increments on each hot-field change).
        self._config_version: int = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def config_version(self) -> int:
        """Monotonic counter that increments each time a hot-reloadable field
        actually changes.  Stable (not incremented) for restart-required-only
        or no-op applies."""
        with self._lock:
            return self._config_version

    @property
    def current(self) -> LumiConfig:
        """Return the current config snapshot.

        Blocks briefly if a notification round is in progress so that no
        caller ever observes a new config version whose observers have not
        yet been notified (CR-13).
        """
        # Fast path: no notification round active — just grab the config.
        with self._lock:
            epoch = self._notify_epoch
            done_epoch = self._notify_done_epoch
            cfg = self._config

        if epoch == done_epoch:
            # No active notification round.
            return cfg

        # Slow path: a notification round is in progress.  Wait until it
        # completes, then re-read the config.
        #
        # Liveness assumption: observers must not block indefinitely inside
        # reconfigure().  The current observers are trivial attribute
        # assignments, so this is safe.  A hung observer (one that blocks
        # rather than raises) would stall every config reader here — there is
        # no timeout-return because returning early would weaken the
        # no-stale-read guarantee (CR-13).  If observers are ever extended to
        # do I/O or lock acquisition, add an explicit timeout + cancellation
        # mechanism in the observer contract rather than weakening this wait.
        with self._notify_cond:
            while self._notify_epoch != self._notify_done_epoch:
                self._notify_cond.wait(timeout=1.0)
        with self._lock:
            return self._config

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------

    def register_observer(self, name: str, observer: ConfigObserver) -> None:
        """Register an observer to receive live config-change notifications.

        Args:
            name:     Unique label for this observer (used in log messages).
            observer: Object implementing the ``ConfigObserver`` protocol.
        """
        with self._lock:
            self._observers[name] = observer
            logger.debug("ConfigManager: registered observer '%s'.", name)

    # ------------------------------------------------------------------
    # Apply changes
    # ------------------------------------------------------------------

    def apply(
        self, changes: dict[str, Any], persist: bool = False
    ) -> ConfigUpdateResult:
        """Validate and apply a batch of dotted-path config changes.

        Args:
            changes: Dict of ``{"dotted.key": value, ...}`` pairs.
            persist: If ``True``, write the new config to ``config.yaml``
                     after applying.

        Returns:
            A ``ConfigUpdateResult`` describing which keys were applied
            live, which require a restart, and which had errors.

        Notes:
            - Unknown keys are rejected with an error.
            - Type or range violations are rejected with an error.
            - If ANY key has an error, NO keys are applied and the full
              ``errors`` dict is returned (all-or-nothing per call).
            - The config object is updated atomically under the RLock.
        """
        applied_live: list[str] = []
        pending_restart: list[str] = []
        errors: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Phase 1: Validate all keys before applying any of them.
        # ------------------------------------------------------------------
        coerced: dict[str, Any] = {}

        for key, raw_value in changes.items():
            # ---- Look up metadata ----------------------------------------
            meta = FIELD_META.get(key)
            if meta is None:
                errors[key] = (
                    f"Unknown config key '{key}'. "
                    "Check FIELD_META in src/core/config_schema.py for "
                    "supported keys."
                )
                continue

            # ---- Validate / coerce value ---------------------------------
            coerced_value, err = _coerce_value(key, raw_value, meta)
            if err is not None:
                errors[key] = err
                continue

            coerced[key] = coerced_value

        # If any errors, return immediately without modifying state.
        if errors:
            logger.warning(
                "ConfigManager.apply(): rejected %d key(s) with errors: %s",
                len(errors),
                list(errors.keys()),
            )
            return ConfigUpdateResult(
                applied_live=[], pending_restart=[], errors=errors
            )

        # ------------------------------------------------------------------
        # Phase 1b: Cross-field IPC address loopback validation.
        #
        # If ipc.address is being changed, enforce that it is loopback (or that
        # ipc.allow_non_loopback is True — either already in the current config
        # or in this same batch of changes).
        # ------------------------------------------------------------------
        if "ipc.address" in coerced:
            effective_allow = coerced.get(
                "ipc.allow_non_loopback",
                self._config.ipc.allow_non_loopback,
            )
            try:
                _validate_ipc_address(coerced["ipc.address"], effective_allow)
            except ValueError as exc:
                return ConfigUpdateResult(
                    applied_live=[],
                    pending_restart=[],
                    errors={"ipc.address": str(exc)},
                )

        # ------------------------------------------------------------------
        # Phase 2: Build the updated config under the lock.
        # ------------------------------------------------------------------
        hot_changed_keys: list[str] = []

        with self._lock:
            old_config = self._config

            # Accumulate per-section field updates.
            # section_updates: section_name → {field_name: new_value}
            section_updates: dict[str, dict[str, Any]] = {}
            top_updates: dict[str, Any] = {}

            for key, new_value in coerced.items():
                meta = FIELD_META[key]
                restart_required: bool = meta["restart_required"]

                if key in _TOP_LEVEL_SCALARS:
                    top_updates[key] = new_value
                    if restart_required:
                        pending_restart.append(key)
                    else:
                        applied_live.append(key)
                        # Check if actually changed.
                        if getattr(old_config, key) != new_value:
                            hot_changed_keys.append(key)
                else:
                    section_name, field_name = key.split(".", 1)
                    section_updates.setdefault(section_name, {})[field_name] = new_value
                    if restart_required:
                        pending_restart.append(key)
                    else:
                        applied_live.append(key)
                        # Check if actually changed.
                        old_sub = getattr(old_config, section_name)
                        if getattr(old_sub, field_name) != new_value:
                            hot_changed_keys.append(key)

            # Build new sub-config instances using dataclasses.replace().
            new_section_kwargs: dict[str, Any] = {}
            for section_name, field_updates in section_updates.items():
                old_sub = getattr(old_config, section_name)
                new_sub = dataclasses.replace(old_sub, **field_updates)
                new_section_kwargs[section_name] = new_sub

            # Build the new top-level LumiConfig.
            new_config = dataclasses.replace(
                old_config, **top_updates, **new_section_kwargs
            )
            self._config = new_config

            # If any hot-reloadable field actually changed, bump the version
            # and open a notification epoch BEFORE releasing the lock.  This
            # ensures that any reader acquiring the lock after this point will
            # see the new config AND the in-progress epoch, causing them to
            # wait for the notification round to complete (CR-13).
            #
            # We capture the epoch value opened for THIS round so that Phase 4
            # can close it precisely — even if a concurrent apply() call
            # increments _notify_epoch further while observers are running, we
            # only close our own round (setting _notify_done_epoch to our
            # epoch, not bumping blindly with +=1 which could overshoot).
            if hot_changed_keys:
                self._config_version += 1
                self._notify_epoch += 1
                opened_epoch = self._notify_epoch

        # ------------------------------------------------------------------
        # Phase 3: Persist (outside the lock — IO should not block readers).
        # ------------------------------------------------------------------
        if persist:
            try:
                write_config(new_config)
                logger.info("ConfigManager: config persisted to config.yaml.")
            except Exception as exc:
                logger.error("ConfigManager: failed to persist config: %s", exc)

        # ------------------------------------------------------------------
        # Phase 4: Notify observers for hot-reloadable changes (outside lock).
        # ------------------------------------------------------------------
        if hot_changed_keys:
            observers_snapshot: dict[str, ConfigObserver]
            with self._lock:
                observers_snapshot = dict(self._observers)

            # The try/finally guarantees that _notify_done_epoch is ALWAYS
            # advanced to match our opened_epoch and _notify_cond is ALWAYS
            # broadcast — even if a BaseException (KeyboardInterrupt, etc.)
            # fires during observer iteration, or if notify_all itself raises.
            # Without this guard any such failure leaves _notify_done_epoch
            # behind _notify_epoch permanently, causing every future
            # manager.current slow-path caller to spin-wait forever.
            #
            # The done-epoch is SET to opened_epoch (not +=1) so that this
            # round closes exactly its own slot even when a concurrent apply()
            # has already bumped _notify_epoch further.
            try:
                for obs_name, observer in observers_snapshot.items():
                    try:
                        observer.reconfigure(new_config)
                        logger.debug(
                            "ConfigManager: notified observer '%s' for keys %s.",
                            obs_name,
                            hot_changed_keys,
                        )
                    except Exception as exc:
                        logger.error(
                            "ConfigManager: observer '%s' raised during "
                            "reconfigure: %s",
                            obs_name,
                            exc,
                        )
            finally:
                # Close the notification epoch: mark the round complete and
                # wake any readers that were waiting in current().
                with self._notify_cond:
                    self._notify_done_epoch = opened_epoch
                    self._notify_cond.notify_all()

        logger.debug(
            "ConfigManager.apply(): live=%s restart=%s errors=%s",
            applied_live,
            pending_restart,
            list(errors.keys()),
        )
        return ConfigUpdateResult(
            applied_live=applied_live,
            pending_restart=pending_restart,
            errors=errors,
        )
