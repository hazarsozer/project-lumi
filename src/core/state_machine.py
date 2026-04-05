"""
Thread-safe state machine for Project Lumi.

Enforces valid state transitions and notifies registered observers.
This module imports ONLY from stdlib to prevent circular imports --
it is deliberately unaware of the event system.

Usage:
    from src.core.state_machine import StateMachine, LumiState, InvalidTransitionError
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class LumiState(Enum):
    """All possible states for the Lumi voice assistant."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


# Valid transitions: mapping from (current_state, target_state) to True.
_VALID_TRANSITIONS: frozenset[tuple[LumiState, LumiState]] = frozenset(
    {
        (LumiState.IDLE, LumiState.LISTENING),
        (LumiState.LISTENING, LumiState.PROCESSING),
        (LumiState.LISTENING, LumiState.IDLE),
        (LumiState.PROCESSING, LumiState.SPEAKING),
        (LumiState.PROCESSING, LumiState.IDLE),
        (LumiState.SPEAKING, LumiState.IDLE),
    }
)


class StateMachine:
    """Thread-safe finite state machine for Lumi's lifecycle.

    Args:
        initial_state: The starting state. Defaults to IDLE.
    """

    def __init__(self, initial_state: LumiState = LumiState.IDLE) -> None:
        self._state: LumiState = initial_state
        self._lock: threading.Lock = threading.Lock()
        self._observers: list[Callable[[LumiState, LumiState], None]] = []

    @property
    def current_state(self) -> LumiState:
        """Return the current state (thread-safe read)."""
        with self._lock:
            return self._state

    def transition_to(self, target: LumiState) -> None:
        """Attempt a state transition.

        Args:
            target: The desired new state.

        Raises:
            InvalidTransitionError: If the transition is not in the
                valid transitions set.
        """
        with self._lock:
            old = self._state
            if (old, target) not in _VALID_TRANSITIONS:
                raise InvalidTransitionError(
                    f"Invalid transition: {old.value} -> {target.value}"
                )
            self._state = target
            logger.info("State: %s -> %s", old.value, target.value)
            # Copy observer list under lock; call outside would risk deadlock
            observers = list(self._observers)

        # Notify observers outside the lock to prevent deadlocks if an
        # observer tries to read current_state or transition again.
        for callback in observers:
            try:
                callback(old, target)
            except Exception:
                logger.exception("Observer raised an exception during state transition")

    def register_observer(
        self, callback: Callable[[LumiState, LumiState], None]
    ) -> None:
        """Register a callback to be notified on every state transition.

        The callback receives (old_state, new_state) after the transition
        has been committed. Observers are called outside the internal lock
        so they may safely read current_state.

        Args:
            callback: A callable(old_state, new_state).
        """
        with self._lock:
            self._observers.append(callback)
