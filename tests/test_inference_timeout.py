"""Tests for the per-turn LLM inference watchdog.

Covers:
- LLMConfig.inference_timeout_s default value
- Watchdog fires and sets cancel_flag when inference thread hangs
- Watchdog does NOT fire when inference completes before deadline
- Watchdog is a no-op when inference_timeout_s == 0.0
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.audio.speaker import SpeakerThread
from src.core.config import LLMConfig, LumiConfig
from src.core.orchestrator import Orchestrator
from src.core.state_machine import LumiState


# ---------------------------------------------------------------------------
# Config field defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_llm_config_inference_timeout_default() -> None:
    """LLMConfig must default inference_timeout_s to 30.0."""
    cfg = LLMConfig()
    assert cfg.inference_timeout_s == 30.0


@pytest.mark.unit
def test_llm_config_inference_timeout_is_configurable() -> None:
    """inference_timeout_s can be overridden at construction time."""
    cfg = LLMConfig(inference_timeout_s=5.0)
    assert cfg.inference_timeout_s == 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orch(timeout_s: float) -> Orchestrator:
    """Build a minimal Orchestrator with the given inference timeout."""
    mock_speaker = MagicMock(spec=SpeakerThread)
    llm_cfg = LLMConfig(inference_timeout_s=timeout_s)
    cfg = LumiConfig(llm=llm_cfg)
    orch = Orchestrator(config=cfg, speaker=mock_speaker)
    # Mark the model as already loaded so the lazy-load path is skipped.
    # ModelLoader.is_loaded checks ``self._model is not None``.
    orch._model_loader._model = MagicMock()
    return orch


def _drive_to_processing(orch: Orchestrator) -> None:
    """Transition the orchestrator state machine into PROCESSING."""
    orch._state_machine.transition_to(LumiState.LISTENING)
    orch._state_machine.transition_to(LumiState.PROCESSING)
    orch._llm_cancel_flag.clear()


def _patch_slow_path(orch: Orchestrator, generate_side_effect: object) -> tuple:
    """Return context managers that force the slow (LLM) path.

    Patches:
    - reflex_router.route        → None  (no shortcut match)
    - reflex_router.route_rag_intent → False
    - reasoning_router.generate  → provided side_effect
    """
    return (
        patch.object(orch._reflex_router, "route", return_value=None),
        patch.object(orch._reflex_router, "route_rag_intent", return_value=False),
        patch.object(
            orch._reasoning_router, "generate", side_effect=generate_side_effect
        ),
    )


# ---------------------------------------------------------------------------
# Watchdog fires when inference hangs
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)  # hard ceiling so CI can't stall
def test_watchdog_sets_cancel_flag_on_timeout() -> None:
    """When the LLM thread hangs past inference_timeout_s the watchdog must
    set the cancel flag."""
    orch = _make_orch(timeout_s=0.2)
    hang_event = threading.Event()

    def _hanging_generate(*_args, **_kwargs):  # type: ignore[override]
        hang_event.wait(timeout=10)
        raise InterruptedError("cancelled by watchdog")

    p_route, p_rag, p_gen = _patch_slow_path(orch, _hanging_generate)
    with p_route, p_rag, p_gen:
        _drive_to_processing(orch)

        threading.Thread(
            target=orch._dispatch_user_turn,
            args=("status of the project", "test"),
            daemon=True,
        ).start()

        # Wait for watchdog to fire (budget = 0.2 s; allow up to 3 s).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if orch._llm_cancel_flag.is_set():
                break
            time.sleep(0.05)

    hang_event.set()  # unblock the hanging thread

    assert orch._llm_cancel_flag.is_set(), "Watchdog should have set the cancel flag"


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_watchdog_restores_idle_state_on_timeout() -> None:
    """After the watchdog fires the state machine must return to IDLE."""
    orch = _make_orch(timeout_s=0.2)
    hang_event = threading.Event()

    def _hanging_generate(*_args, **_kwargs):  # type: ignore[override]
        hang_event.wait(timeout=10)
        raise InterruptedError("cancelled")

    p_route, p_rag, p_gen = _patch_slow_path(orch, _hanging_generate)
    with p_route, p_rag, p_gen:
        _drive_to_processing(orch)

        threading.Thread(
            target=orch._dispatch_user_turn,
            args=("status of the project", "test"),
            daemon=True,
        ).start()

        # Wait for watchdog to drive state back to IDLE.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if orch._state_machine.current_state == LumiState.IDLE:
                break
            time.sleep(0.05)

    hang_event.set()

    assert orch._state_machine.current_state == LumiState.IDLE, (
        "Watchdog should have transitioned state machine back to IDLE"
    )


# ---------------------------------------------------------------------------
# Watchdog does NOT fire when inference completes in time
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_watchdog_cancelled_on_normal_completion() -> None:
    """When inference finishes before the deadline the cancel flag must remain
    unset (the watchdog timer is cancelled in the finally block)."""
    orch = _make_orch(timeout_s=5.0)  # generous timeout

    def _fast_generate(*_args, **_kwargs):  # type: ignore[override]
        return "pong"  # completes immediately

    p_route, p_rag, p_gen = _patch_slow_path(orch, _fast_generate)
    with p_route, p_rag, p_gen, \
         patch.object(orch._memory, "save"), \
         patch.object(orch, "post_event"):
        _drive_to_processing(orch)

        threading.Thread(
            target=orch._dispatch_user_turn,
            args=("status of the project", "test"),
            daemon=True,
        ).start()

        time.sleep(0.3)  # let the thread finish

    # Cancel flag must NOT be set — watchdog was cancelled on normal exit.
    assert not orch._llm_cancel_flag.is_set(), (
        "Cancel flag must not be set when inference finishes before timeout"
    )


# ---------------------------------------------------------------------------
# Zero timeout disables watchdog entirely
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_zero_timeout_disables_watchdog() -> None:
    """Setting inference_timeout_s=0.0 must not start a watchdog timer."""
    orch = _make_orch(timeout_s=0.0)
    hang_event = threading.Event()

    def _hanging_generate(*_args, **_kwargs):  # type: ignore[override]
        # Block briefly — a 0-second watchdog (if created) would fire instantly.
        hang_event.wait(timeout=0.4)
        raise InterruptedError("cancelled")

    p_route, p_rag, p_gen = _patch_slow_path(orch, _hanging_generate)
    with p_route, p_rag, p_gen:
        _drive_to_processing(orch)

        threading.Thread(
            target=orch._dispatch_user_turn,
            args=("status of the project", "test"),
            daemon=True,
        ).start()

        time.sleep(0.5)  # no watchdog should fire in this window

    hang_event.set()

    assert not orch._llm_cancel_flag.is_set(), (
        "With timeout=0.0 the watchdog must not set the cancel flag"
    )
