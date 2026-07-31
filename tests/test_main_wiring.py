"""Wiring tests for src/main.py — the production construction seam.

These assert that ``main()`` actually passes configuration into the objects it
builds.  This is the missing-test class that let two ship-grade bugs through:
R4 (the Orchestrator was built without ``tts=``) and the VAD dead-config bug
(``AudioConfig`` declared ``vad_threshold`` / ``silence_timeout_s`` /
``recording_timeout_s`` but ``main()`` only passed ``sensitivity``).  A wiring
test at "does the production entry point pass config X into component Y?" is the
3-line guard the 2026-05-30 postmortem called for.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_main_wires_vad_config_into_ears() -> None:
    """main() must pass every config.audio VAD knob into Ears (dead-config guard)."""
    from src.core.config import load_config

    base = load_config()
    fake = dataclasses.replace(
        base,
        audio=dataclasses.replace(
            base.audio,
            sensitivity=0.81,
            vad_threshold=0.66,
            silence_timeout_s=2.22,
            recording_timeout_s=7.77,
            no_speech_timeout_s=3.33,
            wake_word_enabled=True,
        ),
    )

    import src.main as main_mod

    with (
        patch("src.main.setup_logging"),
        patch("src.main.load_config", return_value=fake),
        patch("src.main.generate_and_write_token"),
        patch("src.main.run_startup_checks", return_value=[]),
        patch("src.main.Ears") as MockEars,
        patch("src.main.Scribe"),
        patch("src.main.Orchestrator") as MockOrch,
    ):
        MockOrch.return_value.run.return_value = None
        main_mod.main()

    MockEars.assert_called_once()
    kwargs = MockEars.call_args.kwargs
    assert kwargs["sensitivity"] == 0.81
    assert kwargs["vad_threshold"] == 0.66
    assert kwargs["silence_timeout_s"] == 2.22
    assert kwargs["recording_timeout_s"] == 7.77
    assert kwargs["no_speech_timeout_s"] == 3.33
