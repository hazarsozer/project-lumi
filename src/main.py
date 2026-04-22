"""
Entry point for Project Lumi.

Bootstraps the application in five steps:
1. setup_logging()       — configure the root logger (level and format from config)
2. load_config()         — load config.yaml into a frozen LumiConfig instance
3. run_startup_checks()  — validate model paths, openwakeword version, microphone
4. Construct Ears + Scribe (audio-in pipeline)
5. Orchestrator.run()    — start the event loop; blocks until ShutdownEvent

SIGINT and SIGTERM both post ShutdownEvent to the Orchestrator for graceful exit.
The Orchestrator calls ears.start() on entry to run() and ears.stop() on ShutdownEvent.

--setup flag:
    Pass ``--setup`` (optionally with ``--yes``) to run the interactive first-run
    setup wizard before any model is loaded.  The wizard is imported lazily so
    this works even when the venv is only partially populated.
"""

# ---------------------------------------------------------------------------
# --setup early exit — must appear before any heavy imports so the wizard
# works even when optional extras (llama-cpp-python, kokoro-onnx, …) are
# not yet installed.
# ---------------------------------------------------------------------------
import sys as _sys

if "--setup" in _sys.argv:
    from scripts.setup_wizard import run_setup as _run_setup

    _yes = "--yes" in _sys.argv or "-y" in _sys.argv
    _sys.exit(_run_setup(yes=_yes))

import logging
import signal

from src.audio.ears import Ears
from src.audio.scribe import Scribe
from src.core.config import load_config
from src.core.events import ShutdownEvent
from src.core.logging_config import setup_logging
from src.core.orchestrator import Orchestrator
from src.core.startup_check import run_startup_checks

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    config = load_config()
    run_startup_checks(config)

    # Construct audio-in pipeline components.
    # Ears reads the wake-word model path and sensitivity from AudioConfig.
    ears = Ears(
        sensitivity=config.audio.sensitivity,
        model_paths=[config.audio.wake_word_model_path],
    )

    # Scribe wraps faster-whisper for speech-to-text.
    scribe = Scribe(
        model_size=config.scribe.model_size,
        initial_prompt=config.scribe.initial_prompt or "Lumi, Firefox, browser, desktop assistant.",
    )

    orchestrator = Orchestrator(config, ears=ears, scribe=scribe)
    signal.signal(signal.SIGINT, lambda s, f: orchestrator.post_event(ShutdownEvent()))
    signal.signal(signal.SIGTERM, lambda s, f: orchestrator.post_event(ShutdownEvent()))
    orchestrator.run()


if __name__ == "__main__":
    main()
