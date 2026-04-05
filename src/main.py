"""
Entry point for Project Lumi.

Bootstraps the application in four steps:
1. setup_logging()       — configure the root logger (level and format from config)
2. load_config()         — load config.yaml into a frozen LumiConfig instance
3. run_startup_checks()  — validate model paths, openwakeword version, microphone
4. Orchestrator.run()    — start the event loop; blocks until ShutdownEvent

SIGINT and SIGTERM both post ShutdownEvent to the Orchestrator for graceful exit.
"""

import signal

from src.core.config import load_config
from src.core.events import ShutdownEvent
from src.core.logging_config import setup_logging
from src.core.orchestrator import Orchestrator
from src.core.startup_check import run_startup_checks


def main() -> None:
    setup_logging()
    config = load_config()
    run_startup_checks(config)
    orchestrator = Orchestrator(config)
    signal.signal(signal.SIGINT, lambda s, f: orchestrator.post_event(ShutdownEvent()))
    signal.signal(signal.SIGTERM, lambda s, f: orchestrator.post_event(ShutdownEvent()))
    orchestrator.run()


if __name__ == "__main__":
    main()
