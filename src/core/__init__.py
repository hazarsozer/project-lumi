"""
src.core — Project Lumi infrastructure layer.

Public API exposed by this package:

    from src.core.config import load_config, detect_edition
    from src.core.config import (
        LumiConfig, AudioConfig, ScribeConfig, LLMConfig, IPCConfig
    )
    from src.core.logging_config import setup_logging
    from src.core.startup_check import run_startup_checks
    from src.core.config_runtime import ConfigManager, ConfigUpdateResult, ConfigObserver
    from src.core.config_schema import FIELD_META
    from src.core.config_writer import write_config, ConfigWriteError

Downstream agents (event-architect, llm-engineer, ipc-engineer,
test-architect) should import directly from the sub-modules above rather
than from this package root to keep import paths explicit and avoid
accidental re-exports.
"""
