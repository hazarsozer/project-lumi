"""
Tests for the ConfigManager wiring in src/core/orchestrator.py (Wave S2).

Covers:
- Orchestrator.__init__ creates _config_manager (ConfigManager instance)
- _config_manager.current returns the config passed to Orchestrator
- _handle_config_schema_request: no-op when _zmq_server is None
- _handle_config_schema_request: calls send_config_schema with FIELD_META and
  a flat dotted-path current_values dict containing expected keys
- _handle_config_update: hot field → applied_live, send_config_update_result called
- _handle_config_update: restart-required field → pending_restart
- _handle_config_update: unknown field → errors dict populated
- _handle_config_update: persist=True updates _config_manager.current
- _handle_config_update: no crash when _zmq_server is None
- _flatten_config helper: returns all expected dotted-path keys
- _flatten_config: tools.allowed_tools is a list (not a tuple)
- _flatten_config: llm.kv_cache_quant is None (the default)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.audio.speaker import SpeakerThread
from src.core.config import LumiConfig
from src.core.config_runtime import ConfigManager
from src.core.config_schema import FIELD_META
from src.core.events import ConfigSchemaRequestEvent, ConfigUpdateEvent
from src.core.orchestrator import Orchestrator, _flatten_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(zmq_server: object = None) -> Orchestrator:
    """Construct an Orchestrator using mock-speaker and optional mock ZMQ server.

    The SpeakerThread is mocked to avoid touching real audio devices.
    ``_speaker.stop()`` is called by the caller when needed to clean up
    the mock (the real thread is never started for a MagicMock).
    """
    mock_speaker = MagicMock(spec=SpeakerThread)
    return Orchestrator(
        config=LumiConfig(),
        speaker=mock_speaker,
        tts=None,
        zmq_server=zmq_server,
        ears=None,
        scribe=None,
    )


# ---------------------------------------------------------------------------
# ConfigManager instantiation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orchestrator_creates_config_manager() -> None:
    """Orchestrator.__init__ must create a _config_manager attribute."""
    orch = _make_orchestrator()
    assert hasattr(orch, "_config_manager")
    assert isinstance(orch._config_manager, ConfigManager)


@pytest.mark.unit
def test_config_manager_current_matches_passed_config() -> None:
    """_config_manager.current should return the config given to Orchestrator."""
    config = LumiConfig()
    mock_speaker = MagicMock(spec=SpeakerThread)
    orch = Orchestrator(
        config=config,
        speaker=mock_speaker,
        tts=None,
        zmq_server=None,
        ears=None,
        scribe=None,
    )
    assert orch._config_manager.current is config


# ---------------------------------------------------------------------------
# _handle_config_schema_request
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schema_request_no_crash_when_zmq_server_none() -> None:
    """_handle_config_schema_request must return silently when _zmq_server is None."""
    orch = _make_orchestrator(zmq_server=None)
    # Must not raise.
    orch._handle_config_schema_request(ConfigSchemaRequestEvent())


@pytest.mark.unit
def test_schema_request_calls_send_config_schema() -> None:
    """When _zmq_server is present, send_config_schema must be called once."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    orch._handle_config_schema_request(ConfigSchemaRequestEvent())

    mock_zmq.send_config_schema.assert_called_once()


@pytest.mark.unit
def test_schema_request_passes_field_meta_as_first_arg() -> None:
    """send_config_schema must receive FIELD_META as its first positional arg."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    orch._handle_config_schema_request(ConfigSchemaRequestEvent())

    call_args = mock_zmq.send_config_schema.call_args
    # First positional argument
    assert call_args[0][0] is FIELD_META


@pytest.mark.unit
def test_schema_request_current_values_contains_top_level_keys() -> None:
    """current_values dict must include top-level scalar keys."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    orch._handle_config_schema_request(ConfigSchemaRequestEvent())

    call_args = mock_zmq.send_config_schema.call_args
    current_values: dict = call_args[0][1]

    assert "edition" in current_values
    assert "log_level" in current_values
    assert "json_logs" in current_values


@pytest.mark.unit
def test_schema_request_current_values_contains_audio_sensitivity() -> None:
    """current_values dict must include 'audio.sensitivity'."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    orch._handle_config_schema_request(ConfigSchemaRequestEvent())

    call_args = mock_zmq.send_config_schema.call_args
    current_values: dict = call_args[0][1]

    assert "audio.sensitivity" in current_values
    assert current_values["audio.sensitivity"] == LumiConfig().audio.sensitivity


@pytest.mark.unit
def test_schema_request_allowed_tools_is_list() -> None:
    """tools.allowed_tools in current_values must be a list, not a tuple."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    orch._handle_config_schema_request(ConfigSchemaRequestEvent())

    call_args = mock_zmq.send_config_schema.call_args
    current_values: dict = call_args[0][1]

    assert "tools.allowed_tools" in current_values
    assert isinstance(current_values["tools.allowed_tools"], list)


# ---------------------------------------------------------------------------
# _handle_config_update
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_update_hot_field_calls_send_result_applied_live() -> None:
    """Hot field change must result in applied_live=[key], pending_restart=[], errors={}."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    event = ConfigUpdateEvent(changes={"audio.sensitivity": 0.6}, persist=False)
    orch._handle_config_update(event)

    mock_zmq.send_config_update_result.assert_called_once_with(
        applied_live=["audio.sensitivity"],
        pending_restart=[],
        errors={},
    )


@pytest.mark.unit
def test_config_update_restart_required_field_calls_send_result_pending_restart() -> None:
    """Restart-required field must appear in pending_restart, not applied_live."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    event = ConfigUpdateEvent(
        changes={"llm.model_path": "models/new.gguf"}, persist=False
    )
    orch._handle_config_update(event)

    mock_zmq.send_config_update_result.assert_called_once_with(
        applied_live=[],
        pending_restart=["llm.model_path"],
        errors={},
    )


@pytest.mark.unit
def test_config_update_unknown_field_calls_send_result_with_error() -> None:
    """Unknown field must appear in errors dict."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    event = ConfigUpdateEvent(changes={"nonexistent.field": 42}, persist=False)
    orch._handle_config_update(event)

    mock_zmq.send_config_update_result.assert_called_once()
    call_kwargs = mock_zmq.send_config_update_result.call_args[1]
    assert "nonexistent.field" in call_kwargs["errors"]
    assert call_kwargs["applied_live"] == []
    assert call_kwargs["pending_restart"] == []


@pytest.mark.unit
def test_config_update_persist_true_updates_config_manager() -> None:
    """With persist=True, _config_manager.current must reflect the applied change."""
    mock_zmq = MagicMock()
    orch = _make_orchestrator(zmq_server=mock_zmq)

    original_sensitivity = orch._config_manager.current.audio.sensitivity
    new_sensitivity = round(original_sensitivity + 0.1, 2)
    # Clamp to [0.0, 1.0] to avoid range-check rejection.
    new_sensitivity = min(new_sensitivity, 1.0)

    # Patch write_config so we don't touch the filesystem.
    with patch("src.core.config_runtime.write_config") as mock_write:
        event = ConfigUpdateEvent(
            changes={"audio.sensitivity": new_sensitivity}, persist=True
        )
        orch._handle_config_update(event)
        mock_write.assert_called_once()

    assert orch._config_manager.current.audio.sensitivity == new_sensitivity


@pytest.mark.unit
def test_config_update_no_crash_when_zmq_server_none() -> None:
    """_handle_config_update must apply config without crashing when _zmq_server is None."""
    orch = _make_orchestrator(zmq_server=None)

    original_sensitivity = orch._config_manager.current.audio.sensitivity
    new_sensitivity = min(round(original_sensitivity + 0.1, 2), 1.0)

    event = ConfigUpdateEvent(
        changes={"audio.sensitivity": new_sensitivity}, persist=False
    )
    # Must not raise.
    orch._handle_config_update(event)

    # Config update should still have been applied.
    assert orch._config_manager.current.audio.sensitivity == new_sensitivity


# ---------------------------------------------------------------------------
# _flatten_config helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flatten_config_contains_all_top_level_keys() -> None:
    """_flatten_config must include edition, log_level, and json_logs."""
    result = _flatten_config(LumiConfig())
    assert "edition" in result
    assert "log_level" in result
    assert "json_logs" in result


@pytest.mark.unit
def test_flatten_config_contains_expected_section_keys() -> None:
    """_flatten_config must include keys for all major config sections."""
    result = _flatten_config(LumiConfig())

    expected_keys = [
        "audio.sensitivity",
        "audio.sample_rate",
        "audio.chunk_size",
        "audio.vad_threshold",
        "audio.silence_timeout_s",
        "audio.recording_timeout_s",
        "audio.wake_word_model_path",
        "scribe.model_size",
        "scribe.beam_size",
        "scribe.compute_type",
        "scribe.model_path",
        "scribe.initial_prompt",
        "llm.model_path",
        "llm.n_gpu_layers",
        "llm.context_length",
        "llm.max_tokens",
        "llm.temperature",
        "llm.vram_budget_gb",
        "llm.kv_cache_quant",
        "llm.memory_dir",
        "tts.enabled",
        "tts.voice",
        "tts.model_path",
        "tts.voices_path",
        "ipc.enabled",
        "ipc.address",
        "ipc.port",
        "tools.enabled",
        "tools.allowed_tools",
        "tools.execution_timeout_s",
        "vision.enabled",
        "vision.model_path",
        "vision.capture_method",
        "vision.max_resolution",
        "rag.enabled",
        "rag.db_path",
        "rag.embedding_model",
        "rag.chunk_size",
        "rag.chunk_overlap",
        "rag.retrieval_top_k",
        "rag.context_char_budget",
        "rag.min_score",
        "rag.corpus_dir",
        "rag.retrieval_timeout_s",
        "persona.system_prompt",
    ]
    for key in expected_keys:
        assert key in result, f"Expected key '{key}' missing from _flatten_config result"


@pytest.mark.unit
def test_flatten_config_allowed_tools_is_list() -> None:
    """tools.allowed_tools must be a list (not a tuple) in the flattened result."""
    result = _flatten_config(LumiConfig())
    assert isinstance(result["tools.allowed_tools"], list)


@pytest.mark.unit
def test_flatten_config_llm_kv_cache_quant_is_none() -> None:
    """llm.kv_cache_quant default value must be None in the flattened result."""
    result = _flatten_config(LumiConfig())
    assert result["llm.kv_cache_quant"] is None


@pytest.mark.unit
def test_flatten_config_values_match_config_fields() -> None:
    """Spot-check: flattened values must match the actual config field values."""
    config = LumiConfig()
    result = _flatten_config(config)

    assert result["edition"] == config.edition
    assert result["audio.sensitivity"] == config.audio.sensitivity
    assert result["scribe.model_size"] == config.scribe.model_size
    assert result["llm.temperature"] == config.llm.temperature
    assert result["tts.voice"] == config.tts.voice
    assert result["ipc.port"] == config.ipc.port
    assert result["tools.execution_timeout_s"] == config.tools.execution_timeout_s
    assert result["vision.capture_method"] == config.vision.capture_method
    assert result["rag.min_score"] == config.rag.min_score
    assert result["persona.system_prompt"] == config.persona.system_prompt
