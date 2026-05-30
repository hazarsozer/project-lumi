"""
Integration tests for CR-09 retention config wiring (Issue #10).

These tests assert that the ORCHESTRATOR correctly wires:
  1. ConversationMemory is constructed with max_age_s = memory_max_age_days * 86400.
  2. DocumentStore.purge_expired_chunks is called at startup when rag_max_age_days > 0.
  3. purge_expired_chunks is NOT called when rag_max_age_days == 0.0 (default).
  4. purge_expired_chunks is NOT called when RAG is disabled.

All tests mock at the orchestrator boundary — no sqlite_vec or real model required.

RED→GREEN proof
---------------
Before the wiring fix:
  - ConversationMemory is called with only (memory_dir, summariser=...) — no max_age_s.
    test_memory_max_age_s_passed_to_conversation_memory FAILS because the constructor
    call does not include max_age_s.
  - purge_expired_chunks is never called.
    test_rag_store_purged_at_startup_when_configured FAILS because call_count == 0.
    test_rag_store_not_purged_when_zero PASSES trivially but test_rag_store_purged_...
    is the sentinel that catches the gap.
After the wiring fix all four tests pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DAYS_7 = 7.0
_SECS_7 = _DAYS_7 * 86_400.0   # 604800.0

_DAYS_30 = 30.0
_SECS_30 = _DAYS_30 * 86_400.0  # 2592000.0


def _make_config(
    *,
    memory_max_age_days: float = 0.0,
    rag_max_age_days: float = 0.0,
    rag_enabled: bool = False,
) -> MagicMock:
    """Build a minimal MagicMock config that the Orchestrator can consume."""
    cfg = MagicMock()
    cfg.llm.memory_dir = "/tmp/lumi_retention_test_memory"
    cfg.llm.memory_max_age_days = memory_max_age_days
    cfg.llm.max_tokens = 5
    cfg.llm.temperature = 0.7
    cfg.llm.context_length = 512
    cfg.llm.inference_timeout_s = 30.0
    cfg.llm.tools_in_system_prompt = False
    cfg.llm.model_path = ""
    cfg.rag.enabled = rag_enabled
    cfg.rag.rag_max_age_days = rag_max_age_days
    cfg.rag.db_path = "/tmp/lumi_test_rag.db"
    cfg.tools.enabled = False
    cfg.vision.enabled = False
    cfg.ipc.enabled = False
    cfg.audio.send_visemes = False
    cfg.observability.heartbeat_interval_s = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Shared patch context — mirrors pattern from test_orchestrator_rag.py
# ---------------------------------------------------------------------------

def _build_orchestrator(cfg: MagicMock):
    """Construct an Orchestrator under full mocking, returning (orch, mock_mem_cls, mock_store)."""
    speaker = MagicMock()
    speaker.start = MagicMock()
    speaker.stop = MagicMock()

    mock_store_instance = MagicMock()
    mock_store_cls = MagicMock(return_value=mock_store_instance)

    with (
        patch("src.core.orchestrator.ConversationMemory") as mock_mem_cls,
        patch("src.core.orchestrator.ModelLoader"),
        patch("src.core.orchestrator.PromptEngine"),
        patch("src.core.orchestrator.ReasoningRouter") as mock_rr,
        patch("src.core.orchestrator.ReflexRouter") as mock_reflex,
        patch("src.core.orchestrator.ToolRegistry"),
        patch("src.core.orchestrator.ToolExecutor"),
    ):
        mem_instance = MagicMock()
        mem_instance.load = MagicMock()
        mem_instance.get_history = MagicMock(return_value=[])
        mem_instance.add_turn = MagicMock()
        mem_instance.save = MagicMock()
        mock_mem_cls.return_value = mem_instance

        reflex_instance = MagicMock()
        reflex_instance.route.return_value = None
        reflex_instance.route_rag_intent.return_value = False
        mock_reflex.return_value = reflex_instance

        rr_instance = MagicMock()
        rr_instance.generate.return_value = "response"
        mock_rr.return_value = rr_instance

        from src.core.orchestrator import Orchestrator

        if cfg.rag.enabled:
            # DocumentStore and RAGRetriever are imported locally inside the RAG
            # try/except block (from src.rag.*), so we patch their canonical module
            # paths — that's where the local import resolves to.
            with (
                patch("src.rag.store.DocumentStore", mock_store_cls),
                patch("src.rag.retriever.RAGRetriever", MagicMock()),
            ):
                orch = Orchestrator(cfg, speaker=speaker)
        else:
            orch = Orchestrator(cfg, speaker=speaker)

    return orch, mock_mem_cls, mock_store_instance


# ===========================================================================
# 1. ConversationMemory wiring: max_age_s = memory_max_age_days * 86400
# ===========================================================================


class TestConversationMemoryMaxAgeWiring:
    """Orchestrator must pass max_age_s=days*86400 to ConversationMemory."""

    @pytest.mark.unit
    def test_memory_max_age_s_passed_when_nonzero(self) -> None:
        """With memory_max_age_days=7, ConversationMemory must receive max_age_s=604800.0."""
        cfg = _make_config(memory_max_age_days=_DAYS_7)
        _, mock_mem_cls, _ = _build_orchestrator(cfg)

        # The constructor must have been called with max_age_s as a keyword arg.
        assert mock_mem_cls.call_count == 1, "ConversationMemory should be constructed once"
        _, kwargs = mock_mem_cls.call_args
        assert "max_age_s" in kwargs, (
            "ConversationMemory must be called with max_age_s keyword argument; "
            f"actual kwargs: {kwargs}"
        )
        assert kwargs["max_age_s"] == pytest.approx(_SECS_7), (
            f"max_age_s should be {_SECS_7} (7 days * 86400), got {kwargs['max_age_s']}"
        )

    @pytest.mark.unit
    def test_memory_max_age_s_zero_when_days_zero(self) -> None:
        """Default memory_max_age_days=0.0 must produce max_age_s=0.0 (no purge)."""
        cfg = _make_config(memory_max_age_days=0.0)
        _, mock_mem_cls, _ = _build_orchestrator(cfg)

        assert mock_mem_cls.call_count == 1
        _, kwargs = mock_mem_cls.call_args
        # max_age_s must be 0.0 OR the kwarg must be absent (both mean no purge).
        actual_max_age_s = kwargs.get("max_age_s", 0.0)
        assert actual_max_age_s == pytest.approx(0.0), (
            f"max_age_s should be 0.0 for memory_max_age_days=0.0, got {actual_max_age_s}"
        )

    @pytest.mark.unit
    def test_memory_max_age_s_correct_for_30_days(self) -> None:
        """Spot-check: 30 days → 2592000.0 seconds."""
        cfg = _make_config(memory_max_age_days=_DAYS_30)
        _, mock_mem_cls, _ = _build_orchestrator(cfg)

        _, kwargs = mock_mem_cls.call_args
        assert kwargs.get("max_age_s") == pytest.approx(_SECS_30)


# ===========================================================================
# 2. RAG store startup purge: called iff rag_max_age_days > 0
# ===========================================================================


class TestRAGStorePurgeWiring:
    """Orchestrator must call purge_expired_chunks at startup iff rag_max_age_days > 0."""

    @pytest.mark.unit
    def test_rag_store_purged_at_startup_when_configured(self) -> None:
        """With rag_max_age_days=30 and RAG enabled, purge_expired_chunks must be called."""
        cfg = _make_config(rag_enabled=True, rag_max_age_days=_DAYS_30)
        _, _, mock_store = _build_orchestrator(cfg)

        assert mock_store.purge_expired_chunks.call_count == 1, (
            "purge_expired_chunks must be called once at startup when rag_max_age_days > 0; "
            f"actual call_count={mock_store.purge_expired_chunks.call_count}"
        )
        args, kwargs = mock_store.purge_expired_chunks.call_args
        # First positional arg is max_age_s.
        actual_max_age_s = args[0] if args else kwargs.get("max_age_s")
        assert actual_max_age_s == pytest.approx(_SECS_30), (
            f"purge_expired_chunks must be called with max_age_s={_SECS_30} "
            f"(30 days * 86400), got {actual_max_age_s}"
        )

    @pytest.mark.unit
    def test_rag_store_not_purged_when_max_age_zero(self) -> None:
        """Default rag_max_age_days=0.0: purge_expired_chunks must NOT be called."""
        cfg = _make_config(rag_enabled=True, rag_max_age_days=0.0)
        _, _, mock_store = _build_orchestrator(cfg)

        assert mock_store.purge_expired_chunks.call_count == 0, (
            "purge_expired_chunks must NOT be called when rag_max_age_days=0.0; "
            f"actual call_count={mock_store.purge_expired_chunks.call_count}"
        )

    @pytest.mark.unit
    def test_rag_store_not_purged_when_rag_disabled(self) -> None:
        """When RAG is disabled, purge_expired_chunks must never be called (no store exists)."""
        cfg = _make_config(rag_enabled=False, rag_max_age_days=_DAYS_30)
        # RAG is disabled, so _rag_store is None — no purge possible.
        orch, _, mock_store = _build_orchestrator(cfg)
        assert orch._rag_store is None, "_rag_store must be None when RAG is disabled"
        # mock_store is never assigned to orch._rag_store, so purge is unreachable.
        assert mock_store.purge_expired_chunks.call_count == 0
