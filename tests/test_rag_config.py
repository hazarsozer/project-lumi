"""Tests for RAGConfig — defaults, YAML merge, and LumiConfig wiring."""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from src.core.config import RAGConfig, LumiConfig, load_config


class TestRAGConfigDefaults:
    def test_disabled_by_default(self):
        assert RAGConfig().enabled is False

    def test_default_db_path(self):
        assert RAGConfig().db_path == "~/.lumi/rag.db"

    def test_default_embedding_model(self):
        assert RAGConfig().embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_default_chunk_size(self):
        assert RAGConfig().chunk_size == 512

    def test_default_chunk_overlap(self):
        assert RAGConfig().chunk_overlap == 64

    def test_default_retrieval_top_k(self):
        assert RAGConfig().retrieval_top_k == 8

    def test_default_context_char_budget(self):
        assert RAGConfig().context_char_budget == 2400

    def test_default_min_score(self):
        assert RAGConfig().min_score == 0.15

    def test_default_corpus_dir(self):
        assert RAGConfig().corpus_dir == "~/.lumi/docs"

    def test_default_retrieval_timeout(self):
        assert RAGConfig().retrieval_timeout_s == 0.4

    def test_frozen(self):
        cfg = RAGConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.enabled = True  # type: ignore[misc]


class TestRAGConfigYAMLMerge:
    def _write_yaml(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        f.write(textwrap.dedent(content))
        f.flush()
        return f.name

    def test_enabled_overrides(self):
        path = self._write_yaml("rag:\n  enabled: true\n")
        cfg = load_config(path)
        assert cfg.rag.enabled is True

    def test_db_path_overrides(self):
        path = self._write_yaml("rag:\n  db_path: /tmp/test.db\n")
        cfg = load_config(path)
        assert cfg.rag.db_path == "/tmp/test.db"

    def test_chunk_size_overrides(self):
        path = self._write_yaml("rag:\n  chunk_size: 256\n")
        cfg = load_config(path)
        assert cfg.rag.chunk_size == 256

    def test_min_score_overrides(self):
        path = self._write_yaml("rag:\n  min_score: 0.3\n")
        cfg = load_config(path)
        assert cfg.rag.min_score == pytest.approx(0.3)

    def test_unknown_key_ignored(self):
        path = self._write_yaml("rag:\n  nonexistent_key: 99\n")
        cfg = load_config(path)
        assert cfg.rag.chunk_size == 512  # default unchanged

    def test_empty_rag_section_uses_defaults(self):
        path = self._write_yaml("rag: {}\n")
        cfg = load_config(path)
        assert cfg.rag.enabled is False
        assert cfg.rag.retrieval_top_k == 8

    def test_absent_rag_section_uses_defaults(self):
        path = self._write_yaml("log_level: INFO\n")
        cfg = load_config(path)
        assert isinstance(cfg.rag, RAGConfig)
        assert cfg.rag.enabled is False


class TestLumiConfigWiring:
    def test_lumi_config_has_rag_field(self):
        cfg = LumiConfig()
        assert hasattr(cfg, "rag")
        assert isinstance(cfg.rag, RAGConfig)

    def test_rag_defaults_independent_of_other_sections(self):
        cfg = LumiConfig()
        assert cfg.rag.enabled is False
        assert cfg.llm.n_gpu_layers == 0
