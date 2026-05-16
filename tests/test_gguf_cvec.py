"""Tests for the native GGUF control-vector wiring in model_loader.py.

Tier 1 — CI (mock, fast): mocks the llama_cpp module so no LLM install is
    needed.  Verifies the correct ctypes call is made, the buffer ref is
    retained, and error paths raise cleanly.

Tier 2 — slow (live, requires GGUF on disk and --cvec-live flag): loads
    the real model, applies the cvec, and asserts logits change vs. baseline.
    Run with:  uv run pytest tests/test_gguf_cvec.py --cvec-live -v
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import LLMConfig
from src.llm.model_loader import ModelLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LAYERS = 32
N_EMBD = 3072
IL_START, IL_END = 12, 28


def _make_cvec_bin(tmp_path: Path, alpha: float = 8.0) -> Path:
    buf = np.zeros((N_LAYERS, N_EMBD), dtype=np.float32)
    for layer in range(IL_START, IL_END + 1):
        buf[layer] = -alpha * np.ones(N_EMBD, dtype=np.float32) * 0.01
    out = tmp_path / "test.cvec.bin"
    buf.flatten().tofile(out)
    return out


def _make_config(cvec_path: Path) -> LLMConfig:
    return LLMConfig(
        model_path="dummy.gguf",
        persona_steering_enabled=True,
        persona_steering_backend="gguf_cvec",
        persona_steering_vector_path=str(cvec_path),
        persona_steering_layer_range=(IL_START, IL_END),
    )


def _mock_llama_cpp() -> tuple[ModuleType, MagicMock]:
    """Return (mock_module, mock_set_adapter_cvec)."""
    mock_cvec_fn = MagicMock(return_value=0)
    mock_mod = MagicMock(spec=ModuleType)
    mock_mod.llama_set_adapter_cvec = mock_cvec_fn
    return mock_mod, mock_cvec_fn


# ---------------------------------------------------------------------------
# Tier 1: mock tests (run in CI without llm extra)
# ---------------------------------------------------------------------------


class TestApplyGgufCvec:
    def test_set_adapter_cvec_called_with_correct_args(self, tmp_path: Path) -> None:
        """llama_set_adapter_cvec is called with the right shape arguments."""
        cvec = _make_cvec_bin(tmp_path)
        config = _make_config(cvec)
        loader = ModelLoader()
        loader._model = MagicMock()
        loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

        mock_mod, mock_cvec_fn = _mock_llama_cpp()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            loader._apply_gguf_cvec(config)

        mock_cvec_fn.assert_called_once()
        _, _, buf_len, n_embd, il_s, il_e = mock_cvec_fn.call_args[0]
        assert buf_len == N_LAYERS * N_EMBD
        assert n_embd == N_EMBD
        assert il_s == IL_START
        assert il_e == IL_END

    def test_raises_on_nonzero_rc(self, tmp_path: Path) -> None:
        cvec = _make_cvec_bin(tmp_path)
        config = _make_config(cvec)
        loader = ModelLoader()
        loader._model = MagicMock()
        loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

        mock_mod, _ = _mock_llama_cpp()
        mock_mod.llama_set_adapter_cvec.return_value = 1
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with pytest.raises(RuntimeError, match="llama_set_adapter_cvec failed"):
                loader._apply_gguf_cvec(config)

    def test_buffer_held_on_loader(self, tmp_path: Path) -> None:
        """_cvec_buffer must be set before data_as() to prevent GC."""
        cvec = _make_cvec_bin(tmp_path)
        config = _make_config(cvec)
        loader = ModelLoader()
        loader._model = MagicMock()
        loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

        mock_mod, _ = _mock_llama_cpp()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            loader._apply_gguf_cvec(config)

        assert loader._cvec_buffer is not None
        assert isinstance(loader._cvec_buffer, np.ndarray)
        assert loader._cvec_buffer.dtype == np.float32
        assert loader._cvec_buffer.size == N_LAYERS * N_EMBD

    def test_skipped_when_steering_disabled(self) -> None:
        """load() must not call _apply_gguf_cvec when steering is off."""
        config = LLMConfig(
            model_path="dummy.gguf",
            persona_steering_enabled=False,
        )
        loader = ModelLoader()
        with patch.object(loader, "_load_gguf"):
            with patch.object(loader, "_apply_gguf_cvec") as mock_apply:
                loader.load(config)
        mock_apply.assert_not_called()

    def test_raises_on_pt_path(self, tmp_path: Path) -> None:
        config = LLMConfig(
            model_path="dummy.gguf",
            persona_steering_enabled=True,
            persona_steering_backend="gguf_cvec",
            persona_steering_vector_path="models/persona_vectors/phi_prior_v1.pt",
        )
        loader = ModelLoader()
        loader._model = MagicMock()
        mock_mod, _ = _mock_llama_cpp()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with pytest.raises(ValueError, match=r"\.cvec\.bin"):
                loader._apply_gguf_cvec(config)

    def test_raises_on_missing_bin(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path / "nonexistent.cvec.bin")
        loader = ModelLoader()
        loader._model = MagicMock()
        mock_mod, _ = _mock_llama_cpp()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with pytest.raises(FileNotFoundError, match="cvec buffer not found"):
                loader._apply_gguf_cvec(config)

    def test_checksum_logged(self, tmp_path: Path, caplog) -> None:
        import logging
        cvec = _make_cvec_bin(tmp_path)
        config = _make_config(cvec)
        loader = ModelLoader()
        loader._model = MagicMock()
        loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

        mock_mod, _ = _mock_llama_cpp()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with caplog.at_level(logging.INFO, logger="src.llm.model_loader"):
                loader._apply_gguf_cvec(config)

        assert any("checksum=" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tier 2: live logit-diff test (skipped unless --cvec-live)
# ---------------------------------------------------------------------------

GGUF_PATH = Path("models/llm/lumi-phi35-v2.1-Q5_K_M.gguf")
CVEC_PATH = Path("models/persona_vectors/phi_prior_v1_alpha8_l12-28.cvec.bin")


@pytest.mark.slow
def test_cvec_changes_logits(request) -> None:
    """Live R1/R7 detector: logits differ before/after cvec by norm > 1e-3.

    If llama_set_adapter_cvec is a silent no-op, logits_off == logits_on
    and this test fails, directly catching risks R1 and R7.
    """
    if not getattr(request.config.option, "cvec_live", False):
        pytest.skip("Pass --cvec-live to run live GGUF cvec tests.")
    if not GGUF_PATH.exists():
        pytest.skip(f"GGUF not found: {GGUF_PATH}")
    if not CVEC_PATH.exists():
        pytest.skip(f"cvec bin not found: {CVEC_PATH}")

    try:
        import llama_cpp
    except ImportError:
        pytest.skip("llama_cpp not installed (uv sync --extra llm)")

    probe = "Send an email to my boss."

    def _get_logits(apply_cvec: bool) -> np.ndarray:
        model = llama_cpp.Llama(model_path=str(GGUF_PATH), n_gpu_layers=-1,
                                n_ctx=512, verbose=False)
        if apply_cvec:
            buf = np.fromfile(CVEC_PATH, dtype=np.float32)
            buf = np.ascontiguousarray(buf)
            data_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            n_embd = buf.size // N_LAYERS
            rc = llama_cpp.llama_set_adapter_cvec(
                model.ctx, data_ptr, buf.size, n_embd, IL_START, IL_END,
            )
            assert rc == 0, f"llama_set_adapter_cvec rc={rc}"
        tokens = model.tokenize(probe.encode())
        model.eval(tokens)
        logits = np.array(model.eval_logits[-1], dtype=np.float32)
        del model
        return logits

    logits_off = _get_logits(apply_cvec=False)
    logits_on  = _get_logits(apply_cvec=True)
    diff_norm = float(np.linalg.norm(logits_off - logits_on))
    assert diff_norm > 1e-3, (
        f"cvec is a silent no-op: ||logits_off - logits_on|| = {diff_norm:.6f}"
    )
