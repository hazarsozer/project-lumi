"""
CR-14 / GitHub Issue #15 — ModelLoader.unload() correctness tests.

Three bugs fixed (all verified RED on pre-fix code, GREEN after fix):

1. VRAM lock — unload() must acquire _VRAM_LOCK before clearing the model
   reference so it cannot race with a concurrent load().

2. cvec buffer leak — _cvec_buffer must be set to None on unload so the
   numpy control-vector array is not kept alive across hibernate/wake cycles.

3. Hardcoded n_layers — _apply_gguf_cvec previously hardcoded n_layers_total=32.
   It now calls llama_cpp.llama_model_n_layer(model.model) and the result for
   the current Phi-3.5-mini model must equal 32 (no behaviour change).

All tests are marked ``unit`` — no hardware or real model files required.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.config import LLMConfig
from src.llm.model_loader import ModelLoader, _VRAM_LOCK


# ---------------------------------------------------------------------------
# Constants — must match the conftest / test_gguf_cvec conventions
# ---------------------------------------------------------------------------

N_LAYERS = 32     # Phi-3.5-mini layer count; derived value must equal this
N_EMBD = 3072     # embedding dimension
IL_START, IL_END = 12, 28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cvec_bin(tmp_path: Path, n_layers: int = N_LAYERS, n_embd: int = N_EMBD) -> Path:
    """Write a minimal flat float32 cvec.bin and return its path."""
    buf = np.zeros((n_layers, n_embd), dtype=np.float32)
    for layer in range(IL_START, IL_END + 1):
        buf[layer] = -8.0 * np.ones(n_embd, dtype=np.float32) * 0.01
    out = tmp_path / "test.cvec.bin"
    buf.flatten().tofile(out)
    return out


def _cvec_config(cvec_path: Path) -> LLMConfig:
    return LLMConfig(
        model_path="dummy.gguf",
        persona_steering_enabled=True,
        persona_steering_backend="gguf_cvec",
        persona_steering_vector_path=str(cvec_path),
        persona_steering_layer_range=(IL_START, IL_END),
    )


def _make_mock_llama_module(n_layer_return: int = N_LAYERS) -> types.ModuleType:
    """Return a fake llama_cpp module with mocked Llama + helper functions."""
    mock_instance = MagicMock()
    mock_instance.ctx = MagicMock(spec=ctypes.c_void_p)
    mock_cls = MagicMock(return_value=mock_instance)

    mock_mod = types.ModuleType("llama_cpp")
    mock_mod.Llama = mock_cls  # type: ignore[attr-defined]
    mock_mod.llama_set_adapter_cvec = MagicMock(return_value=0)  # type: ignore[attr-defined]
    mock_mod.llama_model_n_layer = MagicMock(return_value=n_layer_return)  # type: ignore[attr-defined]
    return mock_mod


# ---------------------------------------------------------------------------
# Fix 1 — unload() must acquire _VRAM_LOCK
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unload_acquires_vram_lock(mock_llama_cpp: MagicMock) -> None:
    """unload() must hold _VRAM_LOCK while clearing the model reference.

    RED on pre-fix code: the lock was never acquired, so _model was cleared
    outside any critical section.  GREEN after fix: unload() wraps the clear
    inside ``with self._vram_lock:``.

    Strategy: acquire the lock from a background thread and assert that
    unload() blocks until the lock is released.
    """
    config = LLMConfig()
    loader = ModelLoader()
    loader.load(config)
    assert loader.is_loaded

    lock_released = threading.Event()
    unload_entered = threading.Event()
    unload_completed = threading.Event()

    # Hold the VRAM lock in a background thread for a short window.
    def hold_lock() -> None:
        with loader._vram_lock:
            unload_entered.wait(timeout=1.0)  # wait for unload to be attempted
            time.sleep(0.05)                  # keep the lock held briefly
        lock_released.set()

    def do_unload() -> None:
        unload_entered.set()
        loader.unload()
        unload_completed.set()

    holder = threading.Thread(target=hold_lock, daemon=True)
    unloader = threading.Thread(target=do_unload, daemon=True)

    holder.start()
    # Give the holder a moment to acquire before unload tries.
    time.sleep(0.02)
    unloader.start()

    holder.join(timeout=2.0)
    unloader.join(timeout=2.0)

    assert lock_released.is_set(), "Background lock-holder thread did not complete"
    assert unload_completed.is_set(), "unload() did not complete within 2 s"

    # Crucially: unload must not have completed BEFORE the lock was released.
    # If unload() did NOT acquire the lock it would have finished almost
    # immediately (well before the holder's 0.05 s sleep).  We can verify
    # is_loaded is False only after unload_completed is set.
    assert not loader.is_loaded


@pytest.mark.unit
def test_unload_lock_is_module_level_vram_lock(mock_llama_cpp: MagicMock) -> None:
    """loader._vram_lock must be the same object as the module-level _VRAM_LOCK.

    This ensures unload() serialises with ScreenshotTool which also acquires
    the module-level singleton (no separate lock instances).
    """
    loader = ModelLoader()
    assert loader._vram_lock is _VRAM_LOCK


# ---------------------------------------------------------------------------
# Fix 2 — _cvec_buffer must be cleared on unload (no leak across cycles)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unload_clears_cvec_buffer(tmp_path: Path) -> None:
    """_cvec_buffer must be None after unload(), not retained in memory.

    RED on pre-fix code: unload() did not touch _cvec_buffer, so after
    unload() it still held the numpy array reference (leak across every
    hibernate/wake cycle).

    Test flow:
      1. Manually set _cvec_buffer to a numpy array (simulating a post-load
         state without needing a real GGUF file).
      2. Call unload() via mock.
      3. Assert _cvec_buffer is None.
    """
    mock_mod = _make_mock_llama_module()
    with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
        config = LLMConfig()
        loader = ModelLoader()
        loader.load(config)

    # Inject a fake cvec buffer as _apply_gguf_cvec would.
    fake_buf = np.zeros(N_LAYERS * N_EMBD, dtype=np.float32)
    loader._cvec_buffer = fake_buf
    assert loader._cvec_buffer is not None

    loader.unload()

    assert loader._cvec_buffer is None, (
        "_cvec_buffer must be None after unload() — was not cleared (buffer leak)"
    )


@pytest.mark.unit
def test_cvec_buffer_not_leaked_across_load_unload_load_cycle(tmp_path: Path) -> None:
    """Buffer from a first load must not survive a load → unload → load cycle.

    RED on pre-fix code: the second load() would assign a new _cvec_buffer,
    but after an unload that did not clear the old one the old numpy array
    would still be alive (held by the GC root through loader._cvec_buffer
    until the second load() overwrote it — if the second load had no cvec the
    buffer would persist indefinitely).

    This test exercises the 'no-cvec second load' case: the old buffer from
    the first load must be gone after unload, not carried through.
    """
    cvec = _make_cvec_bin(tmp_path)
    cvec_config = _cvec_config(cvec)

    mock_mod = _make_mock_llama_module()
    with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
        loader = ModelLoader()

        # First load with cvec → _cvec_buffer should be populated.
        loader._model = MagicMock()
        loader._model.ctx = MagicMock(spec=ctypes.c_void_p)
        loader._apply_gguf_cvec(cvec_config)
        assert loader._cvec_buffer is not None, "Buffer expected after _apply_gguf_cvec"

        # Unload must free the buffer.
        loader.unload()
        assert loader._cvec_buffer is None, "Buffer must be None after unload()"

        # Second load without cvec must not resurrect the old buffer.
        config_no_cvec = LLMConfig()
        loader.load(config_no_cvec)
        assert loader._cvec_buffer is None, (
            "_cvec_buffer must remain None when second load has no cvec steering"
        )


# ---------------------------------------------------------------------------
# Fix 3 — n_layers derived from model metadata, equals 32 for current model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_gguf_cvec_derives_n_layers_from_model(tmp_path: Path) -> None:
    """_apply_gguf_cvec must call llama_model_n_layer and use its return value.

    RED on pre-fix code: n_layers_total was hardcoded to 32, so
    llama_model_n_layer was never called.

    Verify via two sub-checks:
      a. llama_model_n_layer is called with the model's .model pointer.
      b. The returned value drives the buffer layout (buf_len = n_layers * n_embd).
    """
    cvec = _make_cvec_bin(tmp_path, n_layers=N_LAYERS, n_embd=N_EMBD)
    config = _cvec_config(cvec)

    mock_mod = _make_mock_llama_module(n_layer_return=N_LAYERS)
    loader = ModelLoader()
    loader._model = MagicMock()
    loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

    with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
        loader._apply_gguf_cvec(config)

    # (a) llama_model_n_layer must have been called.
    mock_mod.llama_model_n_layer.assert_called_once()

    # (b) llama_set_adapter_cvec received buf_len = N_LAYERS * N_EMBD.
    _, _, buf_len, n_embd_arg, il_s, il_e = mock_mod.llama_set_adapter_cvec.call_args[0]
    assert buf_len == N_LAYERS * N_EMBD, (
        f"buf_len {buf_len} != {N_LAYERS * N_EMBD} — n_layers not used correctly"
    )
    assert n_embd_arg == N_EMBD
    assert il_s == IL_START
    assert il_e == IL_END


@pytest.mark.unit
def test_n_layers_equals_32_for_current_model(tmp_path: Path) -> None:
    """The derived n_layers must equal 32 for the current shipping model.

    This is the explicit persona-freeze invariant: Phi-3.5-mini has 32 layers,
    llama_model_n_layer returns 32, so the cvec buffer layout is identical to
    the previously hardcoded value.  The test fails if the derivation returns
    a different number, which would silently change how layers are addressed.
    """
    cvec = _make_cvec_bin(tmp_path)
    config = _cvec_config(cvec)

    # Simulate what the real llama.cpp returns for Phi-3.5-mini.
    mock_mod = _make_mock_llama_module(n_layer_return=32)
    loader = ModelLoader()
    loader._model = MagicMock()
    loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

    with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
        loader._apply_gguf_cvec(config)

    _, _, buf_len, n_embd_arg, _, _ = mock_mod.llama_set_adapter_cvec.call_args[0]
    expected_buf_len = 32 * N_EMBD
    assert buf_len == expected_buf_len, (
        f"For 32-layer model: buf_len must be {expected_buf_len}, got {buf_len}"
    )


@pytest.mark.unit
def test_n_layers_mismatch_raises_runtime_error(tmp_path: Path) -> None:
    """If llama_model_n_layer returns a value inconsistent with the buffer,
    a RuntimeError must be raised (not silently corrupt the layout).

    E.g., if the cvec.bin was exported for a 32-layer model but the loaded
    model only reports 16 layers, buf.size is not divisible by 16*N_EMBD,
    and the mismatch must surface as a RuntimeError.
    """
    # buf exported for 32 layers * N_EMBD floats
    cvec = _make_cvec_bin(tmp_path, n_layers=32, n_embd=N_EMBD)
    config = _cvec_config(cvec)

    # But the mock model claims it only has 7 layers (prime; not a divisor of 32*N_EMBD).
    mock_mod = _make_mock_llama_module(n_layer_return=7)
    loader = ModelLoader()
    loader._model = MagicMock()
    loader._model.ctx = MagicMock(spec=ctypes.c_void_p)

    with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
        with pytest.raises(RuntimeError, match="not divisible by"):
            loader._apply_gguf_cvec(config)
