"""
VRAM lifecycle manager for local GGUF model inference.

Wraps ``llama_cpp.Llama`` with load/unload semantics so the orchestrator
can keep VRAM usage at zero when the model is not actively generating.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

from src.core.config import LLMConfig

logger = logging.getLogger(__name__)

# Shared across all ModelLoader instances and ScreenshotTool so that LLM load
# and vision-model load are serialised — prevents two GGUF models from occupying
# VRAM simultaneously.  Using a module-level lock (not a class attribute) means
# the same object is reachable from both modules without a circular import.
_VRAM_LOCK: threading.Lock = threading.Lock()


class ModelLoader:
    """Manages the lifecycle of a local GGUF model via llama-cpp-python."""

    def __init__(self) -> None:
        self._model: Any | None = None
        # Expose the shared lock so ScreenshotTool can acquire the same object.
        self._vram_lock: threading.Lock = _VRAM_LOCK
        # Holds the numpy cvec buffer alive for the lifetime of the GGUF context.
        # ctypes data_as() does NOT extend the numpy array's lifetime — if this
        # attribute is not set, the GC can free the buffer while llama.cpp holds
        # a dangling pointer, causing silent corruption or segfaults.
        self._cvec_buffer: Any | None = None
        # Cached tokenisation of identity guard strings — built on first call to
        # get_identity_guard_logit_bias() and invalidated on unload().
        self._identity_guard_cache: dict[int, float] | None = None

    def load(self, config: LLMConfig) -> None:
        """Load the LLM into memory.

        Routing is controlled by two config fields:
        - ``persona_steering_enabled`` — master on/off switch (default False).
        - ``persona_steering_backend`` — discriminates the implementation:
          - ``"hf_hooks"``  → HF Phi3ForCausalLM fp16 + residual-stream forward
            hooks (Track 1.A debug backend, ~5 GB extra VRAM).
          - ``"gguf_cvec"`` → native llama_set_adapter_cvec on the GGUF runtime
            (Track A production backend, zero extra VRAM).  The cvec is applied
            inside ``_load_gguf`` after ``Llama()`` construction.

        When steering is disabled (default), loads a GGUF model via
        llama_cpp.Llama and acquires the shared VRAM lock to prevent
        simultaneous GGUF + vision-model occupation.

        Raises:
            FileNotFoundError: If the model path does not exist on disk.
        """
        if config.persona_steering_enabled and config.persona_steering_backend == "hf_hooks":
            self._load_hf_steered(config)
        else:
            self._load_gguf(config)

    def _load_hf_steered(self, config: LLMConfig) -> None:
        from src.llm.hf_steered_model import HFSteeredModel

        hf_path = Path(config.hf_model_path)
        vec_path = Path(config.persona_steering_vector_path)

        if not hf_path.exists():
            raise FileNotFoundError(f"HF model directory not found: {hf_path}")
        if not vec_path.exists():
            raise FileNotFoundError(f"Steering vector file not found: {vec_path}")

        # Resolve device independently of n_gpu_layers — that field is a GGUF
        # partial-offload count with no meaning for the HF backend.
        import torch as _torch
        if config.persona_steering_device == "auto":
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            device = config.persona_steering_device
        self._model = HFSteeredModel(
            hf_model_path=hf_path,
            vector_path=vec_path,
            hooked_layers=config.persona_steering_layers,
            alpha=config.persona_steering_alpha,
            device=device,
        )
        logger.info(
            "Steered HF model loaded from %s (layers=%s alpha=%.1f device=%s)",
            hf_path,
            list(config.persona_steering_layers),
            config.persona_steering_alpha,
            device,
        )

    def _load_gguf(self, config: LLMConfig) -> None:
        """Load a GGUF model into memory via llama_cpp.

        Acquires the shared VRAM lock so this call is mutually exclusive with
        ScreenshotTool's vision-model load, preventing two GGUF models from
        occupying VRAM simultaneously.

        If ``config.kv_cache_quant`` is set, ``cache_type_k`` / ``cache_type_v``
        are forwarded to ``llama_cpp.Llama``.  Older llama-cpp-python builds
        that predate upstream PR #21089 do not accept those kwargs — we detect
        the resulting ``TypeError``, log a warning, and retry with the default
        FP16 cache so the model still loads.

        Raises:
            FileNotFoundError: If ``config.model_path`` does not exist on disk.
        """
        import llama_cpp  # optional extra; only needed at inference time

        model_path = Path(config.model_path)

        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_gpu_layers": config.n_gpu_layers,
            "n_ctx": config.context_length,
            "verbose": False,
        }
        if config.kv_cache_quant is not None:
            kwargs["cache_type_k"] = config.kv_cache_quant
            kwargs["cache_type_v"] = config.kv_cache_quant

        if config.lora_path is not None:
            kwargs["lora_path"] = config.lora_path
            kwargs["lora_scale"] = config.lora_scale

        with self._vram_lock:
            try:
                self._model = llama_cpp.Llama(**kwargs)
            except TypeError as exc:
                if "cache_type" in str(exc):
                    logger.warning(
                        "TurboQuant KV cache types not supported by installed "
                        "llama-cpp-python (upstream PR #21089 not yet shipped); "
                        "falling back to FP16 KV cache"
                    )
                    kwargs.pop("cache_type_k", None)
                    kwargs.pop("cache_type_v", None)
                    self._model = llama_cpp.Llama(**kwargs)
                else:
                    raise
            except ValueError as exc:
                if "does not exist" in str(exc):
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}"
                    ) from exc
                raise

        logger.info(
            "GGUF model loaded from %s (n_gpu_layers=%d, n_ctx=%d%s)",
            model_path,
            config.n_gpu_layers,
            config.context_length,
            f", lora={config.lora_path}" if config.lora_path else "",
        )

        if config.persona_steering_enabled and config.persona_steering_backend == "gguf_cvec":
            self._apply_gguf_cvec(config)

    def _apply_gguf_cvec(self, config: LLMConfig) -> None:
        """Apply a pre-exported control vector to the live GGUF context.

        Reads the flat float32 .cvec.bin file produced by
        scripts/export_cvec_to_llama.py and passes it to
        llama_set_adapter_cvec().  The buffer is held on self so the GC
        cannot free it while llama.cpp holds the pointer.

        Buffer layout: [n_layers_total × n_embd] float32, where entry
        layer_i * n_embd corresponds to transformer layer layer_i.  Layers
        outside [il_start, il_end] must be zero (handled at export time).

        Sign convention: the exporter negates the direction so that llama.cpp's
        additive residual update reproduces the HF hook's subtractive effect.
        """
        import llama_cpp
        import numpy as np

        bin_path = Path(config.persona_steering_vector_path)
        if bin_path.suffix == ".pt":
            raise ValueError(
                "persona_steering_vector_path must be a .cvec.bin file produced by "
                "scripts/export_cvec_to_llama.py, not a .pt file. "
                f"Got: {bin_path}"
            )
        if not bin_path.exists():
            raise FileNotFoundError(f"cvec buffer not found: {bin_path}")

        buf = np.fromfile(bin_path, dtype=np.float32)
        il_start, il_end = config.persona_steering_layer_range
        n_layers_total = 32  # Phi-3.5-mini; consistent with export convention
        n_embd = buf.size // n_layers_total
        if buf.size != n_layers_total * n_embd:
            raise RuntimeError(
                f"cvec buffer size {buf.size} not divisible by "
                f"n_layers_total={n_layers_total}"
            )

        buf = np.ascontiguousarray(buf, dtype=np.float32)
        # Hold ref on self BEFORE calling data_as() — ctypes pointer does not
        # extend the numpy array lifetime.
        self._cvec_buffer = buf
        data_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rc = llama_cpp.llama_set_adapter_cvec(
            self._model.ctx,
            data_ptr,
            buf.size,
            n_embd,
            il_start,
            il_end,
        )
        if rc != 0:
            raise RuntimeError(f"llama_set_adapter_cvec failed: rc={rc}")

        checksum = hashlib.sha256(buf.tobytes()).hexdigest()[:16]
        logger.info(
            "GGUF cvec applied: layers=[%d, %d] n_embd=%d checksum=%s path=%s",
            il_start, il_end, n_embd, checksum, bin_path,
        )

    def get_identity_guard_logit_bias(self, config: LLMConfig) -> dict[int, float]:
        """Return {token_id: bias} for suppressing Phi self-ID tokens.

        Tokenises each string in config.identity_guard_tokens exactly once
        and caches the result.  Returns an empty dict when the guard is
        disabled or no model is loaded.
        """
        if not config.identity_guard_enabled or self._model is None:
            return {}
        if self._identity_guard_cache is not None:
            return self._identity_guard_cache
        bias: dict[int, float] = {}
        for token_str in config.identity_guard_tokens:
            try:
                tids = self._model.tokenize(token_str.encode(), add_bos=False)
                for tid in tids:
                    bias[tid] = config.identity_guard_bias
            except Exception:
                logger.warning("identity guard: tokenisation failed for %r", token_str)
        self._identity_guard_cache = bias
        logger.info(
            "Identity guard active: %d token IDs suppressed (bias=%.0f)",
            len(bias),
            config.identity_guard_bias,
        )
        return bias

    def unload(self) -> None:
        """Release the model reference so memory can be reclaimed."""
        self._model = None
        self._identity_guard_cache = None
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is currently held in memory."""
        return self._model is not None

    @property
    def model(self) -> Any:
        """Return the underlying llama_cpp.Llama instance.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        return self._model
