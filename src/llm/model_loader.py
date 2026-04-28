"""
VRAM lifecycle manager for local GGUF model inference.

Wraps ``llama_cpp.Llama`` with load/unload semantics so the orchestrator
can keep VRAM usage at zero when the model is not actively generating.
"""

from __future__ import annotations

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

    def load(self, config: LLMConfig) -> None:
        """Load a GGUF model into memory.

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
            "Model loaded from %s (n_gpu_layers=%d, n_ctx=%d%s)",
            model_path,
            config.n_gpu_layers,
            config.context_length,
            f", lora={config.lora_path}" if config.lora_path else "",
        )

    def unload(self) -> None:
        """Release the model reference so memory can be reclaimed."""
        self._model = None
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
