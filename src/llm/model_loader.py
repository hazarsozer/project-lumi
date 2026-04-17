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

import llama_cpp

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

        Raises:
            FileNotFoundError: If ``config.model_path`` does not exist on disk.
        """
        model_path = Path(config.model_path)

        with self._vram_lock:
            try:
                self._model = llama_cpp.Llama(
                    model_path=str(model_path),
                    n_gpu_layers=config.n_gpu_layers,
                    n_ctx=config.context_length,
                    verbose=False,
                )
            except ValueError as exc:
                if "does not exist" in str(exc):
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}"
                    ) from exc
                raise

        logger.info("Model loaded from %s (n_gpu_layers=%d, n_ctx=%d)",
                     model_path, config.n_gpu_layers, config.context_length)

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
