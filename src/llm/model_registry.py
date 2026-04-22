"""Named GGUF model registry with hot-swap support."""

from __future__ import annotations

import logging
from typing import Any

from src.core.config import LLMConfig
from src.llm.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry of named GGUF model configs with sequential hot-swap.

    Owns one ModelLoader internally. Switching models unloads the current
    GGUF and loads the requested one (~2.5–7s). This is the recommended
    strategy for llama-cpp-python 0.3.20, which exposes LoRA bindings only
    at the C level with no high-level hot-swap API (see docs/lora_api_probe.md).
    """

    def __init__(self) -> None:
        self._configs: dict[str, LLMConfig] = {}
        self._loader: ModelLoader = ModelLoader()
        self._current_name: str | None = None

    def register(self, name: str, config: LLMConfig) -> None:
        """Register a named model config. Silently overwrites if name exists."""
        self._configs[name] = config
        logger.debug("Registered model %r", name)

    def load(self, name: str) -> None:
        """Unload current model (if any) and load the named model.

        Args:
            name: A name previously passed to register().

        Raises:
            KeyError: If name has not been registered.
        """
        if name not in self._configs:
            raise KeyError(f"Model {name!r} is not registered")
        if self._loader.is_loaded:
            self._loader.unload()
        self._loader.load(self._configs[name])
        self._current_name = name
        logger.info("Loaded model %r", name)

    def unload(self) -> None:
        """Unload the current model without loading another."""
        self._loader.unload()
        self._current_name = None

    @property
    def current_name(self) -> str | None:
        """Name of the currently loaded model, or None if none is loaded."""
        return self._current_name

    @property
    def is_loaded(self) -> bool:
        """True if a model is currently loaded."""
        return self._loader.is_loaded

    @property
    def model(self) -> Any:
        """The underlying llama_cpp.Llama instance.

        Raises:
            RuntimeError: If no model is loaded.
        """
        return self._loader.model

    def list_registered(self) -> list[str]:
        """Return names of all registered model configs."""
        return list(self._configs)
