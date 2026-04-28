"""Named GGUF model registry with hot-swap support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.core.config import LLMConfig
from src.llm.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterSpec:
    """Maps a (persona, task) pair to a LoRA adapter path and scale."""

    persona: str
    task: str | None
    lora_path: str
    lora_scale: float = 1.0


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
        self._adapters: dict[tuple[str, str | None], AdapterSpec] = {}

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

    def register_adapter(self, spec: AdapterSpec) -> None:
        """Register a LoRA adapter spec keyed by (persona, task).

        Silently overwrites any existing entry for the same (persona, task) pair.
        """
        self._adapters[(spec.persona, spec.task)] = spec
        logger.debug(
            "Registered adapter for persona=%r task=%r at %s",
            spec.persona,
            spec.task,
            spec.lora_path,
        )

    def resolve(self, persona: str, task: str | None = None) -> AdapterSpec | None:
        """Look up a LoRA adapter for a (persona, task) pair.

        Resolution order:
        1. Exact match on (persona, task).
        2. Fallback to (persona, None) when the exact match is missing.
        3. Returns None when neither entry exists.
        """
        exact = self._adapters.get((persona, task))
        if exact is not None:
            return exact
        if task is not None:
            return self._adapters.get((persona, None))
        return None
