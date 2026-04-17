"""
Lazy-loaded sentence-embedding singleton for the Lumi RAG pipeline.

A single Embedder instance is cached per model name at the module level.
The model is loaded on the first call to encode() and stays in RAM for the
lifetime of the process.  All operations run on CPU — no GPU allocation.

Thread safety: a per-instance lock serialises concurrent encode() calls so
that the same model object is not used from multiple threads simultaneously.
The inference thread and the ingest CLI run in separate processes, so this
lock only matters if multiple inference threads share one Embedder.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)

# Module-level cache: model_name → Embedder instance.
_instances: dict[str, "Embedder"] = {}
_instances_lock = threading.Lock()


class Embedder:
    """Thin wrapper around SentenceTransformer for the RAG pipeline.

    Use :func:`get_embedder` instead of instantiating directly — it returns
    the cached singleton for the given model name.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # loaded lazily on first encode()
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model '%s' (CPU) ...", self._model_name)
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self._model = SentenceTransformer(self._model_name, device="cpu")
        logger.info("Embedding model loaded.")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.  Returns one float list per input string."""
        if not texts:
            return []
        with self._lock:
            self._ensure_loaded()
            assert self._model is not None
            vecs = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [v.tolist() for v in vecs]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the output vector dimension, loading the model if needed."""
        with self._lock:
            self._ensure_loaded()
            assert self._model is not None
            return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]


def get_embedder(model_name: str) -> Embedder:
    """Return the cached :class:`Embedder` for *model_name*, creating it if needed."""
    with _instances_lock:
        if model_name not in _instances:
            _instances[model_name] = Embedder(model_name)
        return _instances[model_name]
