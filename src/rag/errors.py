"""Exceptions raised by the src.rag package."""


class RAGUnavailableError(RuntimeError):
    """Raised when the RAG subsystem cannot initialise.

    Common causes: sqlite-vec extension not loadable, database path not
    writable, or the embedding model failed to load.
    """


class IngestError(RuntimeError):
    """Raised when document ingestion fails for a specific file."""


class RetrievalError(RuntimeError):
    """Raised when a retrieval query fails unexpectedly at runtime."""
