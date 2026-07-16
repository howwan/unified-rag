"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Unified interface for vector storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Lazy-load connections and create schema if needed."""
        ...

    @abstractmethod
    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Batch insert or update chunks."""
        ...

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Returns list of dicts with keys: id, text, metadata, score.
        Score must be cosine similarity in [0, 1].
        """
        ...

    @abstractmethod
    def delete_all(self) -> None:
        """Remove all stored chunks."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...
