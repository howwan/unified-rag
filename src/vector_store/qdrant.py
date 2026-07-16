"""Qdrant backend implementation."""

import logging
import uuid
from typing import Any

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


def _to_qdrant_id(s: str) -> str:
    """Qdrant requires unsigned integer or UUID point IDs.
    Convert arbitrary string to deterministic UUID5.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


class QdrantVectorStore(BaseVectorStore):
    """Qdrant remote/local vector store."""

    def __init__(self, config: dict) -> None:
        self._host = config.get("QDRANT_HOST", "localhost")
        self._port = int(config.get("QDRANT_PORT", "6333"))
        self._collection_name = config.get("QDRANT_COLLECTION", "rag_documents")
        self._client = None
        self._dim: int | None = None

    def initialize(self, probe_dim: int | None = None) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(host=self._host, port=self._port)
        logger.info("QdrantVectorStore connected: %s:%s", self._host, self._port)

        # Check if collection exists and validate dimension
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if exists:
            info = self._client.get_collection(self._collection_name)
            existing_dim = info.config.params.vectors.size
            if probe_dim and existing_dim != probe_dim:
                logger.warning(
                    "Existing Qdrant collection has %s-dim vectors, but current "
                    "model produces %s-dim vectors. Run 'reset' and re-index "
                    "if you switched models.",
                    existing_dim,
                    probe_dim,
                )
            else:
                logger.info(
                    "Qdrant collection '%s' exists with dimension %s",
                    self._collection_name,
                    existing_dim,
                )
            self._dim = existing_dim
        else:
            if probe_dim:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=probe_dim, distance=Distance.COSINE),
                )
                self._dim = probe_dim
                logger.info(
                    "Created Qdrant collection '%s' with dimension %s",
                    self._collection_name,
                    probe_dim,
                )
            else:
                logger.warning(
                    "Qdrant collection '%s' does not exist and no probe_dim provided. "
                    "Collection will be created on first upsert.",
                    self._collection_name,
                )

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if self._client is None:
            raise RuntimeError("Store not initialized")

        from qdrant_client.models import PointStruct

        # Auto-create collection if missing (e.g., after delete_all without probe_dim)
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)
        if not exists:
            if embeddings:
                dim = len(embeddings[0])
                from qdrant_client.models import Distance, VectorParams
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                self._dim = dim
                logger.info("Auto-created Qdrant collection with dimension %s", dim)

        points = []
        for i, doc_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas[i] else {}
            points.append(
                PointStruct(
                    id=_to_qdrant_id(doc_id),
                    vector=embeddings[i],
                    payload={
                        "_original_id": doc_id,
                        "text": texts[i],
                        "title": meta.get("title", ""),
                        "source": meta.get("source", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                    },
                )
            )

        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("Store not initialized")

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        output = []
        for p in response.points:
            # Qdrant returns cosine *similarity* directly in newer query_points API
            # (score field). Range is [0, 1] for normalized vectors.
            score = float(p.score) if p.score is not None else 0.0
            payload = p.payload or {}
            output.append({
                "id": payload.get("_original_id", str(p.id)),
                "text": payload.get("text", ""),
                "metadata": {
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                },
                "score": score,
            })
        return output

    def delete_all(self) -> None:
        if self._client is None:
            return
        try:
            self._client.delete_collection(collection_name=self._collection_name)
            logger.info("Deleted Qdrant collection: %s", self._collection_name)
        except Exception:
            pass
        # Re-create immediately so store remains usable
        # Collection will be auto-created on next upsert if dim is unknown
        self._dim = None

    def count(self) -> int:
        if self._client is None:
            return 0
        try:
            return self._client.count(collection_name=self._collection_name).count
        except Exception:
            return 0

    def close(self) -> None:
        if self._client:
            self._client.close()
        self._client = None
        logger.info("QdrantVectorStore closed")
