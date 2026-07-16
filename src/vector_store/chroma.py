"""ChromaDB backend implementation."""

import logging
from pathlib import Path
from typing import Any

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Local ChromaDB persistent vector store."""

    def __init__(self, config: dict) -> None:
        self._persist_dir = config.get("CHROMA_PERSIST_DIR", "data/chroma")
        self._client = None
        self._collection = None

    def initialize(self, probe_dim: int | None = None) -> None:
        import chromadb

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaVectorStore initialized: %s", self._persist_dir)

        if probe_dim:
            count = self._collection.count()
            if count > 0:
                sample = self._collection.peek(limit=1)
                embeddings = sample.get("embeddings") if sample else None
                if embeddings is not None and len(embeddings) > 0:
                    existing_dim = len(embeddings[0])
                    if existing_dim != probe_dim:
                        logger.warning(
                            "Existing collection has %s-dim embeddings, but current "
                            "model produces %s-dim vectors. Run 'reset' and re-index "
                            "if you switched models.",
                            existing_dim,
                            probe_dim,
                        )
                    else:
                        logger.info(
                            "Collection dimension matches probe: %s", probe_dim
                        )

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if self._collection is None:
            raise RuntimeError("Store not initialized")

        str_metadatas = []
        for m in metadatas:
            str_metadatas.append({k: str(v) for k, v in m.items()})

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=str_metadatas,
        )

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if self._collection is None:
            raise RuntimeError("Store not initialized")

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        output = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            distance = distances[i] if distances else 0.0
            score = 1.0 - distance
            meta = metadatas[i] if metadatas else {}
            output.append({
                "id": str(doc_id),
                "text": documents[i] if documents else "",
                "metadata": {
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                },
                "score": score,
            })
        return output

    def delete_all(self) -> None:
        if self._client is None:
            return
        try:
            self._client.delete_collection(name="rag_documents")
            logger.info("Deleted Chroma collection: rag_documents")
        except Exception:
            pass
        self._collection = None
        # Re-create collection so the store remains usable
        self._collection = self._client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def close(self) -> None:
        self._client = None
        self._collection = None
        logger.info("ChromaVectorStore closed")
