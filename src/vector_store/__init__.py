"""Vector store factory."""

from .base import BaseVectorStore
from .pgvector import PgVectorStore
from .chroma import ChromaVectorStore
from .qdrant import QdrantVectorStore


def get_vector_store(config: dict) -> BaseVectorStore:
    """Return a vector store instance based on config['VECTOR_STORE']."""
    vtype = config.get("VECTOR_STORE", "auto").lower()

    if vtype == "auto":
        if config.get("DATABASE_URL", "").startswith("postgresql") or config.get(
            "DB_CONFIG", {}
        ).get("host"):
            return PgVectorStore(config)
        return ChromaVectorStore(config)
    elif vtype == "pgvector":
        return PgVectorStore(config)
    elif vtype == "chromadb":
        return ChromaVectorStore(config)
    elif vtype == "qdrant":
        return QdrantVectorStore(config)
    else:
        raise ValueError(f"Unknown VECTOR_STORE: {vtype}")
