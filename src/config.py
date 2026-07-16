"""Configuration loader from .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _parse_database_url(db_url: str) -> dict:
    from urllib.parse import urlparse
    p = urlparse(db_url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 5432,
        "dbname": p.path.lstrip("/") or "rag_db",
        "user": p.username or "postgres",
        "password": p.password or "postgres",
    }


def get_config() -> dict:
    """Return unified configuration dictionary."""
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgresql"):
        db_cfg = _parse_database_url(db_url)
    else:
        db_cfg = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "rag_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASS", "postgres"),
        }

    return {
        "VECTOR_STORE": os.getenv("VECTOR_STORE", "auto").lower(),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "data/chroma"),
        "DB_CONFIG": db_cfg,
        "DATABASE_URL": db_url,
        "EMBEDDING_BASE_URL": os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "EMBEDDING_API_KEY": os.getenv("EMBEDDING_API_KEY")
        or os.getenv("LLM_API_KEY", ""),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL")
        or os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small"),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-5.2"),
        "LLM_VERIFY_SSL": os.getenv("LLM_VERIFY_SSL", "true").lower()
        not in ("false", "0", "no"),
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "rag_documents"),
        "RAG_CHUNK_SIZE": int(os.getenv("RAG_CHUNK_SIZE", "512")),
        "RAG_CHUNK_OVERLAP": int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        "RAG_TOP_K": int(os.getenv("RAG_TOP_K", "5")),
    }
