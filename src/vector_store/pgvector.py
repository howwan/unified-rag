"""PgVector backend implementation."""

import logging
from typing import Any

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector vector store."""

    def __init__(self, config: dict) -> None:
        self._db_cfg = config["DB_CONFIG"]
        self._conn = None
        self._dim: int | None = None

    def initialize(self, probe_dim: int | None = None) -> None:
        import psycopg2

        self._conn = psycopg2.connect(**self._db_cfg)
        logger.info(
            "PgVectorStore connected: %s:%s/%s",
            self._db_cfg["host"],
            self._db_cfg["port"],
            self._db_cfg["dbname"],
        )

        if probe_dim:
            self._dim = probe_dim
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        if self._conn is None or self._dim is None:
            raise RuntimeError("Store not initialized with dimension")

        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'rag_documents'
                );
            """)
            table_exists = cur.fetchone()[0]

            if not table_exists:
                cur.execute(f"""
                    CREATE TABLE rag_documents (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT DEFAULT '',
                        chunk_index INTEGER DEFAULT 0,
                        embedding vector({self._dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute(f"""
                    CREATE INDEX idx_rag_docs_embedding
                    ON rag_documents
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                logger.info("Created rag_documents with vector(%s)", self._dim)
            else:
                cur.execute("""
                    SELECT atttypmod FROM pg_attribute
                    WHERE attrelid = 'rag_documents'::regclass
                      AND attname = 'embedding';
                """)
                row = cur.fetchone()
                existing_dim = row[0] if row and row[0] > 0 else None
                if existing_dim and existing_dim != self._dim:
                    logger.warning(
                        "Existing embedding column is vector(%s), but current model "
                        "produces %s-dim vectors. Run 'reset' and re-index if you "
                        "switched models.",
                        existing_dim,
                        self._dim,
                    )
                else:
                    logger.info("Table rag_documents exists (vector dim: %s)", existing_dim)

            self._conn.commit()

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        from psycopg2.extras import execute_values

        if self._conn is None:
            raise RuntimeError("Store not initialized")

        rows = []
        for i, text in enumerate(texts):
            meta = metadatas[i]
            rows.append((
                meta.get("title", ""),
                text,
                meta.get("source", ""),
                meta.get("chunk_index", 0),
                embeddings[i],
            ))

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO rag_documents (title, content, source, chunk_index, embedding)
                   VALUES %s""",
                rows,
                template="(%s, %s, %s, %s, %s::vector)",
            )
        self._conn.commit()

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("Store not initialized")

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, content, source, chunk_index,
                       1 - (embedding <=> %s::vector) AS score
                FROM rag_documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            columns = [desc[0] for desc in cur.description]
            results = []
            for row in cur.fetchall():
                d = dict(zip(columns, row))
                results.append({
                    "id": str(d["id"]),
                    "text": d["content"],
                    "metadata": {
                        "title": d["title"],
                        "source": d["source"],
                        "chunk_index": d["chunk_index"],
                    },
                    "score": float(d["score"]),
                })
            return results

    def delete_all(self) -> None:
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS rag_documents;")
                self._conn.commit()
            logger.info("Dropped rag_documents table")
        except Exception:
            self._conn.rollback()
            pass

    def count(self) -> int:
        if self._conn is None:
            return 0
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rag_documents;")
                return cur.fetchone()[0]
        except Exception:
            self._conn.rollback()
            return 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("PgVectorStore connection closed")
