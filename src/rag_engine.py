"""RAG engine: indexing and query pipeline."""

import logging
from typing import Any

from src.embedding import EmbeddingClient
from src.chunking import read_markdown_files, chunk_text
from src.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """Orchestrates embedding, storage, retrieval, and generation."""

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: EmbeddingClient,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._initialized = False

    def initialize(self) -> None:
        """Probe embedding dim and initialize store."""
        dim = self._embedder.probe_dim()
        self._dim = dim
        self._store.initialize(probe_dim=dim)
        self._initialized = True
        logger.info("RAGEngine initialized")

    def index_documents(self, docs: list[dict]) -> None:
        """Chunk, embed, and upsert documents into the store."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        total_chunks = 0
        for doc in docs:
            chunks = chunk_text(doc["content"], self._chunk_size, self._chunk_overlap)
            if not chunks:
                continue

            logger.info("  📄 %s: %s chunks", doc["title"], len(chunks))

            batch_size = 64
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                all_embeddings.extend(self._embedder.embed_batch(batch))

            ids = [f"{doc['source']}__{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "title": doc["title"],
                    "source": doc["source"],
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]

            self._store.upsert_chunks(ids, chunks, all_embeddings, metadatas)
            total_chunks += len(chunks)

        logger.info("Indexed %s chunks from %s documents", total_chunks, len(docs))

    def query(self, question: str, llm_config: dict | None = None) -> str:
        """Full RAG pipeline: retrieve chunks and call LLM."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if not self._embedder.available:
            logger.warning("Embedding service unavailable; returning empty result")
            return ""

        query_emb = self._embedder.embed(question)
        results = self._store.search(query_emb, top_k=self._top_k)

        print(f"\n{'─' * 60}")
        print(f"🔍 Question: {question}")
        print(f"{'─' * 60}")
        print(f"📚 Retrieved {len(results)} chunks:")
        for i, r in enumerate(results, 1):
            print(
                f"  [{i}] {r['metadata']['title']} "
                f"(chunk #{r['metadata']['chunk_index']}, score: {r['score']:.4f})"
            )

        context_parts = []
        for r in results:
            snippet = r["text"][:500]
            context_parts.append(
                f"### {r['metadata']['title']}\n{snippet}\n(source: {r['metadata']['source']})"
            )
        context = "\n\n".join(context_parts)

        answer = self._call_llm(context, question, llm_config)
        print(f"\n💬 Answer:\n{answer}")
        return answer

    def _call_llm(
        self, context: str, question: str, llm_config: dict | None = None
    ) -> str:
        import httpx
        from openai import OpenAI

        cfg = llm_config or {}
        base_url = cfg.get("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = cfg.get("LLM_API_KEY", "")
        model = cfg.get("LLM_MODEL", "gpt-5.2")
        verify_ssl = cfg.get("LLM_VERIFY_SSL", True)

        http_client = httpx.Client(
            headers={"X-Api-Key": api_key},
            verify=verify_ssl,
        )
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant. Answer the user's question based on "
                    "the provided reference materials. If the materials are insufficient, say so. "
                    "Cite sources when possible. Answer in the same language as the question."
                ),
            },
            {
                "role": "user",
                "content": f"Reference materials:\n{context}\n\nQuestion: {question}",
            },
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM API error: %s", e)
            return ""

    def get_stats(self) -> dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        count = self._store.count()
        return {
            "total_chunks": count,
        }

    def reset(self) -> None:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        self._store.delete_all()
        self._store.initialize(probe_dim=self._dim)
        logger.info("Store reset complete")
