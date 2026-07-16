"""Integration tests for full RAG pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.vector_store import get_vector_store
from src.embedding import EmbeddingClient
from src.rag_engine import RAGEngine


def _make_engine(store_type: str):
    config = get_config()
    config["VECTOR_STORE"] = store_type
    store = get_vector_store(config)
    embedder = EmbeddingClient(config)
    engine = RAGEngine(
        store,
        embedder,
        chunk_size=128,
        chunk_overlap=10,
        top_k=2,
    )
    engine.initialize()
    return engine


def _create_test_docs():
    return [
        {
            "title": "Test Doc A",
            "content": "This is a sample document about machine learning. " * 10,
            "source": "/tmp/test_a.md",
        },
        {
            "title": "Test Doc B",
            "content": "Another document discussing vector databases and embeddings. " * 10,
            "source": "/tmp/test_b.md",
        },
    ]


def test_pipeline_pgvector():
    engine = _make_engine("pgvector")
    engine.reset()
    assert engine.get_stats()["total_chunks"] == 0

    docs = _create_test_docs()
    engine.index_documents(docs)
    assert engine.get_stats()["total_chunks"] > 0

    engine.reset()
    assert engine.get_stats()["total_chunks"] == 0
    engine._store.close()
    print("  ✅ pgvector pipeline passed")


def test_pipeline_chromadb():
    engine = _make_engine("chromadb")
    engine.reset()
    assert engine.get_stats()["total_chunks"] == 0

    docs = _create_test_docs()
    engine.index_documents(docs)
    assert engine.get_stats()["total_chunks"] > 0

    engine.reset()
    assert engine.get_stats()["total_chunks"] == 0
    engine._store.close()
    print("  ✅ chromadb pipeline passed")


def test_pipeline_qdrant():
    engine = _make_engine("qdrant")
    engine.reset()
    assert engine.get_stats()["total_chunks"] == 0

    docs = _create_test_docs()
    engine.index_documents(docs)
    assert engine.get_stats()["total_chunks"] > 0

    engine.reset()
    assert engine.get_stats()["total_chunks"] == 0
    engine._store.close()
    print("  ✅ qdrant pipeline passed")


def run():
    print("\n=== test_rag_pipeline ===")
    test_pipeline_pgvector()
    test_pipeline_chromadb()
    test_pipeline_qdrant()
    print("=== test_rag_pipeline passed ===\n")


if __name__ == "__main__":
    run()
