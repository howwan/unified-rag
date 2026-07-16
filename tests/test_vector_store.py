"""Unit tests for vector store backends."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.vector_store import get_vector_store

SAMPLE_EMBEDDING_1 = [0.1] * 64
SAMPLE_EMBEDDING_2 = [0.2] * 64


def _make_store(store_type: str):
    config = get_config()
    config["VECTOR_STORE"] = store_type
    store = get_vector_store(config)
    store.initialize(probe_dim=64)
    return store


def test_pgvector_crud():
    store = _make_store("pgvector")
    store.delete_all()
    store.initialize(probe_dim=64)
    assert store.count() == 0

    store.upsert_chunks(
        ids=["doc1__0", "doc1__1"],
        texts=["hello world", "foo bar"],
        embeddings=[SAMPLE_EMBEDDING_1, SAMPLE_EMBEDDING_2],
        metadatas=[
            {"title": "Doc1", "source": "/tmp/doc1.md", "chunk_index": 0},
            {"title": "Doc1", "source": "/tmp/doc1.md", "chunk_index": 1},
        ],
    )
    assert store.count() == 2

    results = store.search(SAMPLE_EMBEDDING_1, top_k=2)
    assert len(results) == 2
    assert all("score" in r for r in results)

    store.delete_all()
    assert store.count() == 0
    store.close()
    print("  ✅ pgvector CRUD passed")


def test_chromadb_crud():
    store = _make_store("chromadb")
    store.delete_all()
    assert store.count() == 0

    store.upsert_chunks(
        ids=["doc1__0", "doc1__1"],
        texts=["hello world", "foo bar"],
        embeddings=[SAMPLE_EMBEDDING_1, SAMPLE_EMBEDDING_2],
        metadatas=[
            {"title": "Doc1", "source": "/tmp/doc1.md", "chunk_index": 0},
            {"title": "Doc1", "source": "/tmp/doc1.md", "chunk_index": 1},
        ],
    )
    assert store.count() == 2

    results = store.search(SAMPLE_EMBEDDING_1, top_k=2)
    assert len(results) == 2
    assert all("score" in r for r in results)

    store.delete_all()
    assert store.count() == 0
    store.close()
    print("  ✅ chromadb CRUD passed")


def test_qdrant_crud():
    store = _make_store("qdrant")
    store.delete_all()
    store.initialize(probe_dim=64)
    assert store.count() == 0

    store.upsert_chunks(
        ids=["doc1__0", "doc1__1"],
        texts=["hello world", "foo bar"],
        embeddings=[SAMPLE_EMBEDDING_1, SAMPLE_EMBEDDING_2],
        metadatas=[
            {"title": "Doc1", "source": "/tmp/doc1.md", "chunk_index": 0},
            {"title": "Doc1", "source": "/tmp/doc1.md", "chunk_index": 1},
        ],
    )
    assert store.count() == 2

    results = store.search(SAMPLE_EMBEDDING_1, top_k=2)
    assert len(results) == 2
    assert all("score" in r for r in results)

    store.delete_all()
    assert store.count() == 0
    store.close()
    print("  ✅ qdrant CRUD passed")


def run():
    print("\n=== test_vector_store ===")
    test_pgvector_crud()
    test_chromadb_crud()
    test_qdrant_crud()
    print("=== test_vector_store passed ===\n")


if __name__ == "__main__":
    run()
