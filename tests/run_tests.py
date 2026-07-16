"""Unified test runner with auto data generation."""

import sys
import tempfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.vector_store import get_vector_store
from src.embedding import EmbeddingClient
from src.rag_engine import RAGEngine
from src.chunking import read_markdown_files


def _ensure_test_data():
    """If vector store is empty, generate synthetic test docs and index them."""
    config = get_config()
    for store_type in ("pgvector", "chromadb", "qdrant"):
        config["VECTOR_STORE"] = store_type
        store = get_vector_store(config)
        embedder = EmbeddingClient(config)
        engine = RAGEngine(store, embedder, chunk_size=128, chunk_overlap=10, top_k=2)
        try:
            engine.initialize()
        except Exception as e:
            print(f"  ⚠ Could not initialize {store_type}: {e}")
            store.close()
            continue

        count = engine.get_stats()["total_chunks"]
        if count == 0:
            print(f"  📦 {store_type} is empty, generating test data...")
            with tempfile.TemporaryDirectory() as tmpdir:
                for title in ["Alpha", "Beta", "Gamma"]:
                    path = os.path.join(tmpdir, f"{title.lower()}.md")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"# {title}\n\n")
                        f.write(f"This document explains {title} concepts in detail. " * 20)

                docs = read_markdown_files(tmpdir)
                engine.index_documents(docs)

            count = engine.get_stats()["total_chunks"]
            print(f"  ✅ {store_type} now has {count} chunks")
        else:
            print(f"  ✅ {store_type} already has {count} chunks")

        store.close()


def run():
    print("\n==============================")
    print(" Unified RAG Demo Test Runner ")
    print("==============================\n")

    print("Step 1: Ensuring test data...")
    _ensure_test_data()

    print("\nStep 2: Running unit tests...")
    from tests import test_vector_store
    test_vector_store.run()

    print("Step 3: Running integration tests...")
    from tests import test_rag_pipeline
    test_rag_pipeline.run()

    print("==============================")
    print(" All tests passed!")
    print("==============================\n")


if __name__ == "__main__":
    run()
