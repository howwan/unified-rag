"""CLI dispatcher for unified RAG demo."""

import argparse
import sys

from src.config import get_config
from src.vector_store import get_vector_store
from src.embedding import EmbeddingClient
from src.rag_engine import RAGEngine
from src.chunking import read_markdown_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified-rag-demo",
        description="Unified RAG demo with pgvector / ChromaDB / Qdrant backends",
    )
    parser.add_argument(
        "--store",
        choices=["auto", "pgvector", "chromadb", "qdrant"],
        default=None,
        help="Override VECTOR_STORE env variable",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Index markdown files from directory")
    idx.add_argument("directory", help="Path to directory containing .md files")

    query_parser = sub.add_parser("query", help="One-shot RAG query")
    query_parser.add_argument("question", nargs="+", help="Question text")

    sub.add_parser("chat", help="Interactive RAG chat")
    sub.add_parser("stats", help="Show knowledge base stats")
    sub.add_parser("reset", help="Clear all indexed data")

    return parser


def run() -> None:
    import sys

    # Pre-parse --store so it works before OR after the subcommand
    store_override = None
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--store" and i + 1 < len(sys.argv):
            store_override = sys.argv[i + 1]
            # Remove both --store and its value so argparse doesn't complain
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
        elif sys.argv[i].startswith("--store="):
            store_override = sys.argv[i].split("=", 1)[1]
            sys.argv.pop(i)
            break
        i += 1

    parser = build_parser()
    args = parser.parse_args()

    config = get_config()
    if store_override:
        config["VECTOR_STORE"] = store_override
    elif args.store:
        config["VECTOR_STORE"] = args.store

    store = get_vector_store(config)
    embedder = EmbeddingClient(config)
    engine = RAGEngine(
        store,
        embedder,
        chunk_size=config["RAG_CHUNK_SIZE"],
        chunk_overlap=config["RAG_CHUNK_OVERLAP"],
        top_k=config["RAG_TOP_K"],
    )

    engine.initialize()

    if args.command == "index":
        directory = args.directory
        print(f"\n📂 Scanning markdown files in: {directory}")
        docs = read_markdown_files(directory)
        if not docs:
            print("No markdown files found.")
            sys.exit(0)

        print(f"Found {len(docs)} markdown files:\n")
        for d in docs:
            chars = len(d["content"])
            print(f"  • {d['title']} ({chars:,} chars) — {d['source']}")
        print()

        engine.index_documents(docs)

    elif args.command == "query":
        question = " ".join(args.question)
        engine.query(question, llm_config=config)

    elif args.command == "chat":
        print("\n🤖 Interactive RAG chat (type 'quit' to exit)\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not question or question.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if question.lower() == "stats":
                s = engine.get_stats()
                print(f"  📊 {s['total_chunks']} chunks")
                continue

            engine.query(question, llm_config=config)
            print()

    elif args.command == "stats":
        s = engine.get_stats()
        print(f"\n📊 Knowledge Base Stats:")
        print(f"   Total chunks: {s['total_chunks']}")

    elif args.command == "reset":
        engine.reset()
        print("🗑️  All indexed data cleared. Run 'index' to re-index.")

    store.close()


if __name__ == "__main__":
    run()
