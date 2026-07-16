# unified-rag
一个支持**多后端向量存储**的 RAG（检索增强生成）演示项目。通过统一的抽象层，可在 **pgvector（生产）**、**ChromaDB（本地开发）**、**Qdrant（高性能检索）** 之间无缝切换，所有后端共享同一套 Embedding、文本切分、LLM 生成和 CLI 交互逻辑。
