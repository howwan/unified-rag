# Unified RAG

一个支持**多后端向量存储**的 RAG（检索增强生成）项目。通过统一的抽象层，可在 **pgvector（生产）**、**ChromaDB（本地开发）**、**Qdrant（高性能检索）** 之间无缝切换，所有后端共享同一套 Embedding、文本切分、LLM 生成和 CLI 交互逻辑。

---

## 一、软件功能

| 功能模块 | 说明 |
|---------|------|
| **多后端向量存储** | 支持 pgvector / ChromaDB / Qdrant 三种后端，通过环境变量或 CLI 参数一键切换 |
| **Markdown 知识库导入** | 递归读取目录下的 `.md` 文件，自动提取标题、切分文本、生成向量并入库 |
| **向量相似度检索** | 基于 Cosine Similarity 的 HNSW 索引检索，返回 Top-K 相关文本块及相似度分数 |
| **LLM 问答生成** | 使用 OpenAI-compatible API，自动拼接检索上下文，生成带引用来源的回答 |
| **交互式 Chat** | 支持持续多轮对话，内置 `stats` / `quit` 命令 |
| **运行时维度探测** | 自动检测 Embedding 模型输出维度，动态适配各后端的表/Collection 结构 |
| **自动化测试** | 三后端各自的 CRUD 单元测试 + 完整 RAG 流程集成测试，自动检测空库并生成测试数据 |
| **统一日志管理** | 自动写入 `logs/rag.log`（轮转 5MB×3 份），支持 `LOG_LEVEL` 环境变量控制 |
| **Docker 一键启动** | `docker-compose.yaml` 同时拉起 PostgreSQL+pgvector 和 Qdrant 服务 |

---

## 二、快速开始

### 2.1 启动依赖服务

```bash
cd unified-rag
docker compose up -d
```

启动后端口映射：
- **PostgreSQL + pgvector**: `localhost:5432`
- **pgweb（数据库管理 UI）**: `http://localhost:8081`
- **Qdrant**: `http://localhost:6333`
- **Qdrant（数据库管理 UI）**: `http://localhost:6333/dashboard#/welcome`

### 2.2 安装 Python 依赖

```bash
pip install -r requirements.txt
```

依赖清单：`psycopg2-binary`, `openai`, `python-dotenv`, `chromadb`, `qdrant-client`

### 2.3 配置环境变量

复制 `.env` 备份为 `.env.local` ，然后根据实际情况修改`.env`：

```bash
cp .env .env.local
```

关键配置项：

```env
# 后端切换：auto | pgvector | chromadb | qdrant
VECTOR_STORE=auto

# PostgreSQL（pgvector 用）
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=postgres
DB_PASS=postgres

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=rag_documents

# Embedding 服务（OpenAI-compatible）
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=qwen3-embedding:0.6b

# LLM 服务（OpenAI-compatible）
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=your-key
LLM_MODEL=gpt-5.2

# RAG 参数
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=5
```

### 2.4 导入知识库并查询

```bash
# 1. 导入 Markdown 文件
python run.py index "/path/to/your/knowledge/base"

# 2. 单次查询
python run.py query "什么是楼板模式"

# 3. 查看统计
python run.py stats

# 4. 交互式对话
python run.py chat

# 5. 清空数据
python run.py reset
```

### 2.5 后端切换示例

```bash
# 方式一：环境变量（推荐用于脚本/CI）
VECTOR_STORE=pgvector   python run.py stats
VECTOR_STORE=chromadb   python run.py stats
VECTOR_STORE=qdrant     python run.py stats

# 方式二：CLI 参数（--store 放在子命令前或后均可）
python run.py --store qdrant stats
python run.py stats --store qdrant
```

---

## 三、代码结构

```
unified-rag/
├── docker-compose.yaml       # Docker 编排：pgvector + pgweb + Qdrant
├── .env                      # 环境变量配置
├── requirements.txt          # Python 依赖
├── run.py                    # 统一入口脚本
├── src/                      # 源代码
│   ├── config.py             # 环境变量读取与配置字典组装
│   ├── logging_config.py     # 统一日志配置（控制台 + 轮转文件）
│   ├── embedding.py          # OpenAI-compatible Embedding 客户端（批量/探测/降级）
│   ├── chunking.py           # Markdown 读取 + 句子感知文本切分
│   ├── rag_engine.py         # RAG 引擎：index → embed → store → search → LLM
│   ├── cli.py                # argparse CLI：index/query/chat/stats/reset
│   └── vector_store/         # 向量存储抽象层
│       ├── base.py           # BaseVectorStore 抽象基类
│       ├── __init__.py       # get_vector_store(config) 工厂函数
│       ├── pgvector.py       # PostgreSQL + pgvector 实现
│       ├── chroma.py         # ChromaDB 本地持久化实现
│       └── qdrant.py         # Qdrant 远程/本地实现
└── tests/                    # 测试
    ├── test_vector_store.py  # 三后端 CRUD 单元测试
    ├── test_rag_pipeline.py  # 三后端完整流程集成测试
    └── run_tests.py          # 统一测试入口（自动数据准备 + 执行 + 报告）
```

---

## 四、核心设计：代码复用与可扩展性

### 4.1 抽象基类 `BaseVectorStore`

所有后端必须实现同一套契约，上层业务代码完全不感知底层差异：

```python
class BaseVectorStore(ABC):
    @abstractmethod
    def initialize(self, probe_dim: int | None = None) -> None: ...
    @abstractmethod
    def upsert_chunks(self, ids, texts, embeddings, metadatas) -> None: ...
    @abstractmethod
    def search(self, query_embedding, top_k=5) -> list[dict]: ...
    @abstractmethod
    def delete_all(self) -> None: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def close(self) -> None: ...
```

**复用收益**：`RAGEngine`、`CLI`、`测试框架` 均面向 `BaseVectorStore` 编程，新增后端时**零改动**上层代码。

### 4.2 工厂函数 `get_vector_store(config)`

```python
def get_vector_store(config: dict) -> BaseVectorStore:
    vtype = config["VECTOR_STORE"].lower()
    if vtype == "auto":
        # 有 PostgreSQL 配置 → pgvector，否则 → chromadb
        ...
    elif vtype == "pgvector":  return PgVectorStore(config)
    elif vtype == "chromadb":  return ChromaVectorStore(config)
    elif vtype == "qdrant":    return QdrantVectorStore(config)
```

**复用收益**：配置与实例化解耦，`RAGEngine` 无需关心具体后端类型。

### 4.3 RAG 引擎 `RAGEngine`

```
用户调用                    RAGEngine 内部流程
─────────────────────────────────────────────────────────
index_documents(docs)  →  chunk_text() → embed_batch() → store.upsert_chunks()
query(question)        →  embed() → store.search() → _call_llm()
reset()                →  store.delete_all() → store.initialize(probe_dim)
```

**复用收益**：无论后端是 pgvector / ChromaDB / Qdrant，`RAGEngine` 的 `index_documents`、`query`、`reset` 逻辑完全一致。

### 4.4 如何添加新后端（以 Milvus 为例）

只需 **3 步**，无需修改任何上层代码：

1. **新建实现文件**：`src/vector_store/milvus.py`

```python
from .base import BaseVectorStore

class MilvusVectorStore(BaseVectorStore):
    def initialize(self, probe_dim=None): ...
    def upsert_chunks(self, ids, texts, embeddings, metadatas): ...
    def search(self, query_embedding, top_k=5): ...
    def delete_all(self): ...
    def count(self): ...
    def close(self): ...
```

2. **注册工厂**：在 `src/vector_store/__init__.py` 中导入并添加分支：

```python
from .milvus import MilvusVectorStore
# ...
elif vtype == "milvus":
    return MilvusVectorStore(config)
```

3. **注册 CLI 参数**：在 `src/cli.py` 的 `--store` choices 中加入 `"milvus"`。

完成后即可通过 `VECTOR_STORE=milvus python run.py stats` 使用新后端。

### 4.5 三后端差异屏蔽细节

| 差异点 | pgvector | ChromaDB | Qdrant |
|--------|----------|----------|--------|
| **维度声明** | `CREATE TABLE ... vector({dim})` | 自动推断 | `VectorParams(size={dim})` |
| **距离计算** | `1 - (embedding <=> query::vector)` | `1.0 - distance` | `query_points()` 直接返回 similarity |
| **元数据类型** | JSONB / 独立字段 | 强制 `str()` | Payload 任意 dict |
| **ID 类型** | 字符串（自定义） | 字符串（自定义） | UUID/整数（需映射） |
| **批量写入** | `execute_values` | `collection.upsert` | `client.upsert(points)` |
| **schema 重建** | `DROP TABLE + CREATE` | `delete_collection + get_or_create` | `delete_collection + create` |

所有差异均被封装在各自的 `*VectorStore` 类中，对外暴露完全一致的接口。

---

## 五、测试

### 5.1 一键运行全部测试

```bash
python tests/run_tests.py
```

执行内容：
1. **自动数据准备**：检测三后端是否为空，若空则自动生成 3 个合成 `.md` 文件并索引
2. **单元测试**：各后端的 upsert → count → search → delete_all
3. **集成测试**：reset → index → stats → reset 完整流程

### 5.2 单独运行测试

```bash
python tests/test_vector_store.py   # CRUD 单元测试
python tests/test_rag_pipeline.py   # 集成测试
```

---

## 六、日志

所有模块统一使用 `logging.getLogger(__name__)`，日志配置集中管理于 `src/logging_config.py`。

### 6.1 日志输出

- **控制台**：所有 `INFO` 及以上级别日志实时输出到 stderr
- **文件**：自动写入 `logs/rag.log`，单文件 5MB 自动轮转，保留 3 个历史备份

### 6.2 查看日志

```bash
# 查看当前日志文件
cat logs/rag.log

# 实时跟踪
tail -f logs/rag.log

# 调试模式（输出更详细）
LOG_LEVEL=DEBUG python run.py query "什么是楼板模式"
```

### 6.3 日志格式示例

```
2026-07-14 16:28:47 [INFO] src.embedding: Detected embedding dimension: 1024 (model: qwen3-embedding:0.6b)
2026-07-14 16:28:49 [INFO] src.vector_store.chroma: ChromaVectorStore initialized: data/chroma
2026-07-14 16:28:49 [INFO] src.rag_engine: RAGEngine initialized
```

---

## 七、典型使用场景

| 场景 | 推荐后端 | 启动命令 |
|------|---------|---------|
| 生产环境，已有 PostgreSQL | **pgvector** | `VECTOR_STORE=pgvector python run.py index ./docs` |
| 本地开发，零依赖启动 | **ChromaDB** | `VECTOR_STORE=chromadb python run.py index ./docs` |
| 高并发检索，独立向量服务 | **Qdrant** | `VECTOR_STORE=qdrant python run.py index ./docs` |
| 自动化测试/CI | **ChromaDB** | 无外部服务依赖，纯本地文件 |

---

## 八、注意事项

1. **Embedding 维度切换**：若更换 Embedding 模型导致维度变化，请先执行 `python run.py reset` 清空旧数据，再重新索引。
2. **Qdrant ID 限制**：Qdrant 要求 point ID 为 UUID 或整数。项目内部使用 `uuid.uuid5` 将字符串 ID 转换为确定性 UUID，并在 payload 中保留原始 ID，检索时恢复。
3. **元数据一致性**：ChromaDB 要求 metadata value 必须为字符串，因此在 `ChromaVectorStore.upsert_chunks()` 中自动将所有值转为 `str()`。其他后端无此限制。
4. **auto 模式**：`VECTOR_STORE=auto` 时，若检测到 `DATABASE_URL` 以 `postgresql` 开头或 `DB_HOST` 存在，则优先选择 pgvector；否则选择 ChromaDB。Qdrant 需显式指定 `VECTOR_STORE=qdrant`。
5. **环境变量优先级**：Shell 传入的环境变量优先级高于 `.env` 文件。如果命令行指定 `VECTOR_STORE=chromadb` 无效，请检查是否错误地在 `config.py` 中使用了 `load_dotenv(override=True)`。
6. **`--store` 参数位置**：`--store` 可以放在子命令前或后，两种方式等价：`python run.py --store qdrant stats` == `python run.py stats --store qdrant`。

---

## 九、许可证

MIT License — 自由用于学习、研究和生产环境。
