# ArXiv RAG Assistant

A hybrid Retrieval-Augmented Generation (RAG) system for ArXiv research papers. Combines **dense retrieval** (sentence-transformer embeddings + Chroma), **lexical retrieval** (BM25), and **cross-encoder reranking** to provide accurate, citation-backed answers to research questions.

## 🚀 Key Features

*   **Hybrid Retrieval**: Dense (semantic) + BM25 (keyword) fusion for comprehensive coverage.
*   **Cross-Encoder Reranking**: Re-scores results for maximum precision.
*   **Advanced Metadata Filtering**: Filter by category, author, and publication year.
*   **Similar Papers**: Discover related papers via embedding-based nearest-neighbor search.
*   **Recency Boosting**: Time-decay weighting gives newer research a slight advantage.
*   **Context Compression**: MMR-based sentence selection reduces LLM token usage.
*   **In-Memory Caching**: LRU query cache with 5-minute TTL for instant repeated queries.
*   **Analytics Extraction**: Top authors and categories surfaced per query.
*   **ONNX-First Inference**: FastEmbed + FlashRank avoid heavy PyTorch model loading at runtime.
*   **Memory-Aware Runtime Controls**: BM25 and reranker can be enabled, disabled, or auto-gated by memory limits.
*   **Production Ready**: FastAPI backend with Groq LLM integration.
*   **Academic UI**: Clean, serif-accented frontend designed for research.

## 🏗️ Architecture

```
Query → [Embedding] → Dense Retrieval (Chroma)
                     ↘
                      Merge + Score Fusion (α·dense + β·BM25) + Recency Boost
                     ↗                    ↓
Query → [Tokenize]  → BM25 Retrieval    Cross-Encoder Rerank
                                          ↓
                                    MMR Context Compression → Answer Generation
```

## 🛠️ Quick Start

### 1. Install
```bash
conda activate pytorch
pip install -r requirements.txt
```

### 2. Ingest Data
```bash
# Fetch 20000 papers (cs.AI, cs.LG) & chunk them
python ingest/ingest_arxiv.py --max-papers 20000
python ingest/chunking.py
```

### 3. Build Indexes
```bash
# Build Vector & Keyword Indexes
python index/build_chroma.py
python index/build_bm25.py
```

### 4. Run App
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)**

### 5. Retrieval Runtime Flags (Optional)

Use these environment variables to control memory/performance behavior at startup:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ENABLE_BM25` | auto | Force-enable/disable BM25 (`true`/`false`) |
| `BM25_MIN_MEMORY_MB` | `640` | Auto-disable BM25 when container limit is at or below this value |
| `ENABLE_RERANKER` | auto | Force-enable/disable reranker (`true`/`false`) |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | FlashRank model name |
| `RERANKER_LAZY_LOAD` | `true` | Load reranker model on first use instead of startup |
| `RERANKER_MIN_MEMORY_MB` | `768` | Auto-disable reranker when container limit is at or below this value |

#### Deployment Profiles

**Full retrieval mode (recommended on >= 4GB RAM):**

```bash
ENABLE_BM25=true
ENABLE_RERANKER=true
RERANKER_LAZY_LOAD=true
RERANKER_MODEL=ms-marco-MiniLM-L-6-v2
```

**Low-memory mode (512MB class instances):**

```bash
ENABLE_BM25=false
ENABLE_RERANKER=false
```

When BM25 is disabled, retrieval runs in dense-only mode and still supports metadata filters.

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/query` | Hybrid retrieval + rerank + LLM answer (supports `category`, `author`, `start_year` filters) |
| `GET` | `/paper/{id}` | Paper metadata lookup |
| `GET` | `/paper/{id}/similar` | Find similar papers by embedding similarity |
| `GET` | `/health` | Health check with index stats and cache metrics |

## 💻 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | **Llama 3 70B (Groq)** | SOTA open-source model, <1s inference. |
| **Vector DB** | **ChromaDB** | Semantic search engine. |
| **Embedder** | **FastEmbed (ONNX)** | `all-MiniLM-L6-v2` for lightweight local embeddings. |
| **Reranker** | **FlashRank (ONNX)** | `ms-marco-MiniLM-L-6-v2` reranking with lazy-load support. |
| **Backend** | **FastAPI** | High-performance Python API. |
| **Frontend** | **Vanilla JS/CSS** | Lightweight, academic-style UI. |

## 📊 System Metrics & Performance

> **Note:** Ranking metrics (Recall@k, MRR, nDCG@k) are omitted because ground-truth labels were auto-generated from the retriever itself (self-consistency). Only honest, reproducible system metrics are reported below. Human-labeled evaluation is planned. See [`results/metrics.md`](results/metrics.md) for the full report.

### 🗄️ Storage & Indexing (20,000 papers)

| Component | Size / Count | Details |
| :--- | :--- | :--- |
| **Chroma DB (Dense)** | 393.25 MiB | 24,992 vectors (384-dim) |
| **BM25 Index (Sparse)** | 28.92 MiB | Custom lexical index |
| **SQLite DB** | 38.93 MiB | Document metadata |
| **Embeddings Backup** | 36.61 MiB | NumPy `.npy` file |
| **Total Papers Indexed** | 20,000 | From cs.AI and cs.LG |
| **Total Chunks** | 24,992 | Chunk size 300, overlap 20% (avg tokens: 220.47, median: 240) |

**Cache Config:** In-memory LRU cache with `max_size=128` and `ttl_seconds=300` (5 minutes).

### 🧠 Models & Costs

| Component | Model | Cost / Query | Tokens |
| :--- | :--- | :--- | :--- |
| **Embedder** | `all-MiniLM-L6-v2` (FastEmbed ONNX) | Local execution | - |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` (FlashRank ONNX) | Local execution | - |
| **LLM Inference** | `llama-3.3-70b-versatile` | Free (Groq Tier) | ~500 in, ~300 out |
