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
*   **GPU Accelerated**: Embedding and reranking on CUDA.
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
| **Reranker** | **Cross-Encoder** | `ms-marco-MiniLM` for high precision. |
| **Backend** | **FastAPI** | High-performance Python API. |
| **Frontend** | **Vanilla JS/CSS** | Lightweight, academic-style UI. |

## 📊 System Metrics & Performance

> **Note:** Ranking metrics (Recall@k, MRR, nDCG@k) are omitted because ground-truth labels were auto-generated from the retriever itself (self-consistency). Only honest, reproducible system metrics are reported below. Human-labeled evaluation is planned. See [`results/metrics.md`](results/metrics.md) for the full report.

### 🗄️ Storage & Indexing (10,000 papers)

| Component | Size / Count | Details |
| :--- | :--- | :--- |
| **Chroma DB (Dense)** | ~171 MB | 12,165 vectors (384-dim) |
| **BM25 Index (Sparse)** | 14.2 MB | Custom lexical index |
| **SQLite DB** | 19.3 MB | Document metadata |
| **Embeddings Backup** | 17.8 MB | NumPy .npy file |
| **Total Papers Indexed** | 20,000 | From cs.AI and cs.LG |
| **Total Chunks** | 12,165 | ~300 tokens per chunk, 20% overlap |

**Caching Rule:** If `retrieval_p95 ≥ 200 ms`, system enables in-memory LRU cache with 5-minute TTL.

### 🧠 Models & Costs

| Component | Model | Cost / Query | Tokens |
| :--- | :--- | :--- | :--- |
| **Embedder** | `all-MiniLM-L6-v2` | Local execution | - |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local execution | - |
| **LLM Inference** | `llama-3.3-70b-versatile` | Free (Groq Tier) | ~500 in, ~300 out |
