# ArXiv RAG Assistant

A hybrid Retrieval-Augmented Generation (RAG) system for ArXiv research papers. Combines **dense retrieval** (sentence-transformer embeddings + Chroma), **lexical retrieval** (BM25), and **cross-encoder reranking** to provide accurate, citation-backed answers to research questions.

## 🚀 Key Features

*   **Hybrid Retrieval**: Dense (semantic) + BM25 (keyword) fusion for comprehensive coverage.
*   **Cross-Encoder Reranking**: Re-scores results for maximum precision.
*   **GPU Accelerated**: Embedding and reranking on CUDA.
*   **Production Ready**: FASTAPI backend with Groq LLM integration.
*   **Modern UI**: Clean, glassmorphic frontend for research.

## 🏗️ Architecture

```
Query → [Embedding] → Dense Retrieval (Chroma)
                     ↘
                      Merge + Score Fusion (α·dense + β·BM25)
                     ↗                    ↓
Query → [Tokenize]  → BM25 Retrieval    Cross-Encoder Rerank
                                          ↓
                                    Top-N Passages → Answer Generation
```

## 🛠️ Quick Start

### 1. Install
```bash
conda activate pytorch
pip install -r requirements.txt
```

### 2. Ingest Data
```bash
# Fetch 5000 papers (cs.AI, cs.LG) & chunk them
python ingest/ingest_arxiv.py --max-papers 5000
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

## 💻 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | **Llama 3 70B (Groq)** | SOTA open-source model, <1s inference. |
| **Vector DB** | **ChromaDB** | Semantic search engine. |
| **Reranker** | **Cross-Encoder** | `ms-marco-MiniLM` for high precision. |
| **Backend** | **FastAPI** | High-performance Python API. |
| **Frontend** | **Vanilla JS/CSS** | Lightweight, responsive UI. |

## 📊 System Metrics & Performance

> **Note:** Ranking metrics (Recall@k, MRR, nDCG@k) are omitted because ground-truth labels were auto-generated from the retriever itself (self-consistency). Only honest, reproducible system metrics are reported below. Human-labeled evaluation is planned. See [`results/metrics.md`](results/metrics.md) for the full report.

### ⚡ Latency & Throughput (5,000 papers)

| Component | p50 latency | p95 latency | Avg latency | QPS |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval Pipeline** | 79.6 ms | 96.9 ms | 90.7 ms | 11.02 |
| **End-to-End (incl. LLM)** | - | - | 1,059 ms | - |

**Caching Rule:** If `retrieval_p95 ≥ 200 ms`, system enables Redis top-k cache with 1-minute TTL. Currently at **96.9 ms**, so caching is gracefully bypassed.

### 🗄️ Storage & Indexing

| Component | Size / Count | Details |
| :--- | :--- | :--- |
| **Chroma DB (Dense)** | 82.93 MB | 5,975 vectors (384-dim) |
| **BM25 Index (Sparse)** | 7.09 MB | Custom lexical index |
| **SQLite DB** | 9.58 MB | Document metadata |
| **Total Papers Indexed** | 5,000 | From cs.AI and cs.LG |

### 🧠 Models & Costs

| Component | Model | Cost / Query | Tokens |
| :--- | :--- | :--- | :--- |
| **Embedder** | `all-MiniLM-L6-v2` | Local execution | - |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local execution | - |
| **LLM Inference** | `llama-3.3-70b-versatile` | Free (Groq Tier) | ~500 in, ~300 out |

