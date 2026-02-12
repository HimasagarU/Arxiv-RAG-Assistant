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
