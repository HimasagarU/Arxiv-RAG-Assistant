# ArXiv RAG Assistant — Mechanistic Interpretability

A production-grade **Retrieval-Augmented Generation (RAG)** system specialized for **mechanistic interpretability** research. The current pipeline is text-only: seed-driven corpus building, full-text PDF extraction, text chunking, dense + PostgreSQL FTS retrieval, reranking, and intent-aware answer generation.

## Architecture

```
Seed Papers (15 curated)
  → Citation Expansion (Semantic Scholar API)
  → Keyword Gap-Filling (arXiv API)
  → Timeline Balancing (early/middle/recent)
  → PostgreSQL (papers, chunks, citation_edges)
  → PDF Download + Full Text Extraction (PyMuPDF)
  → Full-Text Chunking
  → Qdrant Text Collection + PostgreSQL FTS
  → Hybrid Retrieval
  → Cross-Encoder Reranking
  → Intent-Aware LLM Generation (Groq)
```

## Key Features

### Seed-Driven Corpus Building
- **15 curated seed papers** spanning 2017–2024 (Attention Is All You Need → Scaling Monosemanticity)
- **Backward citation expansion** — pull in foundational/prerequisite papers via references
- **Forward citation expansion** — capture newer work that builds on seeds
- **Timeline balancing** — enforce coverage across early (15-25%), middle (40-50%), and recent (25-35%) eras
- **Layer tagging** — every paper tagged as `prerequisite`, `foundation`, `core`, or `latest`
- **Corpus target**: 500–1500 papers (quality over quantity)

### Text-Only Retrieval
- **Single Qdrant collection**: `arxiv_text`
- **Dense + PostgreSQL FTS hybrid search** over full-text chunks
- **Intent-aware reranking** with a cross-encoder
- **Layer-aware scoring** so foundational papers stay visible

### Full-Text Extraction
- **PDF download and parsing** with `PyMuPDF`
- **Full-text normalization** to remove line-wrap noise and references
- **Overlapping token chunking** for retrieval-ready text windows

### Intent-Aware Generation
- **5 query intents**: explanatory, comparative, technical, sota, discovery
- **Intent-specific prompts** — different LLM templates per intent
- **Context compression** — full-passage for explanatory, MMR selection for others

## Project Structure

```
db/
  schema.sql              PostgreSQL schema
  database.py             Connection manager + CRUD helpers
ingest/
  ingest_arxiv.py         Seed + keyword ingestion with relevance filter
  citation_expander.py    Semantic Scholar citation expansion
  timeline_balancer.py    Era distribution checker + gap filler
  chunking.py             Full-text chunking
index/
  build_qdrant.py         Text-only Qdrant builder
  build_bm25.py           Legacy stub (FTS is automatic via PostgreSQL)
  params.yaml             Pipeline configuration
api/
  app.py                  FastAPI server with text-only endpoints
  retrieval.py            Text-only hybrid retrieval
  fetch_data.py           Data fetching for deployment
rerank/
  reranker.py             Cross-encoder reranking
  evaluate.py             Retrieval evaluation metrics
frontend/
  index.html              Web UI
scripts/
  run_mech_interp_pipeline.bat   Full pipeline script
tests/
  queries.jsonl           Evaluation queries
```

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Conda environment `pytorch`

### 1. Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/arxiv_rag

# LLM
GROQ_API_KEY=gsk_...

# Cloudflare R2
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=arxiv-rag-assist
R2_ENDPOINT=https://....r2.cloudflarestorage.com

# Models
EMBEDDING_MODEL=intfloat/e5-base-v2
IMAGE_EMBEDDING_MODEL=google/siglip-base-patch16-224

# Corpus
CORPUS_TARGET_MIN=500
CORPUS_TARGET_MAX=5000
```

### 2. Install Dependencies

```bash
conda activate pytorch
pip install -r requirements.txt
```

### 3. Run Full Pipeline

```bash
scripts\run_mech_interp_pipeline.bat
```

Or run steps individually:

```bash
# Step 1: Ingest seed papers
conda run -n pytorch python ingest/ingest_arxiv.py --mode seed

# Step 2: Expand citations
conda run -n pytorch python ingest/citation_expander.py

# Step 3: Keyword gap-fill
conda run -n pytorch python ingest/ingest_arxiv.py --mode keyword

# Step 4: Timeline balance
conda run -n pytorch python ingest/timeline_balancer.py --fill-gaps

# Step 5: Download PDFs + extract text
conda run -n pytorch python ingest/ingest_arxiv.py --mode enrich

# Step 6: Chunk full text
conda run -n pytorch python ingest/chunking.py --source auto --reset

# Step 7: Build indexes (FTS is automatic via PostgreSQL)
conda run -n pytorch python index/build_qdrant.py
```

### 4. Start API

```bash
conda run -n pytorch uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Hybrid retrieval + LLM answer generation |
| `POST` | `/query/stream` | SSE streaming version of `/query` |
| `GET` | `/paper/{id}` | Paper metadata (includes `layer`, `is_seed`) |
| `GET` | `/paper/{id}/similar` | Find similar papers |
| `GET` | `/health` | Health check with per-collection counts |

## PostgreSQL Schema

| Table | Description |
|-------|-------------|
| `papers` | Paper metadata with `is_seed`, `layer`, `source` fields |
| `chunks` | Retrieval units with `chunk_type`, `modality`, `layer` metadata |
| `citation_edges` | Citation graph (reference/citation edges between papers) |

## Layer Tags

| Layer | Description |
|-------|-------------|
| `prerequisite` | Pre-2020 foundational work (transformers, attention, representation learning) |
| `foundation` | 2020-2022 work introducing key mech interp concepts |
| `core` | 2021-2024 direct mechanistic interpretability research |
| `latest` | 2024+ cutting-edge work |

## Configuration

See `index/params.yaml` for all configurable parameters including chunk sizes, embedding models, collection names, retrieval params, and timeline balance targets.
