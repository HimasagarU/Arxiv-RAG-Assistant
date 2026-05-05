# ArXiv RAG Assistant — Mechanistic Interpretability

A production-grade **Retrieval-Augmented Generation (RAG)** system specialized for **mechanistic interpretability** research. Built around a curated corpus of transformer circuits, sparse autoencoders, activation patching, and related papers spanning 2017–2025.

## Architecture

```
Seed Papers (15 curated mech interp papers)
  → Citation Expansion (Semantic Scholar API)
  → Keyword Gap-Filling (arXiv API)
  → Timeline Balancing (early / middle / recent eras)
  → PostgreSQL (papers, chunks, citation_edges)
  → PDF Download + Full Text Extraction (PyMuPDF)
  → Full-Text Chunking (overlapping token windows)
  → Qdrant Cloud (dense vector index, BGE-large-en-v1.5)
  → PostgreSQL FTS (lexical retrieval via tsvector)
  → Intent-Aware Hybrid Retrieval (RRF fusion)
  → Cross-Encoder Reranking (BGE-reranker-large)
  → LLM Answer Generation (Groq / Llama 3.3 70B)
```

## Key Features

### Seed-Driven Corpus Building
- **15 curated seed papers** spanning 2017–2024 (Attention Is All You Need → Scaling Monosemanticity)
- **Backward citation expansion** — pull in foundational/prerequisite papers via references
- **Forward citation expansion** — capture newer work that builds on seeds
- **Keyword gap-filling** — targeted arXiv queries for mech interp topics
- **Timeline balancing** — enforce coverage across early (20–35%), middle (30–45%), and recent (20–35%) eras
- **Layer tagging** — every paper tagged as `prerequisite`, `foundation`, `core`, or `latest`

### Hybrid Retrieval Pipeline
- **Dense retrieval**: Qdrant Cloud with BGE-large-en-v1.5 embeddings (1024-dim, HNSW m=32)
- **Lexical retrieval**: PostgreSQL full-text search with weighted tsvector (title/categories/authors/text)
- **RRF fusion**: Intent-aware Reciprocal Rank Fusion with per-intent weight tuning
- **Cross-encoder reranking**: BGE-reranker-large with combined title+text mode for explanatory queries
- **Paper diversity**: Max 2 chunks per paper to ensure breadth across sources

### Intent-Aware Query Processing
- **5 query intents**: explanatory, comparative, technical, sota, discovery
- **Intent-specific prompts** — different LLM templates per intent
- **Intent-specific retrieval weights** — explanatory favors dense, discovery favors FTS
- **Context compression** — full-passage for explanatory, concatenated for others

### Full-Text Extraction
- **PDF download and parsing** with PyMuPDF (pdfplumber fallback)
- **Full-text normalization** — removes references, acknowledgements, appendix sections
- **Overlapping token chunking** — 450-token windows with 15% overlap
- **Section hint detection** — tags chunks with introduction, method, results, etc.

## Project Structure

```
db/
  schema.sql              PostgreSQL schema (papers, chunks, citation_edges)
  database.py             Connection manager + CRUD + FTS search
ingest/
  ingest_arxiv.py         Seed + keyword ingestion with relevance filter
  citation_expander.py    Semantic Scholar citation expansion
  timeline_balancer.py    Era distribution checker + gap filler
  chunking.py             Full-text chunking with section detection
  r2_storage.py           Cloudflare R2 artifact storage
storage/
  local_pdf_store.py      Local PDF cache management
index/
  build_qdrant.py         Qdrant Cloud vector index builder
  params.yaml             Pipeline configuration
api/
  app.py                  FastAPI server with streaming + non-streaming endpoints
  retrieval.py            Hybrid retrieval (dense + FTS + RRF + reranking)
  fetch_data.py           R2 data fetching for deployment
  entrypoint.sh           Docker entrypoint script
rerank/
  reranker.py             Cross-encoder reranking (BGE-reranker-large)
  evaluate.py             Retrieval evaluation metrics (Recall, MRR, nDCG)
frontend/
  index.html              Web UI (single-page, academic theme)
scripts/
  run_mech_interp_pipeline.bat   Full pipeline script
  run_index.bat                  Index build script
  rebuild_indexes.bat            Rebuild indexes script
tests/
  test_eval.py            Unit + integration tests
  queries.jsonl           Evaluation queries
```

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ (or Neon for cloud deployment)
- Qdrant Cloud cluster
- Groq API key (for LLM generation)

### 1. Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/arxiv_rag

# Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# LLM
GROQ_API_KEY=gsk_...

# Cloudflare R2 (optional, for artifact storage)
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=arxiv-rag-assist
R2_ENDPOINT=https://....r2.cloudflarestorage.com

# Models
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline

```bash
scripts\run_mech_interp_pipeline.bat
```

Or run steps individually:

```bash
# Step 1: Ingest seed papers
python ingest/ingest_arxiv.py --mode seed

# Step 2: Expand citations (Semantic Scholar)
python ingest/citation_expander.py

# Step 3: Keyword gap-fill (arXiv API)
python ingest/ingest_arxiv.py --mode keyword

# Step 4: Timeline balance check + fill
python ingest/timeline_balancer.py --fill-gaps

# Step 5: Download PDFs + extract full text
python ingest/ingest_arxiv.py --mode enrich --pdf-timeout 60

# Step 6: Chunk full text
python ingest/chunking.py --source auto --reset

# Step 7: Build Qdrant vector index (FTS is automatic via PostgreSQL)
python index/build_qdrant.py
```

### 4. Start API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Docker Deployment

```bash
docker compose up -d
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Hybrid retrieval + rerank + LLM answer generation |
| `POST` | `/query/stream` | SSE streaming version of `/query` |
| `GET` | `/paper/{id}` | Paper metadata (includes `layer`, `is_seed`) |
| `GET` | `/paper/{id}/similar` | Find similar papers via embedding similarity |
| `GET` | `/health` | Health check with collection counts + cache stats |
| `GET` | `/keep-alive` | Lightweight ping for uptime monitoring |

## PostgreSQL Schema

| Table | Description |
|-------|-------------|
| `papers` | Paper metadata with `is_seed`, `layer`, `source`, `semantic_scholar_id` fields |
| `chunks` | Retrieval units with `search_tsv` (auto-generated tsvector for FTS), section hints, layer metadata |
| `citation_edges` | Citation graph (reference/citation edges between seed papers and expanded corpus) |

## Retrieval Pipeline Details

### Dense + FTS Hybrid Search
1. **Dense**: Query encoded with BGE-large-en-v1.5 → Qdrant ANN search (ef=200)
2. **Lexical**: PostgreSQL `websearch_to_tsquery` with weighted tsvector ranking
3. **Fusion**: Reciprocal Rank Fusion with intent-aware weights
4. **Diversity**: Max 2 chunks per paper, layer-aware balancing
5. **Reranking**: BGE-reranker-large cross-encoder (top 3×N candidates → top N)

### Intent-Aware Weights (dense, lexical)
| Intent | Dense | Lexical | Rationale |
|--------|-------|---------|-----------|
| Explanatory | 0.7 | 0.3 | Concept queries favor semantic similarity |
| Comparative | 0.6 | 0.4 | Balanced for multi-topic comparison |
| Technical | 0.5 | 0.5 | Equations/formulas need both signals |
| SOTA | 0.7 | 0.3 | Research trends favor semantic |
| Discovery | 0.4 | 0.6 | Keyword-heavy queries favor FTS |

## Layer Tags

| Layer | Description |
|-------|-------------|
| `prerequisite` | Pre-2020 foundational work (transformers, attention, representation learning) |
| `foundation` | 2020–2022 work introducing key mech interp concepts |
| `core` | 2021–2024 direct mechanistic interpretability research |
| `latest` | 2024+ cutting-edge work |

## Configuration

See `index/params.yaml` for all configurable parameters including chunk sizes, embedding models, collection names, retrieval params, and timeline balance targets.
