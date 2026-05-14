---
title: ArXiv RAG Assistant
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: api/app.py
pinned: false
---

# Backend — ArXiv RAG Assistant

FastAPI service: hybrid **Qdrant dense** + **BM25 lexical** retrieval, cross-encoder rerank, Gemini/Groq generation, JWT auth, Redis caching, and optional document ingest.

Full architecture, pipeline stages, evaluation commands, and environment reference are in the **[repository root README](../README.md)**.

## Quick facts

- **Dense:** `arxiv_text` (chunk vectors from contextual text). **Parent:** `arxiv_docs` (one **title + abstract** vector per paper), then best in-paper chunks from `arxiv_text`.
- **Lexical:** BM25 over field-tagged `lexical_index_text` in `data/chunks_text.jsonl` (see `index/build_bm25.py`).
- **Sections:** Normalized once at chunk time (`utils/section_labels.py`) and again at load for legacy JSONL.
- **Artifacts:** `data/artifact_manifest.json` records schema version and build metadata.
- **Similar papers:** Prefer `arxiv_docs` vector; fallback mean over chunk vectors if needed.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## CLI (offline)

```bash
python cli.py --help
python cli.py pipeline qdrant --reset-qdrant   # example: rebuild vectors only
python cli.py index --target bm25
```

## Tests

```bash
pytest -q tests/test_api_contract.py
```

## Key environment variables

| Area | Variables |
|------|-----------|
| Retrieval | `QDRANT_URL`, `QDRANT_API_KEY`, `EMBEDDING_MODEL`, `RETRIEVAL_SKIP_*` (ablations), `ENABLE_PARENT_CHILD`, `ENABLE_RERANKER`, `ENABLE_MMR` |
| Qdrant build | `QDRANT_CHUNK_ENCODE_WINDOW`, `QDRANT_DOC_EMBED_BATCH`, `QDRANT_EMBED_BATCH_SIZE`, `QDRANT_UPSERT_BATCH_SIZE` |
| LLM | `GOOGLE_API_KEY`, `GROQ_API_KEY`, `GEMINI_MODEL`, `GROQ_MODEL`, rate limits |
| Auth | `JWT_SECRET_KEY`, `JWT_TOKEN_VERSION`, `ENVIRONMENT=production` (enforces strong secret), `REDIS_URL` (rate limit + refresh revocation + query cache buster) |
| Data | `DATA_DIR`, `CHUNKS_PATH`, R2 vars for `api/fetch_data.py`, `SKIP_ARTIFACT_FETCH` to skip R2 bootstrap on startup |
