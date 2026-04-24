# ArXiv RAG Assistant

ArXiv RAG Assistant is a hybrid retrieval-augmented generation system for research papers. It combines dense retrieval with Chroma, BM25 lexical retrieval, rank fusion, intent-aware reranking, context compression, and Groq-backed answer generation to answer questions over an ArXiv corpus with citations and source traces.

The current codebase supports both abstract-only and full-text ingestion. In the present workspace snapshot, the local corpus contains 10,440 papers, 10,332 with non-empty `full_text`, and 108 without full text. The current chunk index was built from full text wherever available and contains 397,438 chunks.

## What This Project Does

The system has four main stages:

1. Ingest papers from ArXiv into SQLite.
2. Chunk either abstract text or full text into retrieval units.
3. Build dense and lexical indexes from the chunks.
4. Serve a FastAPI application that retrieves, reranks, compresses, and answers user queries.

The code is designed to be reproducible on a local machine, but it can also run inside Docker or on a smaller deployment with BM25 or reranking disabled through environment flags.

## System Architecture

```text
ArXiv API / existing SQLite corpus
        -> ingest/ingest_arxiv.py
        -> SQLite papers table
        -> ingest/chunking.py
        -> data/chunks.jsonl
        -> index/build_chroma.py  -> data/chroma_db/
        -> index/build_bm25.py    -> data/bm25_index.pkl
        -> api/app.py + api/retrieval.py
        -> dense retrieval + BM25 + RRF fusion + reranking
        -> context compression + LLM answer generation
        -> /query, /query/stream, /paper/{id}, /paper/{id}/similar, /health
```

### Design goals

The implementation is optimized for a practical research assistant rather than a pure benchmark rig.

- Use one persistent SQLite database for metadata and full-text storage.
- Keep dense retrieval fast with ONNX-based embeddings via FastEmbed.
- Keep lexical retrieval simple and inspectable with BM25Okapi.
- Fuse heterogeneous retrieval signals with Reciprocal Rank Fusion instead of raw score blending.
- Use intent classification to change reranking, compression, and prompt style.
- Fall back gracefully when BM25, the reranker, or Groq are unavailable.

## Current Corpus Snapshot

These numbers reflect the current repository data files in `data/`.

| Item | Count / Size |
| --- | ---: |
| SQLite rows in `data/arxiv_papers.db` | 10,440 |
| Distinct `paper_id` values | 10,440 |
| Rows with non-empty `full_text` | 10,332 |
| Rows with empty or null `full_text` | 108 |
| Chunks in `data/chunks.jsonl` | 397,438 |
| Chunk sources in current `chunks.jsonl` | 100% `full_text` |
| Chroma documents | 397,438 |
| Chroma embedding dimension | 384 |
| `data/chroma_db` size | about 9.40 GB |
| `data/bm25_index.pkl` size | about 680 MB |

Important note: the current `chunks.jsonl` and indexes were produced from full-text mode, so every chunk in the current snapshot came from `full_text`. If you rebuild in abstract-only mode, the chunk count and distribution will change.

## Repository Layout

```text
api/
  app.py           FastAPI application and endpoints
  retrieval.py     Hybrid retrieval, fusion, reranking, compression
  Dockerfile       API container image
  entrypoint.sh    Container entrypoint
ingest/
  ingest_arxiv.py  ArXiv fetch, SQLite upsert, optional PDF extraction
  chunking.py      Chunk title + abstract/full_text into JSONL
index/
  build_chroma.py  Sentence embeddings + persistent Chroma collection
  build_bm25.py    BM25 index build and serialization
rerank/
  reranker.py      Cross-encoder reranker
  evaluate.py      Retrieval metrics runner
scripts/
  run_ingest.bat
  run_index.bat
  run_api.bat
  rebuild_indexes.bat
  run_pipeline_8cats_fullpdf.bat
tests/
  queries.jsonl     Evaluation queries
data/
  arxiv_papers.db
  chunks.jsonl
  chroma_db/
  bm25_index.pkl
logs/
  queries.jsonl     API query traces
```

## Ingestion Pipeline

The ingestion script is `ingest/ingest_arxiv.py`. It stores paper metadata in SQLite and can optionally download and extract PDF text.

### Database schema

The `papers` table stores:

- `paper_id`
- `title`
- `abstract`
- `authors`
- `categories`
- `pdf_url`
- `published`
- `updated`
- `full_text`

The ingest step is idempotent. It upserts by `paper_id`, so rerunning the script refreshes metadata instead of duplicating rows.

### Ingestion modes

`ingest_arxiv.py` supports three practical workflows:

1. **Abstract-only ingest**

   ```bash
   python ingest/ingest_arxiv.py --max-papers 20000
   ```

   This fetches ArXiv metadata and abstracts only. `full_text` remains empty unless you enrich later.

2. **Ingest with full-text extraction**

   ```bash
   python ingest/ingest_arxiv.py --max-papers 20000 --include-full-text --max-fulltext-papers 0
   ```

   This fetches metadata and downloads PDFs to extract body text with `pypdf`.

3. **Enrich an existing corpus with full text**

   ```bash
   python ingest/ingest_arxiv.py --enrich-existing-full-text --max-fulltext-papers 0
   ```

   This skips metadata refetch and fills in `full_text` for rows that are missing it.

### Ingestion implementation details

The ingest code is built around a few specific choices:

- ArXiv queries are built from categories using plain `OR` joins, then sent through `requests` for URL encoding.
- Pagination is conservative and retries are built in to reduce transient ArXiv API failures.
- PDF extraction is capped by timeout and maximum character count for safety.
- Full-text extraction requires `pypdf` in the selected environment.
- Existing rows are updated with `COALESCE(NULLIF(excluded.full_text, ''), papers.full_text)` so an empty extracted text does not overwrite a good one.

## Chunking Pipeline

The chunker is `ingest/chunking.py`. It reads rows from SQLite and creates retrieval chunks in JSONL format.

### Source modes

The chunker supports three modes:

- `abstract`: use title + abstract.
- `full_text`: use title + full text only.
- `auto`: prefer full text, fall back to abstract when full text is missing.

### Why chunking exists

Chunking makes the corpus usable for retrieval and reranking. A single paper is too large to store and search as one monolithic text block, so the code creates overlapping token windows.

The current implementation uses:

- `cl100k_base` tokenization via `tiktoken`
- 300-token chunk size
- 20% overlap

Each chunk record contains:

- `chunk_id`
- `paper_id`
- `chunk_text`
- `title`
- `authors`
- `categories`
- `token_count`
- `chunk_index`
- `total_chunks`
- `chunk_source`

### Why the tokenizer change matters

The chunking code now explicitly encodes text with `disallowed_special=()` so literal strings like `<|endoftext|>` are treated as ordinary text instead of crashing the run. This was necessary because some extracted PDFs contain token-like text that is not meant to be interpreted as a special token.

### Recommended chunking command

For a mixed corpus:

```bash
python ingest/chunking.py --source auto
```

For the current full-text snapshot:

```bash
python ingest/chunking.py --source full_text
```

## Dense Indexing

The dense index builder is `index/build_chroma.py`.

### What it does

- Loads `data/chunks.jsonl`.
- Encodes chunk text with `sentence-transformers/all-MiniLM-L6-v2` through SentenceTransformer.
- Normalizes embeddings.
- Stores embeddings and metadata in a persistent Chroma collection named `arxiv_chunks`.
- Saves a NumPy backup of the embeddings alongside the Chroma directory.

### Why this design

This is a practical local-first setup:

- `all-MiniLM-L6-v2` is small, fast, and good enough for semantic retrieval.
- Chroma persistence keeps the index easy to reuse between runs.
- Normalized vectors allow cosine-style similarity with simpler downstream scoring.

## BM25 Indexing

The lexical index builder is `index/build_bm25.py`.

### What it does

- Tokenizes each chunk using a lightweight lowercase alphanumeric tokenizer.
- Removes a compact stopword list.
- Builds a `BM25Okapi` index.
- Serializes the BM25 object plus the chunk ID mapping to `data/bm25_index.pkl`.

### Why BM25 still matters

Dense retrieval is good for semantic matching, but BM25 is still valuable for:

- exact terminology
- rare technical tokens
- acronyms and paper-specific keywords
- queries where lexical overlap matters more than paraphrase similarity

## Retrieval and Answer Generation

The runtime retrieval pipeline lives in `api/retrieval.py`, and the FastAPI layer is in `api/app.py`.

### Query flow

For a `/query` request, the system does the following:

1. Classify the query intent.
2. Retrieve candidates densely from Chroma.
3. Retrieve candidates lexically from BM25 when enabled.
4. Merge the two candidate sets with Reciprocal Rank Fusion.
5. Apply a small recency boost for non-explanatory queries.
6. Optionally filter by category, author, or start year.
7. Apply semantic pruning for explanatory queries.
8. Fall back to the ArXiv API for foundational papers when local confidence is low.
9. Rerank the merged candidates with a cross-encoder.
10. Compress context.
11. Build an intent-aware prompt.
12. Generate the final answer with Groq or a local fallback.

### Intent classification

The system classifies queries into:

- `explanatory`
- `comparative`
- `technical`
- `sota`
- `discovery`

This matters because the code changes behavior based on intent:

- explanatory queries get richer prompts and combined reranker text
- comparative queries use a balanced comparison prompt
- non-explanatory queries get MMR-based sentence compression
- explanatory queries skip recency boosting because foundational papers are often more useful than newest papers

### Dense retrieval

Dense search uses the ONNX-based FastEmbed text embedding model. The query is embedded, then sent to Chroma with an optional metadata filter.

### BM25 retrieval

BM25 scores the tokenized query against the tokenized chunk corpus. The top lexical hits are merged into the candidate pool.

### Fusion method

The current implementation uses Reciprocal Rank Fusion.

This is an important design choice: the repo previously used a weighted score blend idea, but the live code now uses RRF because dense similarity scores and BM25 scores are not naturally calibrated to the same scale. Rank-based fusion is more stable across different retrieval systems.

### Reranking

The runtime reranker is `rerank/reranker.py`, which currently uses `sentence-transformers` `CrossEncoder`.

Key behavior:

- It reranks only the top fused candidates.
- It lazily loads the model by default.
- It falls back to the fusion order if the model cannot load or predict.
- For explanatory queries, it uses a richer `title + chunk_text` pair.

This is a notable replacement in the codebase: the runtime reranker is currently CrossEncoder-based, not FlashRank-based.

### Context compression

The retriever compresses context before calling the LLM.

- For explanatory queries, it keeps full passage structure.
- For other intents, it uses MMR-style sentence selection to reduce redundancy.

This is a tradeoff between context length and information coverage.

### Answer generation

The API uses Groq when `GROQ_API_KEY` is available. If not, it falls back to a local extractive response.

There are two answer endpoints:

- `POST /query` for standard responses
- `POST /query/stream` for server-sent event streaming

### Similar papers

`GET /paper/{paper_id}/similar` computes the mean embedding of all chunks belonging to a paper, then queries Chroma for nearby papers and deduplicates by `paper_id`.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/query` | Hybrid retrieval, reranking, compression, and answer generation |
| `POST` | `/query/stream` | Streaming variant of `/query` using SSE |
| `GET` | `/paper/{paper_id}` | Paper metadata lookup |
| `GET` | `/paper/{paper_id}/similar` | Similar paper search |
| `GET` | `/health` | Basic index, DB, uptime, and cache metrics |
| `GET` | `/` | Frontend entry point if `frontend/index.html` exists |

## Runtime Behavior

The API keeps a small in-memory query cache:

- max size: 128 entries
- TTL: 300 seconds

The retrieval layer also adapts to available memory:

- BM25 can be disabled by environment flag or automatically on low-memory containers.
- The reranker can be disabled by environment flag or automatically on low-memory containers.
- The reranker can lazy load on first use.

These controls make the system more deployable on constrained machines.

## Configuration

### Ingestion environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARXIV_CATEGORIES` | `cs.AI,cs.LG` | Default ingest categories |
| `MAX_PAPERS` | `10000` | Default paper count |
| `DB_PATH` | `data/arxiv_papers.db` | SQLite database path |
| `PDF_TIMEOUT` | `30` | PDF download timeout in seconds |
| `MAX_FULLTEXT_CHARS` | `150000` | Max characters stored per full-text paper |
| `MAX_FULLTEXT_PAPERS` | `500` | Default cap for full-text enrichment |
| `CHUNKS_PATH` | `data/chunks.jsonl` | Chunk output path |
| `CHUNK_SOURCE_MODE` | `abstract` | Default chunking source when no CLI arg is passed |

### Retrieval and API environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHROMA_DIR` | `data/chroma_db` | Persistent dense index directory |
| `BM25_INDEX_PATH` | `data/bm25_index.pkl` | Serialized BM25 index path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Dense embedding model |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `RERANKER_LAZY_LOAD` | `true` | Load reranker on first use |
| `ENABLE_BM25` | auto | Enable or disable BM25 |
| `BM25_MIN_MEMORY_MB` | `640` | Auto-disable BM25 below this memory limit |
| `ENABLE_RERANKER` | auto | Enable or disable reranker |
| `RERANKER_MIN_MEMORY_MB` | `768` | Auto-disable reranker below this memory limit |
| `GROQ_API_KEY` | unset | Enable Groq LLM generation |

## How To Run

### Local development

```bash
conda activate pytorch
pip install -r requirements.txt
```

### Ingest and build indexes

For the current full-text workflow:

```bash
python ingest/chunking.py --source full_text
python index/build_chroma.py
python index/build_bm25.py
```

If you need a mixed corpus that prefers full text but falls back to abstract:

```bash
python ingest/chunking.py --source auto
```

### Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

On Windows, the repo also includes batch launchers under `scripts/`.

- `scripts/run_ingest.bat`
- `scripts/run_index.bat`
- `scripts/run_api.bat`
- `scripts/rebuild_indexes.bat`
- `scripts/run_pipeline_8cats_fullpdf.bat`

`scripts/run_api.bat` is root-aware, so it can be run from the `scripts` directory without breaking imports.

## Evaluation

The evaluation script is `rerank/evaluate.py`.

It reports:

- Recall@K
- Precision@K
- nDCG@K
- MRR

The script expects a JSONL file such as `tests/queries.jsonl` with query text and relevant chunk IDs.

### Example

```bash
python rerank/evaluate.py --queries tests/queries.jsonl
```

The `results/metrics.md` file in this repo contains an older 5k-paper benchmark snapshot. It is useful as a historical reference, but it should not be treated as the current corpus state.

## Why These Implementation Choices Were Made

### SQLite for corpus metadata

SQLite is easy to ship, portable, and good enough for the current corpus size. It avoids needing a server process just to store paper metadata and full text.

### Chroma for dense retrieval

Chroma gives persistent vector storage and fast nearest-neighbor search with minimal code.

### BM25 for lexical retrieval

BM25 is still strong for exact terminology and sparse scientific keywords, especially in technical paper corpora.

### RRF instead of score blending

Dense scores and BM25 scores live on different scales. Rank fusion is more reliable than trying to tune a shared numeric weight across unrelated score distributions.

### CrossEncoder reranking

The reranker improves precision on the merged candidate set by scoring query-passage pairs directly.

### Intent-aware prompt and compression

Not every query should be handled the same way. Explanatory questions benefit from full passage context, while discovery queries benefit from compressed, diverse sentence selection.

### Local fallback paths

The system still returns useful output when Groq is unavailable. It also logs retrieval traces and stores them in `logs/queries.jsonl` for inspection.

## Limitations And Current Tradeoffs

- The current corpus is large enough that Chroma and BM25 are both sizable on disk.
- The local retrieval quality depends on the quality of the chunking and full-text extraction.
- Some papers still lack `full_text`, so `--source full_text` will skip them unless they are enriched or you use `--source auto`.
- The code currently uses a CrossEncoder reranker rather than FlashRank, so dependency documentation should follow the runtime code path.
- Historical benchmark numbers in `results/metrics.md` were produced on an older 5k-paper snapshot.

## Practical Workflow Recommendation

If you want the most reliable full-text workflow on this repo, use this order:

```bash
python ingest/ingest_arxiv.py --max-papers 20000 --include-full-text --max-fulltext-papers 0
python ingest/chunking.py --source full_text
python index/build_chroma.py
python index/build_bm25.py
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

If your corpus mixes papers with and without extracted full text, use `--source auto` for chunking instead.

## Files Worth Reading First

- `ingest/ingest_arxiv.py`
- `ingest/chunking.py`
- `index/build_chroma.py`
- `index/build_bm25.py`
- `api/retrieval.py`
- `api/app.py`
- `rerank/reranker.py`
