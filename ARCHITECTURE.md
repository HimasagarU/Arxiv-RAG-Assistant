# ArXiv RAG Assistant — Technical Architecture

This document contains the deep technical specifications, pipeline details, database schemas, and CLI references for the ArXiv RAG Assistant. For a high-level overview, see the [main README](README.md).

---

## 1. System Architecture

```mermaid
flowchart TB
    subgraph Client
        U[User] --> FE[React SPA\nVercel]
    end

    subgraph Backend ["FastAPI (HF Spaces)"]
        direction TB
        API[REST + SSE]
        AUTH[JWT Auth\nHS256]
        RET[Retriever\nHybrid+BGE]
        CHAT[Chat Mgr\nRedis/Neon]
        DOC[Ingestion\nPyMuPDF]
    end

    subgraph Storage ["Cloud Storage"]
        QDRANT[(Qdrant\nVectors)]
        NEON[(Neon PG\nRelational)]
        R2[(R2 Cloud\nBM25 ZIP)]
        REDIS[(Redis\nCache)]
    end

    subgraph LLM ["Inference"]
        GEMINI[Gemini 2.5\nPrimary]
        GROQ[Groq 3.3\nFallback]
    end

    FE -->|REST + SSE| API
    API --> AUTH & CHAT & DOC & RET
    RET --> QDRANT & REDIS & R2
    CHAT --> NEON & REDIS
    AUTH --> NEON & REDIS
    DOC --> QDRANT & NEON
    RET --> GEMINI & GROQ
```

---

## 2. Offline Ingestion Pipeline

```mermaid
flowchart LR
    S[Seed/KW] --> A[ArXiv API]
    A --> N[(Neon PG)]
    N --> D[PDF Download]
    D --> P[PyMuPDF\nParse]
    P --> C[Section\nChunker]
    C --> J[JSONL\nChunks]
    J --> B[rank-bm25\nIndex]
    J --> E[BGE-Large\nEmbed]
    E --> Q[(Qdrant)]
    B --> R[(R2 Cloud)]
```

### Chunking Strategy: Hierarchical Section-Sentence
1. **Section Detection** — Regex-based heading detection splits papers into sections (Introduction, Methods, Results, etc.)
2. **Sentence Packing** — NLTK sentence tokenizer packs sentences into chunks within each section.
3. **Profile-Based Sizing** — Target chunk sizes: `small` (250 tokens), `medium` (350 tokens), `large` (500 tokens).
4. **Overlap** — Last 1-2 sentences from previous chunk carried forward (≤80 tokens).
5. **Quality Validation** — Short chunks warned; runaway chunks rejected. Strips References/Appendices.
6. **Contextual Text** — Chunks enriched with title, authors, categories, section label, and local summary prefix.

### Parent–Child (`arxiv_docs`)
`arxiv_docs` stores **one vector per paper** built from **title + abstract**. Parent search targets the paper’s core contribution, then expands to pull the best in-paper chunks from `arxiv_text`.

---

## 3. Retrieval Pipeline

```mermaid
flowchart LR
    Q[Query] --> IC{Intent\nClass}
    IC --> D[Dense\nBGE]
    IC --> P[Parent\nDocs]
    IC --> L[Lexical\nBM25]
    D & P & L --> F[RRF\nFusion]
    F --> R[BGE\nRerank]
    R --> M[MMR\nFilter]
    M --> LLM[Gemini 2.5\nGrounding]
```

### Retrieval Features & Ablations

| Feature / Variable | Description |
|--------------------|-------------|
| **Dense Retrieval** | Qdrant `arxiv_text` with BGE-large-en-v1.5 (1024-dim). Disable with `RETRIEVAL_SKIP_DENSE=true`. |
| **Lexical Retrieval**| BM25 on tagged fields. Disable with `RETRIEVAL_SKIP_LEXICAL=true`. |
| **RRF Fusion** | Intent-aware merge weights. Disable rerank with `RETRIEVAL_SKIP_RERANK=true`. |
| **Parent-Child** | Expands chunks from `arxiv_docs` hits. `ENABLE_PARENT_CHILD` |
| **MMR Diversity** | Cosine-similarity deduplication. `ENABLE_MMR` |
| **Section Boosting**| Intent-specific weight multipliers (e.g. Methods×1.32 for technical). |
| **Expansion Gating**| Intent-aware bypass of LLM/embedding expansions for precision lookups. |
| **HyDE / Expansion**| LLM query expansion (Discovery only). `ENABLE_HYDE=false` (default off). |
| **Context Sizing**  | Intent-aware chunk limits (4–6 chunks) to reduce noise. |
| **Verification**    | Post-generation grounding verification heuristic. |

---

## 4. Core Workflows

### General Chat Flow (Streaming)
```mermaid
sequenceDiagram
    participant U as User
    participant API as Backend (HFS)
    participant RET as Retriever
    participant LLM as Inference

    U->>API: /query/stream
    API-->>U: sse: retrieval_start
    API->>RET: Hybrid (BGE+BM25)
    RET-->>API: Grounded Context
    API-->>U: sse: retrieval_done
    API->>LLM: Augmented Prompt
    loop SSE Stream
        LLM-->>U: sse: token
    end
    API-->>U: sse: done
```

### Add Document Flow
```mermaid
flowchart LR
    ID[arXiv ID] --> POST[POST /add]
    POST --> Q[Neon Queue]
    Q --> B[Worker: PDF]
    B --> C[Extract+Chunk]
    C --> E[Embed+Upsert]
    E --> S[Sync BM25]
```

### Chat with Document
```mermaid
flowchart LR
    Q[Query] --> F[Filter: Paper]
    F --> D[Dense: Paper]
    Q --> L[Lexical: Paper]
    D & L --> R[BGE Rerank]
    R --> P[Grounded Prompt]
    P --> G[Gemini/Groq]
```

---

## 5. Database Schema

### Neon PostgreSQL

```mermaid
erDiagram
    papers {
        text paper_id PK
        text title
        text full_text
    }
    chunks {
        text chunk_id PK
        text paper_id FK
        text chunk_text
    }
    citation_edges {
        text source_paper_id FK
        text target_paper_id FK
    }
    users {
        uuid id PK
        text email
    }
    document_jobs {
        uuid id PK
        uuid user_id FK
        text arxiv_id
    }
    conversations {
        uuid id PK
        uuid user_id FK
        text paper_id
    }
    messages {
        uuid id PK
        uuid conversation_id FK
        text content
    }

    papers ||--o{ chunks : decomposes
    papers ||--o{ citation_edges : cites
    users ||--o{ document_jobs : tracks
    users ||--o{ conversations : starts
    conversations ||--o{ messages : records
```

### Qdrant Cloud Collections
- `arxiv_text`: Chunk-level vectors (1024-dim BGE). Payload includes chunk_text, contextual_text, section_hint.
- `arxiv_docs`: Paper-level centroid vectors (1024-dim). Payload includes paper_id, title, abstract.

---

## 5. API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check with collection + cache stats |
| `POST` | `/query` | No | Corpus-wide retrieval + Groq generation |
| `POST` | `/query/stream` | No | SSE streaming variant of `/query` |
| `GET` | `/paper/{id}` | No | Paper metadata lookup |
| `GET` | `/paper/{id}/similar` | No | Mean-embedding similar paper search |
| `POST` | `/auth/login` | No | Login + return JWTs |
| `POST` | `/auth/refresh` | No | Refresh access token |
| `POST` | `/conversations/{id}/query/stream` | Yes | Authenticated SSE streaming chat |
| `POST` | `/documents/add` | Yes | Queue live arXiv document ingestion |
| `GET` | `/metrics/performance` | No | Rolling latency + cache metrics |

---

## 6. Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `QDRANT_URL` / `API_KEY`| Qdrant Cloud cluster |
| `REDIS_URL` | Redis Cloud for session/query cache |
| `GOOGLE_API_KEY` | Gemini 2.5 Flash (authenticated chats) |
| `GROQ_API_KEY` | Groq fallback + public queries |
| `R2_ACCESS_KEY_ID` | Cloudflare R2 credentials |
| `JWT_SECRET_KEY` | JWT signing secret |

---

## 7. CLI Commands Reference

All commands run from `backend/` using `conda run -n pytorch` (or equivalent).

| Command | Purpose |
|---------|---------|
| `python -m cli ingest --mode all` | Run seed, keyword, and citation ingestion |
| `python -m cli chunk --reset` | Hierarchical section-sentence chunking |
| `python -m cli index --target both` | Build Qdrant vectors + BM25 artifacts |
| `python -m cli reset --yes` | Delete derived artifacts + reset Qdrant |
| `python -m cli health` | Corpus health summary from Neon |
| `python -m cli sync-metadata` | Sync paper metadata from artifacts → Neon |
| `python scripts/upload_artifacts.py` | Zip + upload BM25 bundle to R2 |

---

## 8. Evaluation & Observability

- **Retrieval Metrics**: Recall@K, Precision@K, MRR, nDCG@K evaluated via `backend/rerank/evaluate.py`.
- **RAGAS**: Faithfulness, Response Relevancy, Context Precision evaluated via `backend/eval/ragas_eval.py`.
- **Query Traces**: Detailed latency and stage timing written to `logs/queries.jsonl`.
- **Performance**: Exposes rolling P50/P95/P99 latency via `/metrics/performance`.
