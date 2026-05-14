# ArXiv RAG Assistant — Technical Architecture

This document contains the deep technical specifications, pipeline details, database schemas, and CLI references for the ArXiv RAG Assistant. For a high-level overview, see the [main README](README.md).

---

## 1. System Architecture

```mermaid
flowchart TB
    subgraph Client
        U[User] --> FE[React / Vite Frontend<br/>Vercel]
    end

    subgraph Backend ["FastAPI Backend (Docker)"]
        direction TB
        API[REST API + SSE Streaming]
        AUTH[JWT Auth<br/>bcrypt + HS256]
        RET[HybridRetriever<br/>Dense + Lexical + Rerank]
        CHAT[Conversation Manager<br/>Sliding Window Context]
        DOC[Document Ingestion<br/>Live Upload Pipeline]
    end

    subgraph Storage ["Cloud Storage Layer"]
        QDRANT[(Qdrant Cloud<br/>arxiv_text + arxiv_docs)]
        NEON[(Neon PostgreSQL<br/>Papers · Citations · Users · Chats)]
        R2[(Cloudflare R2<br/>BM25 Artifacts ZIP)]
        REDIS[(Redis Cloud<br/>Session + Query Cache)]
    end

    subgraph LLM ["LLM Layer"]
        GEMINI[Gemini 2.5 Flash<br/>Primary Generation]
        GROQ[Groq LLaMA 3.3 70B<br/>Fallback + Compression]
    end

    FE -->|REST + SSE| API
    API --> AUTH & CHAT & DOC & RET
    RET --> QDRANT & REDIS
    RET -->|BM25 Artifacts| R2
    CHAT --> NEON & REDIS
    AUTH --> NEON & REDIS
    DOC --> QDRANT & NEON
    RET --> GEMINI & GROQ
```

---

## 2. Offline Ingestion Pipeline

The offline pipeline runs locally to build the full corpus. PDFs are preserved locally; parsed full text is stored in Neon `papers` table; chunks go to **Qdrant** (vector + payload) and **BM25 artifacts** (local files uploaded to R2).

```mermaid
flowchart LR
    subgraph Ingest ["1. Ingest"]
        SEED[Seed Papers] --> ARXIV[arXiv API]
        KW[Keyword Queries] --> ARXIV
        S2[Semantic Scholar] --> ARXIV
        ARXIV --> NEON_P[(Neon: papers)]
    end

    subgraph Enrich ["2. Enrich"]
        NEON_P --> DL[PDF Download]
        DL --> PARSE[PyMuPDF / pdfplumber]
        PARSE --> NEON_P
    end

    subgraph Chunk ["3. Chunk"]
        NEON_P --> CHUNK[Section-Sentence Chunker]
        CHUNK --> JSONL[data/chunks.jsonl]
    end

    subgraph Index ["4. Index"]
        JSONL --> BM25[BM25Okapi]
        JSONL --> EMBED[BGE-large-en-v1.5]
        EMBED --> QD[(Qdrant Cloud)]
        BM25 --> ZIP[artifacts_v1.zip]
        ZIP --> R2_UP[(Cloudflare R2)]
    end
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
flowchart TB
    Q[User Query] --> IC[Intent Classification]
    IC --> QD_DENSE[Intent-Aware Gating]

    QD_DENSE --> DENSE[Dense: Qdrant arxiv_text]
    QD_DENSE --> PC[Parent-Child: arxiv_docs → arxiv_text]
    QD_DENSE --> HYDE[HyDE Restricted Excerpt]
    IC --> LEX[Lexical: BM25]

    DENSE & PC & LEX --> MERGE[Intent-Aware RRF Fusion]
    HYDE -.->|auxiliary dense| DENSE

    MERGE --> RERANK[BGE-Reranker-v2-m3]
    RERANK --> BOOST[Section + Recency Boosts]
    BOOST --> MMR[MMR Diversity Filter]
    MMR --> COMPRESS[Context Compression]
    COMPRESS --> GEN[LLM Generation: Scientific Grounding]
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
    participant FE as React Frontend
    participant API as FastAPI
    participant RET as HybridRetriever
    participant LLM as Gemini / Groq

    U->>FE: Type query + Send
    FE->>API: POST /conversations/{id}/query/stream
    API->>RET: classify_intent → HyDE → expand → dense+BM25+RRF
    RET->>RET: parent-child · section boosts · rerank · MMR
    API->>LLM: build_prompt → stream_generate_answer
    LLM-->>FE: SSE: token stream
```

### Add Document Flow
```mermaid
flowchart LR
    A["User submits arXiv ID"] --> B["POST /documents/add"]
    B --> C["Queue job (Neon)"]
    C --> D["Background: Fetch & Download PDF"]
    D --> E["Extract & Chunk (PyMuPDF)"]
    E --> F["Embed (BGE) & Upsert (Qdrant)"]
    E --> G["BM25 hot-reload"]
```

### Chat with Document
```mermaid
flowchart LR
    Q["Query"] --> FILT["Qdrant MatchValue filter (paper_id)"]
    FILT --> D["Dense retrieval (within paper)"]
    Q --> BF["BM25 post-filter (paper_id)"]
    D & BF --> RRF["Cross-encoder rerank"]
    RRF --> PROMPT["Document-grounded prompt"]
    PROMPT --> GEN["Gemini / Groq"]
```

---

## 5. Database Schema

### Neon PostgreSQL

```mermaid
erDiagram
    papers {
        text paper_id PK
        text title
        text abstract
        timestamptz published
        text full_text
        text layer
    }
    citation_edges {
        text source_paper_id FK
        text target_paper_id FK
    }
    users {
        uuid id PK
        varchar email
        varchar hashed_password
    }
    conversations {
        uuid id PK
        uuid user_id FK
        varchar paper_id
    }
    messages {
        uuid id PK
        uuid conversation_id FK
        varchar role
        text content
    }
    papers ||--o{ citation_edges : "references"
    users ||--o{ conversations : "owns"
    conversations ||--o{ messages : "contains"
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
