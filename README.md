---
title: Arxiv RAG Assistant
emoji: 📚
colorFrom: red
colorTo: red
sdk: docker
pinned: false
---

# ArXiv RAG Assistant — Mechanistic Interpretability

A production-grade **Retrieval-Augmented Generation (RAG)** research assistant designed for Mechanistic Interpretability papers. The system performs hybrid dense + lexical retrieval, multi-stage reranking, intent-aware fusion, and grounded LLM answer generation across a curated corpus of ~3,000 arXiv papers.

For database schemas, CLI commands, and API docs, see the [**Technical Architecture Guide**](ARCHITECTURE.md).

---

## Key Features

- **Hybrid Retrieval**: Combines dense vector search (Qdrant + BGE-Large) with lexical search (BM25) for high recall.
- **Scientific Rigor**: Intent-aware precision tuning with a dedicated `evidence` intent, rigorous attribution enforcement, and a post-generation grounding verification heuristic.
- **Advanced Pipeline**: BGE-Reranker-v2-m3, Parent-Child chunking, and MMR diversity filtering. Query expansion and HyDE restricted to discovery queries for maximum precision.
- **Fast SSE Streaming**: Token-by-token streaming responses with an optimistic UI and intent-aware context sizing (4–6 chunks).
- **Live Ingestion**: Add new arXiv IDs on the fly. Background workers fetch the PDF, chunk it, and update vector/lexical indexes in real time.
- **Document-Scoped Chat**: Chat exclusively with a single paper using strict database-level filtering.

---

## 1. System Architecture

```mermaid
flowchart TB
    subgraph Client ["Client"]
        U[👤 User] --> FE["React / Vite SPA\n(Vercel)"]
    end

    subgraph Backend ["FastAPI Backend (Docker · HF Spaces)"]
        direction TB
        API["REST API + SSE Streaming\n/query · /query/stream"]
        AUTH["JWT Auth\nbcrypt · HS256 · refresh revocation"]
        CHAT["Conversation Manager\nSliding-window context · 20-query cap"]
        DOC["Document Ingestion\nLive arXiv PDF pipeline"]
        RET["HybridRetriever\nIntent-Aware Precision · BGE-Reranker-v2-m3"]
    end

    subgraph Storage ["Cloud Storage"]
        QDRANT[("Qdrant Cloud\narxiv_text · arxiv_docs")]
        NEON[("Neon PostgreSQL\nPapers · Users · Chats · Jobs")]
        R2[("Cloudflare R2\nBM25 Artifact ZIP")]
        REDIS[("Redis Cloud\nSession + Query Cache")]
    end

    subgraph LLM ["LLM Layer"]
        GEMINI["Gemini 2.5 Flash\nAuthenticated chat (primary)"]
        GROQ["Groq · Llama 3.3 70B\nPublic queries · HyDE · Fallback"]
    end

    FE -->|REST + SSE| API
    API --> AUTH & CHAT & DOC & RET
    RET --> QDRANT & REDIS
    RET -->|BM25 artifacts| R2
    CHAT --> NEON & REDIS
    AUTH --> NEON & REDIS
    DOC --> QDRANT & NEON
    RET --> GEMINI & GROQ
```

---

## 2. Core Workflows

### General Chat Flow (Streaming)

```mermaid
sequenceDiagram
    participant U as User
    participant FE as React Frontend
    participant API as FastAPI
    participant RET as HybridRetriever
    participant LLM as Gemini / Groq

    U->>FE: Type query + Send
    FE->>FE: Optimistic assistant bubble (empty)
    FE->>API: POST /conversations/{id}/query/stream
    API-->>FE: SSE: retrieval_start
    FE->>FE: Show "Hybrid retrieval in progress…"

    API->>RET: classify_intent → HyDE → expand → dense+BM25+RRF
    RET->>RET: parent-child · section boosts · rerank · MMR · compress
    API-->>FE: SSE: retrieval_done {num_chunks, trace}
    FE->>FE: Show "Retrieved N chunks · reranked"

    API->>LLM: build_prompt → stream_generate_answer
    loop Token streaming
        LLM-->>API: token chunk
        API-->>FE: SSE: token {content}
        FE->>FE: Append token to bubble
    end

    API-->>FE: SSE: done {sources, message_count}
    FE->>FE: Attach source citations to bubble
    API->>API: Save messages + cache in Redis
```

### Add Document Flow (Live Ingestion)

```mermaid
flowchart LR
    A["User submits\narXiv ID"] --> B["POST /documents/add\n(authenticated)"]
    B --> C["Queue document job\n(Neon · status=pending)"]
    C --> D["Background thread"]
    D --> E["arXiv API\nMetadata fetch"]
    E --> F["PDF Download\nCloudflare R2 / direct"]
    F --> G["PyMuPDF extraction\n+ pdfplumber fallback"]
    G --> H["Section-Sentence Chunker\nHierarchical · 450-token target"]
    H --> I["BGE-Large-EN-v1.5\nEmbedding (1024-dim)"]
    I --> J["Qdrant upsert\narxiv_text + arxiv_docs"]
    H --> K["BM25 hot-reload\nin-memory index update"]
    J & K --> L["Job status = done\nImmediately queryable"]
    L --> M["Frontend polls\nGET /documents/status/{job_id}"]
```

### Chat with Document Flow

```mermaid
flowchart TB
    U["User opens paper\nin Dashboard"] --> SC["Create conversation\nwith paper_id"]
    SC --> Q["User query"]
    Q --> FILT["Qdrant MatchValue filter\npaper_id == target"]
    FILT --> D["Dense retrieval\narxiv_text (paper only)"]
    Q --> BF["BM25 post-filter\npaper_id == target"]
    D & BF --> RRF["RRF fusion\n(within paper)"]
    RRF --> RR["BGE-Reranker-v2-m3\nPrecision reranking"]
    RR --> PROMPT["Scientific-grounded prompt\nStrict attribution & uncertainty"]
    PROMPT --> GEN["Gemini 1.5 / 3.x Flash\n→ Groq fallback"]
    GEN --> ANS["Grounded answer\nwith claim verification"]

    style FILT fill:#7c3aed,color:#fff
    style BF fill:#7c3aed,color:#fff
    style PROMPT fill:#dc2626,color:#fff
```

### Similar Papers Flow

```mermaid
flowchart LR
    A["GET /paper/{id}/similar"] --> B{"arxiv_docs\nvector exists?"}
    B -->|Yes| C["Use title+abstract\npaper-level vector\n(arxiv_docs collection)"]
    B -->|No| D["Compute mean over\nchunk vectors\n(arxiv_text collection)"]
    C & D --> E["Qdrant nearest-neighbour\nsearch · cosine similarity"]
    E --> F["Filter out source paper\n+ dedup by paper_id"]
    F --> G["Return top-N\nsimilar papers\nwith similarity scores"]
```

---

## 3. Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18 + Vite + Tailwind CSS (Vercel) |
| **Backend** | FastAPI 0.100+ (Python 3.10+) |
| **Vector DB** | Qdrant Cloud (HNSW, cosine, m=32, ef=400) |
| **Relational DB** | Neon PostgreSQL (papers, users, chats, jobs) |
| **Object Storage** | Cloudflare R2 (BM25 artifact bundle) |
| **Cache** | Redis Cloud (session cache, query cache) |
| **Embeddings** | BAAI/bge-large-en-v1.5 (1024-dim) |
| **Reranker** | BAAI/bge-reranker-v2-m3 |
| **Primary LLM** | Google Gemini 1.5 / 3.x Flash |
| **Fallback LLM** | Groq LLaMA 3.3 70B Versatile |

---

## 4. Setup & Deployment

### Backend

Copy `.env.example` to `.env` and fill in API keys (Neon, Qdrant, Google, Groq).

```bash
cd backend
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

Configure `VITE_API_URL` to point to the FastAPI backend.

```bash
cd frontend
npm install
npm run dev
```

### Deployment Strategy
- **Backend**: Deployed as a Docker container (Render, Hugging Face Spaces).
- **Frontend**: Deployed as a serverless SPA on Vercel.
- **Cold Starts**: On container boot, the backend fetches `artifacts_v1.zip` from Cloudflare R2 to load the BM25 lexical index instantly into memory, while vectors are served via Qdrant Cloud.

---

For deeper technical documentation, please see [**ARCHITECTURE.md**](ARCHITECTURE.md).
