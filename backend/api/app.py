"""
app.py â€” FastAPI application for the ArXiv RAG Assistant.

Endpoints:
    POST /query         â€” Hybrid retrieval + rerank + LLM answer generation
    GET  /paper/{id}    â€” Paper metadata lookup
    GET  /paper/{id}/similar â€” Find similar papers
    GET  /health        â€” Health check with basic metrics

Usage:
    conda run -n pytorch uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

import sys
# Ensure project root is in sys.path so 'db' and 'api' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOGS_DIR = "logs"
Path(LOGS_DIR).mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Simple LRU Cache for query results
# ---------------------------------------------------------------------------

class LRUCache:
    """Thread-safe, size-bounded LRU cache for query results."""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _make_key(self, query: str, top_k: int, category: str = None,
                  author: str = None, start_year: int = None) -> str:
        return f"{query.strip().lower()}|{top_k}|{category or ''}|{author or ''}|{start_year or ''}"

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] < self._ttl:
                    self._cache.move_to_end(key)
                    return entry["data"]
                del self._cache[key]
            return None

    def set(self, key: str, data: dict):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = {"data": data, "timestamp": time.time()}
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        with self._lock:
            size = len(self._cache)
        return {"size": size, "max_size": self._max_size, "ttl": self._ttl}


query_cache = LRUCache(max_size=128, ttl_seconds=300)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    category: Optional[str] = Field(default=None, description="Filter by ArXiv category (e.g. cs.AI)")
    author: Optional[str] = Field(default=None, description="Filter by author name")
    start_year: Optional[int] = Field(default=None, ge=2000, le=2030, description="Filter papers from this year onward")


class SourceInfo(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    authors: str = ""
    categories: str = ""
    chunk_text: str
    chunk_type: str = "text"
    modality: str = "text"
    section_hint: str = "other"
    layer: str = "core"
    rerank_score: float = 0.0


class AnalyticsInfo(BaseModel):
    top_authors: list[dict] = Field(default_factory=list)
    top_categories: list[dict] = Field(default_factory=list)
    layer_distribution: dict = Field(default_factory=dict)
    total_unique_papers: int = 0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    latency_ms: float
    retrieval_trace: dict
    analytics: AnalyticsInfo = Field(default_factory=AnalyticsInfo)
    cached: bool = False


class SimilarPaperInfo(BaseModel):
    paper_id: str
    title: str
    authors: str = ""
    categories: str = ""
    layer: str = ""
    similarity_score: float = 0.0
    chunk_text: str = ""


class SimilarPapersResponse(BaseModel):
    paper_id: str
    similar_papers: list[SimilarPaperInfo]


class PaperResponse(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: str
    categories: str
    pdf_url: str
    published: str
    layer: str = "core"
    is_seed: bool = False


class HealthResponse(BaseModel):
    status: str
    collections: dict = Field(default_factory=dict)
    db_papers: int = 0
    uptime_seconds: float = 0.0
    cache_stats: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global state (loaded on startup)
# ---------------------------------------------------------------------------

_state = {
    "retriever": None,
    "start_time": time.time(),
    "query_count": 0,
}

_metrics = {
    "request_latencies_ms": deque(maxlen=5000),
    "query_latencies_ms": deque(maxlen=5000),
    "cache_hit_latencies_ms": deque(maxlen=5000),
    "cache_miss_latencies_ms": deque(maxlen=5000),
}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes on startup."""
    log.info("Starting ArXiv RAG API...")
    _state["start_time"] = time.time()

    # Initialize application database (Supabase — users, conversations, jobs)
    try:
        from db.app_database import init_app_db, close_app_db
        await init_app_db()
        log.info("Application database initialized.")
    except Exception as e:
        log.error(f"Failed to initialize app database: {e}")
        log.warning("Auth, chat history, and document features will be unavailable.")

    def init_retriever():
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from api.fetch_data import fetch_and_extract
            
            log.info("Running data bootstrapper...")
            fetch_and_extract()
            
            from api.retrieval import HybridRetriever
            _state["retriever"] = HybridRetriever()
            log.info("HybridRetriever initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize retriever: {e}")
            log.warning("API will start but /query endpoint will be unavailable.")

    # Start retrieval initialization in background to not block HF Spaces readiness probe
    threading.Thread(target=init_retriever, daemon=True).start()

    yield

    # Shutdown
    try:
        from db.app_database import close_app_db
        await close_app_db()
    except Exception:
        pass
    log.info("Shutting down ArXiv RAG API...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ArXiv RAG Assistant",
    description="Hybrid RAG system for ArXiv papers with Qdrant Cloud dense retrieval, PostgreSQL full-text search, and cross-encoder reranking.",
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
if not allowed_origins:
    allowed_origins = ["http://localhost:3000"]
# Always allow localhost dev and common Vercel deploy patterns
allowed_origins.extend([
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------------------------------------------------
# Mount new routers (auth, chat, documents)
# ---------------------------------------------------------------------------

try:
    from api.auth import router as auth_router
    from api.chat import router as chat_router
    from api.documents import router as documents_router

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    log.info("Mounted routers: auth, chat, documents")
except Exception as e:
    log.warning(f"Failed to mount new routers: {e}. Core /query endpoint still available.")


@app.middleware("http")
async def request_timing_middleware(request, call_next):
    """Track end-to-end request latency for all API routes."""
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    _metrics["request_latencies_ms"].append(elapsed_ms)
    return response

# Serve frontend
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="frontend")


# ---------------------------------------------------------------------------
# Prompt builder â€” intent-aware templates
# ---------------------------------------------------------------------------

def _build_sources_block(passages: list[dict]) -> str:
    """Build numbered source list for the model to reference."""
    lines = []
    for i, p in enumerate(passages, 1):
        title = p.get('title', 'Untitled')
        lines.append(f'[{i}] "{title}"')
    return "\n".join(lines)


def build_prompt(query: str, compressed_context: str, passages: list[dict],
                 intent: str = "discovery") -> str:
    """Build a RAG prompt with intent-specific answer templates."""
    sources_block = _build_sources_block(passages)

    if intent == "explanatory":
        return _build_explanatory_prompt(query, compressed_context, sources_block)
    elif intent == "comparative":
        return _build_comparative_prompt(query, compressed_context, sources_block)
    else:
        return _build_general_prompt(query, compressed_context, sources_block)


def _build_explanatory_prompt(query: str, context: str, sources: str) -> str:
    return f"""You are an expert AI/ML research assistant. A student has asked an explanatory question.
Your job is to give a clear, accurate, step-by-step explanation grounded in the source passages.

CRITICAL RULES:
1. Every key claim must be supported by at least one source. Use citations [1], [2], etc.
2. If the sources lack evidence for a key part of the explanation, say "the provided sources do not cover this step" rather than guessing.
3. Do NOT write a literature review. Write a direct explanation of the mechanism/concept.
4. Use the Feynman Technique: be clear, but do NOT omit important technical details.

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
1. **Definition**: A clear 1-2 sentence definition of the concept. Start with double asterisks **like this**.
2. **How It Works**: A step-by-step explanation of the mechanism or pipeline. Number each step.
3. **Why It Works**: Explain the intuition behind why this approach is effective.
4. **Limitations**: Briefly note known limitations or failure modes.
5. **References**: List the numbered source titles.

---

SOURCE PASSAGES:
{context}

AVAILABLE SOURCES:
{sources}

QUESTION: {query}

ANSWER:"""


def _build_comparative_prompt(query: str, context: str, sources: str) -> str:
    return f"""You are an expert AI/ML research assistant. A student wants to compare two or more approaches.
Your job is to give a balanced, evidence-based comparison grounded in the source passages.

CRITICAL RULES:
1. Every claim must be supported by at least one source. Use citations [1], [2], etc.
2. Be fair and balanced â€” present strengths and weaknesses of each side.
3. Do NOT fabricate benchmark numbers. Only cite numbers found in the sources.

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
1. **Overview**: A 1-2 sentence summary of what is being compared. Start with double asterisks **like this**.
2. **Approach A**: Summary of the first approach â€” key mechanism, strengths.
3. **Approach B**: Summary of the second approach â€” key mechanism, strengths.
4. **Key Differences**: A clear comparison of the main differences (use a list).
5. **When to Use Each**: Practical guidance on when each approach is more appropriate.
6. **References**: List the numbered source titles.

---

SOURCE PASSAGES:
{context}

AVAILABLE SOURCES:
{sources}

QUESTION: {query}

ANSWER:"""


def _build_general_prompt(query: str, context: str, sources: str) -> str:
    return f"""You are an expert AI/ML research assistant. Your goal is to give a thorough,
well-structured answer that a smart graduate student would find genuinely useful.

CRITICAL RULES:
1. Read ALL provided source passages carefully before answering.
2. Use numbered citations [1], [2] to reference sources. NEVER mention chunk IDs.
3. Every key claim must be supported by at least one source passage.
4. If information is insufficient, say so honestly rather than speculating.
5. Use the Feynman Technique: explain clearly, but preserve technical precision.

STRUCTURE YOUR ANSWER LIKE THIS:
1. **Summary**: A bold 1-2 sentence executive summary answering the core question. Start with double asterisks **like this**.
2. **Key Findings**: The most important technical insights from the sources.
3. **Details**: Deeper explanation with evidence from the passages.
4. **References**: Numbered source titles.

---

SOURCE PASSAGES:
{context}

AVAILABLE SOURCES:
{sources}

QUESTION: {query}

ANSWER:"""


# ---------------------------------------------------------------------------
# Intent-aware system prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "explanatory": (
        "You are an expert AI/ML research assistant. When explaining concepts, "
        "give clear step-by-step explanations grounded in source evidence. "
        "Start with a bold definition. Use numbered citations [1], [2]. "
        "Never fabricate steps â€” if the sources don't cover something, say so."
    ),
    "comparative": (
        "You are an expert AI/ML research assistant. When comparing approaches, "
        "be balanced and evidence-based. Present both sides fairly. "
        "Use numbered citations [1], [2]. Start with a bold overview."
    ),
    "default": (
        "You are an expert AI/ML research assistant. Give thorough, well-structured answers "
        "using the Feynman Technique. Use numbered citations [1], [2] to reference sources. "
        "Start with a bold executive summary. Ground every claim in the source passages."
    ),
}


def get_system_prompt(intent: str = "discovery") -> str:
    """Get the system prompt for a given intent."""
    if intent in _SYSTEM_PROMPTS:
        return _SYSTEM_PROMPTS[intent]
    return _SYSTEM_PROMPTS["default"]


def generate_answer(prompt: str, intent: str = "discovery") -> str:
    """
    Generate an answer using Groq API.
    This project requires the remote LLM path.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for answer generation.")

    return _generate_groq(prompt, groq_api_key, intent=intent)


def _generate_groq(prompt: str, api_key: str, intent: str = "discovery") -> str:
    """Generate answer using Groq API with Llama 3.3 70B."""
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        system_prompt = get_system_prompt(intent)
        # Lower temperature for explanatory queries (more focused)
        temp = 0.1 if intent == "explanatory" else 0.2

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=temp,
            max_tokens=2048,
            top_p=0.9,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        log.error(f"Groq API error: {e}")
        raise


def _generate_groq_stream(prompt: str, api_key: str, intent: str = "discovery"):
    """Streaming generator: yields answer tokens one-by-one from Groq."""
    from groq import Groq

    client = Groq(api_key=api_key)
    system_prompt = get_system_prompt(intent)
    temp = 0.1 if intent == "explanatory" else 0.2

    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=temp,
        max_tokens=2048,
        top_p=0.9,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------

def log_query(query: str, response_data: dict):
    """Save query trace to JSON log file."""
    try:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "latency_ms": response_data.get("latency_ms", 0),
            "trace": response_data.get("retrieval_trace", {}),
            "num_sources": len(response_data.get("sources", [])),
            "cached": response_data.get("cached", False),
        }
        log_path = os.path.join(LOGS_DIR, "queries.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        log.warning(f"Failed to log query: {e}")


def _percentile(values: list[float], p: float) -> float:
    """Return percentile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * weight)


def _latency_summary(values: list[float]) -> dict:
    """Build latency summary stats in milliseconds."""
    if not values:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values), 1),
        "p50_ms": round(_percentile(values, 0.50), 1),
        "p95_ms": round(_percentile(values, 0.95), 1),
        "p99_ms": round(_percentile(values, 0.99), 1),
        "max_ms": round(max(values), 1),
    }


def _historical_query_latencies() -> list[float]:
    """Read all historical /query latencies from logs/queries.jsonl."""
    log_path = os.path.join(LOGS_DIR, "queries.jsonl")
    if not os.path.isfile(log_path):
        return []

    latencies: list[float] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                latency = row.get("latency_ms")
                if isinstance(latency, (int, float)):
                    latencies.append(float(latency))
    except Exception as e:
        log.warning(f"Failed to parse historical latency logs: {e}")
    return latencies


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve frontend."""
    index_file = _frontend_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "ArXiv RAG API - visit /docs for Swagger UI"}


@app.head("/", include_in_schema=False)
async def root_head():
    """HEAD probe for frontend root."""
    return Response(status_code=200)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Hybrid retrieval + rerank + answer generation.
    Supports metadata filtering by category, author, and publication year.
    Results are cached for 5 minutes.
    """
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Please ensure indexes are built.",
        )

    t0 = time.time()
    cache_key = query_cache._make_key(
        request.query, request.top_k, request.category,
        request.author, request.start_year
    )
    cached_result = query_cache.get(cache_key)
    if cached_result:
        cached_result["cached"] = True
        hit_ms = round((time.time() - t0) * 1000, 1)
        _metrics["cache_hit_latencies_ms"].append(hit_ms)
        _metrics["query_latencies_ms"].append(hit_ms)
        log_query(request.query, cached_result)
        return QueryResponse(**cached_result)

    _state["query_count"] += 1

    from api.retrieval import classify_query_intent
    intent = classify_query_intent(request.query)

    result = _state["retriever"].retrieve(
        request.query,
        top_n=request.top_k,
        category=request.category,
        author=request.author,
        start_year=request.start_year,
        intent=intent,
    )
    passages = result["passages"]
    trace = result["trace"]
    analytics = result.get("analytics", {})

    t_compress = time.time()
    compressed_context = _state["retriever"].compress_context(
        request.query, passages, intent=intent
    )
    trace["compress_ms"] = round((time.time() - t_compress) * 1000, 1)

    t_gen = time.time()
    prompt = build_prompt(request.query, compressed_context, passages, intent=intent)
    answer = generate_answer(prompt, intent=intent)
    trace["generation_ms"] = round((time.time() - t_gen) * 1000, 1)

    total_ms = round((time.time() - t0) * 1000, 1)
    _metrics["query_latencies_ms"].append(total_ms)
    _metrics["cache_miss_latencies_ms"].append(total_ms)

    sources = [
        SourceInfo(
            chunk_id=p["chunk_id"],
            paper_id=p["paper_id"],
            title=p["title"],
            authors=p.get("authors", ""),
            categories=p.get("categories", ""),
            chunk_text=p["chunk_text"],
            rerank_score=p.get("rerank_score", 0.0),
        )
        for p in passages
    ]

    response_data = {
        "answer": answer,
        "sources": sources,
        "latency_ms": total_ms,
        "retrieval_trace": trace,
        "analytics": AnalyticsInfo(**analytics) if analytics else AnalyticsInfo(),
        "cached": False,
    }

    cache_data = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "latency_ms": total_ms,
        "retrieval_trace": trace,
        "analytics": analytics,
        "cached": False,
    }
    query_cache.set(cache_key, cache_data)

    log_query(request.query, {
        "latency_ms": total_ms,
        "retrieval_trace": trace,
        "sources": [s.model_dump() for s in sources],
        "cached": False,
    })

    return QueryResponse(**response_data)


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming version of /query.
    Returns an SSE stream: first event is metadata (sources, trace, analytics),
    then each subsequent event is an answer token, ending with [DONE].
    """
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Please ensure indexes are built.",
        )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="Streaming requires GROQ_API_KEY to be set.",
        )

    t0 = time.time()
    _state["query_count"] += 1

    from api.retrieval import classify_query_intent
    intent = classify_query_intent(request.query)

    result = _state["retriever"].retrieve(
        request.query,
        top_n=request.top_k,
        category=request.category,
        author=request.author,
        start_year=request.start_year,
        intent=intent,
    )
    passages = result["passages"]
    trace = result["trace"]
    analytics = result.get("analytics", {})

    compressed_context = _state["retriever"].compress_context(
        request.query, passages, intent=intent
    )

    prompt = build_prompt(request.query, compressed_context, passages, intent=intent)

    sources = [
        {
            "chunk_id": p["chunk_id"],
            "paper_id": p["paper_id"],
            "title": p["title"],
            "authors": p.get("authors", ""),
            "categories": p.get("categories", ""),
            "chunk_text": p["chunk_text"],
            "rerank_score": p.get("rerank_score", 0.0),
        }
        for p in passages
    ]

    retrieval_ms = round((time.time() - t0) * 1000, 1)

    def event_generator():
        """SSE generator: metadata event, then token events, then DONE."""
        meta_payload = json.dumps({
            "type": "metadata",
            "sources": sources,
            "retrieval_trace": trace,
            "analytics": analytics,
            "retrieval_ms": retrieval_ms,
            "intent": intent,
        })
        yield f"data: {meta_payload}\n\n"

        try:
            for token in _generate_groq_stream(prompt, groq_api_key, intent=intent):
                token_payload = json.dumps({"type": "token", "content": token})
                yield f"data: {token_payload}\n\n"
        except Exception as e:
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

        total_ms = round((time.time() - t0) * 1000, 1)
        _metrics["query_latencies_ms"].append(total_ms)
        _metrics["cache_miss_latencies_ms"].append(total_ms)
        done_payload = json.dumps({"type": "done", "total_ms": total_ms})
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/paper/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: str):
    """Look up paper metadata by ArXiv ID."""
    if _state["retriever"] is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialized.")

    paper_info = _state["retriever"].papers_meta.get(paper_id)
    if not paper_info:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

    return PaperResponse(
        paper_id=paper_id,
        title=paper_info.get("title", ""),
        abstract=paper_info.get("abstract", ""),
        authors=paper_info.get("authors", ""),
        categories=paper_info.get("categories", ""),
        pdf_url=paper_info.get("pdf_url", ""),
        published=str(paper_info.get("published", "")),
        layer=paper_info.get("layer", "core"),
        is_seed=paper_info.get("is_seed", False),
    )


@app.head("/paper/{paper_id}", include_in_schema=False)
async def get_paper_head(paper_id: str):
    """HEAD probe for paper metadata existence."""
    if _state["retriever"] is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialized.")
    if paper_id not in _state["retriever"].papers_meta:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return Response(status_code=200)


@app.get("/paper/{paper_id}/similar", response_model=SimilarPapersResponse)
async def get_similar_papers(paper_id: str, top_n: int = 5):
    """Find papers similar to the given paper."""
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized.",
        )

    similar = _state["retriever"].find_similar_papers(paper_id, top_n=top_n)

    return SimilarPapersResponse(
        paper_id=paper_id,
        similar_papers=[SimilarPaperInfo(**p) for p in similar],
    )


@app.head("/paper/{paper_id}/similar", include_in_schema=False)
async def get_similar_papers_head(paper_id: str):
    """HEAD probe for similar papers endpoint availability."""
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized.",
        )
    return Response(status_code=200)


@app.get("/keep-alive")
async def keep_alive():
    """Lightweight endpoint to keep the server awake."""
    return {"status": "alive", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.head("/keep-alive", include_in_schema=False)
async def keep_alive_head():
    """HEAD probe for keep-alive endpoint."""
    return Response(status_code=200)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with per-collection counts."""
    collections = {}
    db_papers = 0

    if _state["retriever"] is not None:
        collections = dict(_state["retriever"].collections)
        db_papers = len(_state["retriever"].papers_meta)
        if "arxiv_text" not in collections:
            collections["arxiv_text"] = len(_state["retriever"].chunks_meta)

    return HealthResponse(
        status="healthy",
        collections=collections,
        db_papers=db_papers,
        uptime_seconds=round(time.time() - _state["start_time"], 1),
        cache_stats=query_cache.stats,
    )


@app.head("/health", include_in_schema=False)
async def health_check_head():
    """HEAD probe for health checks and uptime monitors."""
    return Response(status_code=200)


@app.get("/metrics/performance")
async def performance_metrics():
    """Project-wide performance metrics including historical p95 for /query."""
    rolling_request_latencies = list(_metrics["request_latencies_ms"])
    rolling_query_latencies = list(_metrics["query_latencies_ms"])
    cache_hit_latencies = list(_metrics["cache_hit_latencies_ms"])
    cache_miss_latencies = list(_metrics["cache_miss_latencies_ms"])
    historical_query_latencies = _historical_query_latencies()

    hit_summary = _latency_summary(cache_hit_latencies)
    miss_summary = _latency_summary(cache_miss_latencies)
    avg_reduction = max(0, miss_summary["avg_ms"] - hit_summary["avg_ms"])

    return {
        "uptime_seconds": round(time.time() - _state["start_time"], 1),
        "query_count": _state["query_count"],
        "rolling": {
            "requests": _latency_summary(rolling_request_latencies),
            "queries": _latency_summary(rolling_query_latencies),
            "window_size": {
                "requests": _metrics["request_latencies_ms"].maxlen,
                "queries": _metrics["query_latencies_ms"].maxlen,
            },
        },
        "lru_cache_impact": {
            "cache_hit_summary": hit_summary,
            "cache_miss_summary": miss_summary,
            "avg_latency_reduction_ms": round(avg_reduction, 1),
            "estimated_time_saved_ms": round(avg_reduction * hit_summary["count"], 1),
        },
        "historical_queries": _latency_summary(historical_query_latencies),
    }


@app.head("/metrics/performance", include_in_schema=False)
async def performance_metrics_head():
    """HEAD probe for performance metrics endpoint."""
    return Response(status_code=200)
