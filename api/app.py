"""
app.py — FastAPI application for the ArXiv RAG Assistant.

Endpoints:
    POST /query         — Hybrid retrieval + rerank + LLM answer generation
    GET  /paper/{id}    — Paper metadata lookup
    GET  /paper/{id}/similar — Find similar papers
    GET  /health        — Health check with basic metrics

Usage:
    conda run -n pytorch uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("DB_PATH", "data/arxiv_papers.db")
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

    def _make_key(self, query: str, top_k: int, category: str = None,
                  author: str = None, start_year: int = None) -> str:
        return f"{query.strip().lower()}|{top_k}|{category or ''}|{author or ''}|{start_year or ''}"

    def get(self, key: str):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._ttl:
                self._cache.move_to_end(key)
                return entry["data"]
            else:
                del self._cache[key]
        return None

    def set(self, key: str, data: dict):
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = {"data": data, "timestamp": time.time()}
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "max_size": self._max_size, "ttl": self._ttl}


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
    rerank_score: float = 0.0


class AnalyticsInfo(BaseModel):
    top_authors: list[dict] = []
    top_categories: list[dict] = []
    total_unique_papers: int = 0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    latency_ms: float
    retrieval_trace: dict
    analytics: AnalyticsInfo = AnalyticsInfo()
    cached: bool = False


class SimilarPaperInfo(BaseModel):
    paper_id: str
    title: str
    authors: str = ""
    categories: str = ""
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


class HealthResponse(BaseModel):
    status: str
    chroma_docs: int = 0
    db_papers: int = 0
    uptime_seconds: float = 0.0
    cache_stats: dict = {}


# ---------------------------------------------------------------------------
# Global state (loaded on startup)
# ---------------------------------------------------------------------------

_state = {
    "retriever": None,
    "start_time": time.time(),
    "query_count": 0,
}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes on startup."""
    log.info("Starting ArXiv RAG API...")
    _state["start_time"] = time.time()

    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from api.retrieval import HybridRetriever

        _state["retriever"] = HybridRetriever()
        log.info("HybridRetriever initialized successfully.")
    except Exception as e:
        log.error(f"Failed to initialize retriever: {e}")
        log.warning("API will start but /query endpoint will be unavailable.")

    yield

    log.info("Shutting down ArXiv RAG API...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ArXiv RAG Assistant",
    description="Hybrid RAG system for ArXiv papers with dense + BM25 retrieval and cross-encoder reranking.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="frontend")


# ---------------------------------------------------------------------------
# Prompt builder — intent-aware templates
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
2. Be fair and balanced — present strengths and weaknesses of each side.
3. Do NOT fabricate benchmark numbers. Only cite numbers found in the sources.

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
1. **Overview**: A 1-2 sentence summary of what is being compared. Start with double asterisks **like this**.
2. **Approach A**: Summary of the first approach — key mechanism, strengths.
3. **Approach B**: Summary of the second approach — key mechanism, strengths.
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
        "Never fabricate steps — if the sources don't cover something, say so."
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
    Generate an answer using Groq API (free tier, ultra-fast inference).
    Falls back to local extractive summary if GROQ_API_KEY is not set.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        return _generate_groq(prompt, groq_api_key, intent=intent)
    else:
        log.warning("GROQ_API_KEY not set — using local extractive summary.")
        return _generate_local_fallback(prompt)


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
        return f"[Groq API error: {e}] — Falling back to source extraction.\n\n" + _generate_local_fallback(prompt)


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


def _generate_local_fallback(prompt: str) -> str:
    """Fallback: extract and summarize from passages without LLM."""
    lines = prompt.split("\n")
    sources_section = []
    in_context = False

    for line in lines:
        if line.strip().startswith("Context:"):
            in_context = True
            continue
        if line.strip().startswith("Source References:"):
            break
        if in_context and line.strip():
            sources_section.append(line.strip())

    if not sources_section:
        return "I couldn't find relevant information in the retrieved sources to answer this question."

    question_line = [l for l in lines if l.strip().startswith("Question:")]
    query = question_line[0].replace("Question:", "").strip() if question_line else ""

    answer_parts = [f"Based on the retrieved ArXiv papers, here is what I found regarding '{query}':\n"]
    seen = set()
    for text in sources_section[:5]:
        if text not in seen and len(text) > 20:
            seen.add(text)
            answer_parts.append(f"• {text}")

    return "\n".join(answer_parts)


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve frontend."""
    index_file = _frontend_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "ArXiv RAG API — visit /docs for Swagger UI"}


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

    # Check cache
    cache_key = query_cache._make_key(
        request.query, request.top_k, request.category,
        request.author, request.start_year
    )
    cached_result = query_cache.get(cache_key)
    if cached_result:
        cached_result["cached"] = True
        log_query(request.query, cached_result)
        return QueryResponse(**cached_result)

    t0 = time.time()
    _state["query_count"] += 1

    # Classify query intent (drives retrieval, compression, and prompt selection)
    from api.retrieval import classify_query_intent
    intent = classify_query_intent(request.query)

    # Retrieve with filters + intent
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

    # Context compression — intent-aware
    t_compress = time.time()
    compressed_context = _state["retriever"].compress_context(
        request.query, passages, intent=intent
    )
    trace["compress_ms"] = round((time.time() - t_compress) * 1000, 1)

    # Generate answer — intent-aware prompt and temperature
    t_gen = time.time()
    prompt = build_prompt(request.query, compressed_context, passages, intent=intent)
    answer = generate_answer(prompt, intent=intent)
    trace["generation_ms"] = round((time.time() - t_gen) * 1000, 1)

    total_ms = round((time.time() - t0) * 1000, 1)

    # Build response
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

    # Cache the result
    cache_data = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "latency_ms": total_ms,
        "retrieval_trace": trace,
        "analytics": analytics,
        "cached": False,
    }
    query_cache.set(cache_key, cache_data)

    # Log query
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

    # Classify query intent
    from api.retrieval import classify_query_intent
    intent = classify_query_intent(request.query)

    # Retrieve with filters + intent
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

    # Context compression — intent-aware
    compressed_context = _state["retriever"].compress_context(
        request.query, passages, intent=intent
    )

    # Build intent-aware prompt
    prompt = build_prompt(request.query, compressed_context, passages, intent=intent)

    # Build sources list
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
            "arxiv_api_fallback": trace.get("arxiv_api_fallback", False),
            "intent": intent,
        })
        yield f"data: {meta_payload}\n\n"

        # Stream answer tokens — intent-aware
        try:
            for token in _generate_groq_stream(prompt, groq_api_key, intent=intent):
                token_payload = json.dumps({"type": "token", "content": token})
                yield f"data: {token_payload}\n\n"
        except Exception as e:
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

        # Done event
        total_ms = round((time.time() - t0) * 1000, 1)
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
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="Database not found.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

    return PaperResponse(
        paper_id=row["paper_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=row["authors"],
        categories=row["categories"],
        pdf_url=row["pdf_url"],
        published=row["published"],
    )


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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with basic metrics."""
    chroma_docs = 0
    db_papers = 0

    if _state["retriever"] is not None:
        try:
            chroma_docs = _state["retriever"].collection.count()
        except Exception:
            pass

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            db_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            conn.close()
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        chroma_docs=chroma_docs,
        db_papers=db_papers,
        uptime_seconds=round(time.time() - _state["start_time"], 1),
        cache_stats=query_cache.stats,
    )
