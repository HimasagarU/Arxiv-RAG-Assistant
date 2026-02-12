"""
app.py — FastAPI application for the ArXiv RAG Assistant.

Endpoints:
    POST /query         — Hybrid retrieval + rerank + LLM answer generation
    GET  /paper/{id}    — Paper metadata lookup
    GET  /health        — Health check with basic metrics

Usage:
    conda run -n pytorch uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class SourceInfo(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    authors: str = ""
    categories: str = ""
    chunk_text: str
    rerank_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    latency_ms: float
    retrieval_trace: dict


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
        # Import here to avoid loading models at import time
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
    version="1.0.0",
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
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, passages: list[dict]) -> str:
    """Build a RAG prompt using retrieved sources."""
    sources_text = ""
    for i, p in enumerate(passages):
        sources_text += f"\n[Source {i+1}: {p['chunk_id']}]\n"
        sources_text += f"Title: {p.get('title', 'N/A')}\n"
        sources_text += f"{p['chunk_text']}\n"

    prompt = f"""You are a helpful research assistant specializing in AI and Machine Learning papers from ArXiv.

Answer the following question using ONLY the provided sources. Be concise and accurate.
Cite your sources using [Source: chunk_id] format.
If the sources don't contain enough information to answer, say so explicitly.

Sources:
{sources_text}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt: str) -> str:
    """
    Generate an answer using Groq API (free tier, ultra-fast inference).
    Falls back to local extractive summary if GROQ_API_KEY is not set.
    
    Model: llama-3.3-70b-versatile (free on Groq)
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        return _generate_groq(prompt, groq_api_key)
    else:
        log.warning("GROQ_API_KEY not set — using local extractive summary. "
                    "Set GROQ_API_KEY in .env for LLM-powered answers.")
        return _generate_local_fallback(prompt)


def _generate_groq(prompt: str, api_key: str) -> str:
    """Generate answer using Groq API with Llama 3.3 70B."""
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant specializing in AI and Machine Learning papers from ArXiv. Answer concisely and cite sources using [Source: chunk_id] format.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9,
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return f"[Groq API error: {e}] — Falling back to source extraction.\n\n" + _generate_local_fallback(prompt)


def _generate_local_fallback(prompt: str) -> str:
    """Fallback: extract and summarize from passages without LLM."""
    lines = prompt.split("\n")
    sources_section = []
    in_sources = False
    
    for line in lines:
        if line.strip().startswith("[Source"):
            in_sources = True
            continue
        if line.strip().startswith("Question:"):
            break
        if in_sources and line.strip() and not line.strip().startswith("Title:"):
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
    """
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Please ensure indexes are built.",
        )

    t0 = time.time()
    _state["query_count"] += 1

    # Retrieve
    result = _state["retriever"].retrieve(request.query, top_n=request.top_k)
    passages = result["passages"]
    trace = result["trace"]

    # Generate answer
    t_gen = time.time()
    prompt = build_prompt(request.query, passages)
    answer = generate_answer(prompt)
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
    }

    # Log query
    log_query(request.query, {
        "latency_ms": total_ms,
        "retrieval_trace": trace,
        "sources": [s.model_dump() for s in sources],
    })

    return QueryResponse(**response_data)


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
    )
