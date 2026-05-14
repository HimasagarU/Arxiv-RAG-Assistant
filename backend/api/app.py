"""
app.py ” FastAPI application for the ArXiv RAG Assistant.

Endpoints:
    POST /query         ” Hybrid retrieval + rerank + LLM answer generation
    GET  /paper/{id}    ” Paper metadata lookup
    GET  /paper/{id}/similar ” Find similar papers
    GET  /health        ” Health check with basic metrics

Usage:
    conda run -n pytorch uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

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


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid %s=%s; using default %s", name, raw, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid %s=%s; using default %s", name, raw, default)
        return default


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = _get_env_float("GEMINI_TEMPERATURE", 0.2)
GEMINI_MAX_OUTPUT_TOKENS = _get_env_int("GEMINI_MAX_OUTPUT_TOKENS", 8192)
GEMINI_TOP_P = _get_env_float("GEMINI_TOP_P", 0.9)
GEMINI_RPM = _get_env_int("GEMINI_RPM", 5)
GEMINI_RPD = _get_env_int("GEMINI_RPD", 0)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = _get_env_float("GROQ_TEMPERATURE", GEMINI_TEMPERATURE)
GROQ_MAX_OUTPUT_TOKENS = _get_env_int("GROQ_MAX_OUTPUT_TOKENS", 2048)
GROQ_TOP_P = _get_env_float("GROQ_TOP_P", GEMINI_TOP_P)
GENERATION_CONTEXT_TOP_N = _get_env_int("GENERATION_CONTEXT_TOP_N", 10)


def _error_text(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}".lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(
        token in text
        for token in (
            "rate limit",
            "too many requests",
            "resourceexhausted",
            "resource exhausted",
            "quota",
            "429",
            "daily request cap reached",
        )
    )


def _is_gemini_unavailable_error(exc: Exception) -> bool:
    """Return True when the Gemini SDK itself is missing or unavailable."""
    text = _error_text(exc)
    return (
        isinstance(exc, (ModuleNotFoundError, ImportError))
        or "google.generativeai" in text
        or "google.genai" in text
    )


class GenerationSurface(str, Enum):
    """Explicit routing targets for answer generation."""

    PUBLIC = "public"
    CHAT = "chat"
    DOCUMENT_CHAT = "document_chat"


@dataclass(frozen=True)
class GenerationPolicy:
    """Model routing policy for a given request surface."""

    primary: str
    fallback: Optional[str] = None
    fallback_on_rate_limit: bool = False


_GENERATION_POLICIES: dict[GenerationSurface, GenerationPolicy] = {
    GenerationSurface.PUBLIC: GenerationPolicy(primary="groq"),
    GenerationSurface.CHAT: GenerationPolicy(
        primary="gemini",
        fallback="groq",
        fallback_on_rate_limit=True,
    ),
    GenerationSurface.DOCUMENT_CHAT: GenerationPolicy(
        primary="gemini",
        fallback="groq",
        fallback_on_rate_limit=True,
    ),
}


class GeminiRateLimiter:
    """Thread-safe RPM/RPD limiter that sleeps between Gemini calls."""

    def __init__(self, rpm: int, rpd: int | None = None):
        self._min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._rpd = rpd if rpd and rpd > 0 else None
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._day_key = time.strftime("%Y-%m-%d", time.gmtime())
        self._calls_today = 0

    def _rollover_day(self) -> None:
        day_key = time.strftime("%Y-%m-%d", time.gmtime())
        if day_key != self._day_key:
            self._day_key = day_key
            self._calls_today = 0

    def acquire(self) -> None:
        if self._min_interval <= 0 and self._rpd is None:
            return

        now = time.time()
        with self._lock:
            self._rollover_day()
            if self._rpd is not None and self._calls_today >= self._rpd:
                raise RuntimeError(
                    "Gemini daily request cap reached. Set GEMINI_RPD=0 to disable."
                )
            target = max(now, self._last_call + self._min_interval)
            self._last_call = target
            self._calls_today += 1

        wait = target - now
        if wait > 0:
            time.sleep(wait)


class GeminiClient:
    """Shared Gemini client wrapper with rate limiting."""

    def __init__(
        self,
        api_key: str,
        model: str,
        rate_limiter: GeminiRateLimiter,
        temperature: float,
        max_output_tokens: int,
        top_p: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._rate_limiter = rate_limiter
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    def _client(self):
        from google import genai

        return genai.Client(api_key=self._api_key)

    def _generation_config(self, system_prompt: str, temperature: float):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=self._max_output_tokens,
            top_p=self._top_p,
        )

    def _resolve_temperature(self, intent: str) -> float:
        if intent == "explanatory":
            return min(self._temperature, 0.1)
        return self._temperature

    def generate(self, prompt: str, system_prompt: str, intent: str) -> str:
        self._rate_limiter.acquire()
        client = self._client()
        temp = self._resolve_temperature(intent)
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._generation_config(system_prompt, temp),
        )
        return response.text or ""

    def stream(self, prompt: str, system_prompt: str, intent: str):
        self._rate_limiter.acquire()
        client = self._client()
        temp = self._resolve_temperature(intent)
        stream = client.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=self._generation_config(system_prompt, temp),
        )
        for chunk in stream:
            try:
                text = chunk.text
                if text:
                    yield text
            except ValueError:
                # Catch safety or empty text exceptions thrown by the property accessor
                pass


class GroqClient:
    """Shared Groq client wrapper used as a fallback for Gemini rate limits."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        top_p: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    def _client(self):
        from groq import Groq

        return Groq(api_key=self._api_key)

    def _resolve_temperature(self, intent: str) -> float:
        if intent == "explanatory":
            return min(self._temperature, 0.1)
        return self._temperature

    def generate(self, prompt: str, system_prompt: str, intent: str, max_tokens: int | None = None) -> str:
        client = self._client()
        temp = self._resolve_temperature(intent)
        mt = self._max_output_tokens if max_tokens is None else int(max_tokens)
        mt = max(1, min(mt, 8192))
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self._model,
            temperature=temp,
            max_tokens=mt,
            top_p=self._top_p,
        )
        return chat_completion.choices[0].message.content or ""

    def stream(self, prompt: str, system_prompt: str, intent: str):
        client = self._client()
        temp = self._resolve_temperature(intent)
        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self._model,
            temperature=temp,
            max_tokens=self._max_output_tokens,
            top_p=self._top_p,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


_gemini_rate_limiter = GeminiRateLimiter(
    GEMINI_RPM,
    GEMINI_RPD if GEMINI_RPD > 0 else None,
)
_gemini_client_lock = threading.Lock()
_gemini_client: Optional[GeminiClient] = None
_groq_client_lock = threading.Lock()
_groq_client: Optional[GroqClient] = None


def _get_gemini_client() -> GeminiClient:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini answer generation.")

    global _gemini_client
    with _gemini_client_lock:
        if (
            _gemini_client is None
            or _gemini_client.api_key != api_key
            or _gemini_client.model != GEMINI_MODEL
        ):
            _gemini_client = GeminiClient(
                api_key=api_key,
                model=GEMINI_MODEL,
                rate_limiter=_gemini_rate_limiter,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                top_p=GEMINI_TOP_P,
            )
    return _gemini_client


def _get_groq_client() -> GroqClient:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini rate limit reached, but GROQ_API_KEY is not set for fallback generation."
        )

    global _groq_client
    with _groq_client_lock:
        if (
            _groq_client is None
            or _groq_client.api_key != api_key
            or _groq_client.model != GROQ_MODEL
        ):
            _groq_client = GroqClient(
                api_key=api_key,
                model=GROQ_MODEL,
                temperature=GROQ_TEMPERATURE,
                max_output_tokens=GROQ_MAX_OUTPUT_TOKENS,
                top_p=GROQ_TOP_P,
            )
    return _groq_client


def _normalize_generation_surface(surface: GenerationSurface | str) -> GenerationSurface:
    if isinstance(surface, GenerationSurface):
        return surface
    return GenerationSurface(surface)


def _get_generation_policy(surface: GenerationSurface | str) -> GenerationPolicy:
    normalized = _normalize_generation_surface(surface)
    return _GENERATION_POLICIES[normalized]


def _generate_with_gemini(prompt: str, system_prompt: str, intent: str) -> str:
    client = _get_gemini_client()
    return client.generate(prompt, system_prompt, intent)


def _generate_with_groq(prompt: str, system_prompt: str, intent: str) -> str:
    client = _get_groq_client()
    return client.generate(prompt, system_prompt, intent)


def _stream_with_gemini(prompt: str, system_prompt: str, intent: str):
    client = _get_gemini_client()
    yield from client.stream(prompt, system_prompt, intent)


def _stream_with_groq(prompt: str, system_prompt: str, intent: str):
    client = _get_groq_client()
    yield from client.stream(prompt, system_prompt, intent)


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
    top_k: int = Field(default=10, ge=1, le=20, description="Number of results to return")
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

    env = (os.getenv("ENVIRONMENT", "") or "").strip().lower()
    if env == "production":
        jwt_secret = (os.getenv("JWT_SECRET_KEY", "") or "").strip()
        if not jwt_secret or jwt_secret == "change-me-in-production":
            log.error("JWT_SECRET_KEY must be set to a strong secret when ENVIRONMENT=production.")
            raise RuntimeError("Unsafe JWT_SECRET_KEY in production.")

    def init_retriever():
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if os.getenv("SKIP_ARTIFACT_FETCH", "").lower() not in ("1", "true", "yes"):
                from api.fetch_data import fetch_and_extract

                log.info("Running data bootstrapper...")
                fetch_and_extract()
            else:
                log.info("SKIP_ARTIFACT_FETCH set — skipping R2 artifact download.")
            
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
    description="Hybrid RAG system for ArXiv papers with Qdrant Cloud dense retrieval, BM25 lexical retrieval, cross-encoder reranking, HyDE, query expansion, and MMR diversity filtering.",
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
# Prompt builder ” intent-aware templates
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
    elif intent == "evidence":
        return _build_evidence_prompt(query, compressed_context, sources_block)
    else:
        return _build_general_prompt(query, compressed_context, sources_block)


def _build_explanatory_prompt(query: str, context: str, sources: str) -> str:
    return f"""You are an expert AI/ML research assistant. A student has asked an explanatory question.
Your job is to give a clear, accurate, mechanism-level explanation grounded in the source passages.

CRITICAL RULES:
1. Every key claim must be supported by at least one source. Use citations [1], [2], etc.
2. Only make claims explicitly supported by the retrieved passages.
3. Distinguish clearly between established findings, hypotheses, and your own synthesis.
4. If the sources lack evidence for a key part of the explanation, explicitly state your uncertainty.
5. Do NOT write a literature review. Write a direct explanation of the mechanism/concept.
6. Use the Feynman Technique: be clear, but do NOT omit important technical details.
7. Avoid blending unrelated retrieved concepts into one narrative.

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
1. **Definition**: A clear 1-2 sentence definition of the concept. Start with double asterisks **like this**.
2. **How It Works**: A step-by-step explanation of the mechanism or pipeline. Number each step.
3. **Why It Works**: Explain the intuition behind why this approach is effective.
4. **Limitations**: Briefly note known limitations or failure modes based on evidence.
5. **Evidence Gaps / Inference**: Explicitly call out anything that is weak or partially supported by the retrieved passages.
6. **References**: List the numbered source titles.

---

SOURCE PASSAGES:
{context}

AVAILABLE SOURCES:
{sources}

QUESTION: {query}

ANSWER:"""


def _build_comparative_prompt(query: str, context: str, sources: str) -> str:
    return f"""You are an expert AI/ML research assistant. A student wants to compare two or more approaches.
Your job is to give a balanced, evidence-based comparison grounded strictly in the source passages.

CRITICAL RULES:
1. Every claim must be supported by at least one source. Use citations [1], [2], etc.
2. Be fair and balanced ” present strengths and weaknesses of each side based purely on evidence.
3. Do NOT fabricate benchmark numbers. Only cite numbers found in the sources.
4. Distinguish between empirical facts, author hypotheses, and speculation.
5. Explicitly state uncertainty when comparative evidence is weak.

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
1. **Overview**: A 1-2 sentence summary of what is being compared. Start with double asterisks **like this**.
2. **Approach A**: Summary of the first approach ” key mechanism, strengths.
3. **Approach B**: Summary of the second approach ” key mechanism, strengths.
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


def _build_evidence_prompt(query: str, context: str, sources: str) -> str:
    return f"""You are an expert AI/ML research assistant focusing on mechanistic interpretability and scientific rigor.
Your job is to provide an evidence-grounded answer based strictly on the provided passages.

CRITICAL RULES:
1. Only make claims supported by retrieved passages. Provide exact numbered citations [1], [2].
2. Attribute findings to papers explicitly whenever possible (e.g., "Olsson et al. observed..."). Avoid generic "Researchers found...".
3. Explicitly state uncertainty when evidence is weak or absent.
4. Clearly distinguish between established findings, hypotheses, interpretations, and speculation.
5. Avoid blending unrelated retrieved concepts into one narrative. Treat each piece of evidence rigorously.

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
1. **Core Finding**: A bold 1-2 sentence statement summarizing the strongest evidence answering the question. Start with double asterisks **like this**.
2. **Empirical Evidence**: List the specific experiments, ablations, or results that support the core finding. Provide explicit attribution.
3. **Interpretations & Hypotheses**: Describe the authors' interpretations of these results.
4. **Uncertainty & Gaps**: State clearly what the evidence does *not* show or where it is weak.
5. **References**: List the numbered source titles.

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
3. Only make claims supported by the retrieved passages.
4. Distinguish clearly between established findings, hypotheses, and what you infer from them.
5. If information is insufficient or evidence is weak, say so honestly rather than speculating.
6. Attribute findings explicitly to papers or authors whenever possible.
7. Avoid blending unrelated retrieved concepts into one narrative.

STRUCTURE YOUR ANSWER LIKE THIS:
1. **Summary**: A bold 1-2 sentence executive summary answering the core question. Start with double asterisks **like this**.
2. **Key Findings**: The most important technical insights from the sources with explicit attribution.
3. **Mechanism / Evidence**: Explain the main mechanism or reasoning and cite the strongest evidence.
4. **Implications / Limitations**: What follows from the evidence, and what remains uncertain.
5. **References**: Numbered source titles.

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
        "Never fabricate steps — if the sources don't cover something, say so. "
        "CRITICAL: Keep your answer concise, complete, and strictly under 1500 words to prevent truncation."
    ),
    "comparative": (
        "You are an expert AI/ML research assistant. When comparing approaches, "
        "be balanced and evidence-based. Present both sides fairly. "
        "Use numbered citations [1], [2]. Start with a bold overview. "
        "CRITICAL: Keep your answer concise, complete, and strictly under 1500 words to prevent truncation."
    ),
    "evidence": (
        "You are an expert AI/ML research assistant focusing on scientific rigor. "
        "Ground every claim strictly in retrieved evidence, use precise citations, "
        "and explicitly state uncertainty if evidence is weak. "
        "CRITICAL: Keep your answer concise, complete, and strictly under 1500 words to prevent truncation."
    ),
    "default": (
        "You are an expert AI/ML research assistant. Give thorough, well-structured answers "
        "using the Feynman Technique. Use numbered citations [1], [2] to reference sources. "
        "Start with a bold executive summary. Ground every claim in the source passages. "
        "CRITICAL: Keep your answer concise, complete, and strictly under 1500 words to prevent truncation."
    ),
}


def get_system_prompt(intent: str = "discovery") -> str:
    """Get the system prompt for a given intent."""
    if intent in _SYSTEM_PROMPTS:
        return _SYSTEM_PROMPTS[intent]
    return _SYSTEM_PROMPTS["default"]


def get_context_size(intent: str) -> int:
    """Return intent-aware context limits (number of chunks)."""
    sizes = {
        "explanatory": 6,
        "technical": 4,
        "evidence": 4,
        "comparative": 6,
        "sota": 8,
    }
    return sizes.get(intent, GENERATION_CONTEXT_TOP_N)


def verify_answer(answer: str, passages: list[dict]) -> str:
    """Simple heuristic: if core claim terminology is absent from retrieved evidence, warn about weak grounding."""
    import re
    # Extract long words as a proxy for terminology
    words = re.findall(r'\b[A-Za-z]{6,}\b', answer.lower())
    if not words:
        return answer
    
    context_text = " ".join([p.get("chunk_text", "").lower() for p in passages])
    matched = sum(1 for w in words if w in context_text)
    overlap = matched / len(words)
    
    if overlap < 0.25:
        warning = "\n\n**Warning**: The generated response contains terminology not heavily present in the retrieved evidence. Please verify against the source chunks."
        if warning not in answer:
            return answer + warning
    return answer


def get_retriever():
    """FastAPI dependency for the initialized retriever."""
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Please ensure indexes are built.",
        )
    return _state["retriever"]


def get_intent_classifier():
    """FastAPI dependency for query intent classification."""
    from api.retrieval import classify_query_intent

    return classify_query_intent


def get_answer_generator():
    """FastAPI dependency for answer generation."""
    return generate_answer


def generate_answer(
    prompt: str,
    intent: str = "discovery",
    surface: GenerationSurface | str = GenerationSurface.CHAT,
) -> str:
    """
    Generate an answer using the configured model route for the request surface.
    Public landing queries use Groq directly.
    Chat surfaces use Gemini first, then Groq on Gemini rate limits.
    """
    system_prompt = get_system_prompt(intent)
    policy = _get_generation_policy(surface)

    if policy.primary == "groq":
        try:
            return _generate_with_groq(prompt, system_prompt, intent)
        except Exception as e:
            log.error("Groq API error on public generation path: %s", e)
            raise

    try:
        return _generate_with_gemini(prompt, system_prompt, intent)
    except Exception as e:
        can_fallback = (
            policy.fallback_on_rate_limit
            and policy.fallback is not None
            and (_is_rate_limit_error(e) or _is_gemini_unavailable_error(e))
        )
        if not can_fallback:
            log.error("Gemini API error: %s", e)
            raise

        if _is_gemini_unavailable_error(e):
            log.warning("Gemini SDK unavailable; falling back to Groq for answer generation.")
        else:
            log.warning("Gemini rate limit reached; falling back to Groq for answer generation.")
        try:
            return _generate_with_groq(prompt, system_prompt, intent)
        except Exception as groq_error:
            log.error("Groq fallback failed after Gemini failure: %s", groq_error)
            raise


def stream_generate_answer(
    prompt: str,
    intent: str = "discovery",
    surface: GenerationSurface | str = GenerationSurface.CHAT,
):
    """Public iterator for token streaming (SSE / WebSocket)."""
    yield from _generate_answer_stream(prompt, intent=intent, surface=surface)


def generate_hyde_excerpt(query: str, intent: str = "discovery") -> Optional[str]:
    """Optional HyDE hypothetical passage for dense retrieval (Groq, short)."""
    from api.feature_flags import env_bool

    if not env_bool("ENABLE_HYDE", False):
        return None
    if intent not in ("discovery", "explanatory"):
        return None
    if len(query.split()) < 6:
        return None
    try:
        client = _get_groq_client()
    except Exception as exc:
        log.debug("HyDE skipped (no Groq): %s", exc)
        return None
    system = "You write only dense technical prose. No titles or disclaimers."
    user = (
        f"Research question:\n{query}\n\n"
        "Write 2-4 sentences of a hypothetical paper excerpt that would contain the answer. "
        "Use field-appropriate technical terminology.\n\nExcerpt:"
    )
    try:
        text = client.generate(user, system, "discovery", max_tokens=220)
    except Exception as exc:
        log.warning("HyDE generation failed: %s", exc)
        return None
    text = (text or "").strip()
    return text or None


def compress_context_with_llm(query: str, passages: list[dict], max_chars: int = 12000) -> Optional[str]:
    """Optional Groq compression when retrieved context is very long."""
    from api.feature_flags import env_bool

    if not (
        env_bool("ENABLE_LLM_CONTEXT_COMPRESS", False)
        or env_bool("ENABLE_CONTEXT_COMPRESSION", False)
    ):
        return None
    if not passages:
        return None
    try:
        client = _get_groq_client()
    except Exception:
        return None
    blocks = []
    for i, p in enumerate(passages, 1):
        title = p.get("title", "")
        body = (p.get("chunk_text", "") or "")[:4000]
        blocks.append(f"[{i}] {title}\n{body}")
    packed = "\n\n".join(blocks)
    if len(packed) <= max_chars:
        return None
    system = (
        "Extract only sentences directly relevant to the query. "
        "Keep citation markers like [1], [2] matching chunk indices. Plain text only."
    )
    user = f"Query: {query}\n\nChunks:\n{packed[:120000]}\n\nCompressed evidence:"
    try:
        out = client.generate(user, system, "discovery", max_tokens=min(2048, max_chars // 4))
    except Exception as exc:
        log.warning("LLM context compression failed: %s", exc)
        return None
    out = (out or "").strip()
    return out or None


def expand_query_variants_llm(query: str, max_variants: int = 1) -> list[str]:
    """Optional Groq paraphrases for retrieval (conservative; max 3 new strings)."""
    from api.feature_flags import env_bool

    if not (env_bool("ENABLE_QUERY_EXPANSION_LLM", False) or env_bool("ENABLE_QUERY_EXPANSION", False)):
        return []
    try:
        client = _get_groq_client()
    except Exception:
        return []
    system = "You output numbered alternative search queries only, one per line. No explanations."
    user = (
        f"Original research query:\n{query}\n\n"
        f"Write up to {max_variants} shorter alternative search queries using different terminology.\n"
        "Format:\n1. ...\n2. ...\n3. ..."
    )
    try:
        raw = client.generate(user, system, "discovery", max_tokens=180)
    except Exception as exc:
        log.warning("LLM query expansion failed: %s", exc)
        return []
    lines = re.findall(r"^\s*\d+\.\s*(.+)$", raw or "", re.MULTILINE)
    out: list[str] = []
    for line in lines[:max_variants]:
        q = line.strip()
        if q and q.lower() != query.lower():
            out.append(q)
    return out


def _generate_answer_stream(
    prompt: str,
    intent: str = "discovery",
    surface: GenerationSurface | str = GenerationSurface.CHAT,
):
    """Streaming generator for the configured model route."""
    system_prompt = get_system_prompt(intent)
    policy = _get_generation_policy(surface)

    if policy.primary == "groq":
        try:
            yield from _stream_with_groq(prompt, system_prompt, intent)
            return
        except Exception as e:
            log.error("Groq streaming error on public generation path: %s", e)
            raise

    yielded_any = False
    try:
        for chunk in _stream_with_gemini(prompt, system_prompt, intent):
            yielded_any = True
            yield chunk
    except Exception as e:
        can_fallback = (
            not yielded_any
            and policy.fallback_on_rate_limit
            and policy.fallback is not None
            and (_is_rate_limit_error(e) or _is_gemini_unavailable_error(e))
        )
        if not can_fallback:
            log.error("Gemini streaming error: %s", e)
            raise

        if _is_gemini_unavailable_error(e):
            log.warning("Gemini SDK unavailable; falling back to Groq for streaming generation.")
        else:
            log.warning("Gemini rate limit reached; falling back to Groq for streaming generation.")
        try:
            yield from _stream_with_groq(prompt, system_prompt, intent)
        except Exception as groq_error:
            log.error("Groq streaming fallback failed after Gemini failure: %s", groq_error)
            raise


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------

def log_query(query: str, response_data: dict):
    """Save query trace to JSON log file with detailed stage timing."""
    try:
        trace = response_data.get("retrieval_trace", {})
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "latency_ms": response_data.get("latency_ms", 0),
            "stages_ms": {
                "dense": trace.get("dense_ms", 0),
                "lexical": trace.get("lex_ms", 0),
                "merge": trace.get("merge_ms", 0),
                "rerank": trace.get("rerank_ms", 0),
                "compress": trace.get("compress_ms", 0),
                "generation": trace.get("generation_ms", 0),
            },
            "intent": trace.get("intent", "unknown"),
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
async def query_endpoint(
    request: QueryRequest,
    retriever=Depends(get_retriever),
    classify_query_intent=Depends(get_intent_classifier),
    answer_generator=Depends(get_answer_generator),
):
    """
    Hybrid retrieval + rerank + answer generation.
    Supports metadata filtering by category, author, and publication year.
    Results are cached for 5 minutes.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="Public query generation requires GROQ_API_KEY to be set.",
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

    intent = classify_query_intent(request.query)

    target_top_n = min(request.top_k, get_context_size(intent))
    result = retriever.retrieve(
        request.query,
        top_n=target_top_n,
        category=request.category,
        author=request.author,
        start_year=request.start_year,
        intent=intent,
    )
    passages = result["passages"]
    trace = result["trace"]
    analytics = result.get("analytics", {})

    t_compress = time.time()
    compressed_context = retriever.compress_context(
        request.query, passages, intent=intent
    )
    trace["compress_ms"] = round((time.time() - t_compress) * 1000, 1)

    t_gen = time.time()
    prompt = build_prompt(request.query, compressed_context, passages, intent=intent)
    answer = answer_generator(prompt, intent=intent, surface=GenerationSurface.PUBLIC)
    answer = verify_answer(answer, passages)
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
            detail="Public streaming requires GROQ_API_KEY to be set.",
        )

    t0 = time.time()
    _state["query_count"] += 1

    from api.retrieval import classify_query_intent
    intent = classify_query_intent(request.query)

    target_top_n = min(request.top_k, get_context_size(intent))
    result = _state["retriever"].retrieve(
        request.query,
        top_n=target_top_n,
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
            for token in _generate_answer_stream(
                prompt,
                intent=intent,
                surface=GenerationSurface.PUBLIC,
            ):
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
async def get_paper(paper_id: str, retriever=Depends(get_retriever)):
    """Look up paper metadata by ArXiv ID."""
    paper_info = retriever.papers_meta.get(paper_id)
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
async def get_similar_papers(
    paper_id: str,
    top_n: int = 5,
    retriever=Depends(get_retriever),
):
    """Find papers similar to the given paper."""
    similar = retriever.find_similar_papers(paper_id, top_n=top_n)

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
