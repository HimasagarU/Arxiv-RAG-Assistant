"""
ingest_arxiv.py — Seed-driven ArXiv ingestion for mechanistic interpretability.

Supports three ingestion modes:
  1. seed    — Ingest curated seed papers by arXiv ID
  2. keyword — Run keyword queries on arXiv with relevance filtering
  3. enrich  — Download PDFs and extract full text for existing papers

Usage:
    conda run -n pytorch python ingest/ingest_arxiv.py --mode seed
    conda run -n pytorch python ingest/ingest_arxiv.py --mode keyword
    conda run -n pytorch python ingest/ingest_arxiv.py --mode enrich --pdf-timeout 60
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import feedparser
import requests
from dotenv import load_dotenv
from psycopg import OperationalError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db
from storage.local_pdf_store import LocalPDFStore
from ingest.citation_expander import is_relevant, assign_layer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_REQUEST_TIMEOUT = int(os.getenv("ARXIV_REQUEST_TIMEOUT", "60"))
PDF_TIMEOUT = int(os.getenv("PDF_TIMEOUT", "60"))
MAX_FULLTEXT_CHARS = int(os.getenv("MAX_FULLTEXT_CHARS", "200000"))
CORPUS_TARGET_MAX = int(os.getenv("CORPUS_TARGET_MAX", "5000"))
DB_WRITE_RETRIES = int(os.getenv("DB_WRITE_RETRIES", "3"))
KEYWORD_STATE_PATH = Path(os.getenv("KEYWORD_STATE_PATH", "data/keyword_ingest_state.json"))
MAX_PDF_DOWNLOAD_RETRIES = int(os.getenv("PDF_DOWNLOAD_RETRIES", "2"))

# Local PDF cache directory
PDF_CACHE_DIR = Path("data/pdfs")
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Seed paper definitions
# ---------------------------------------------------------------------------

SEED_PAPERS = [
    {
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "layer": "prerequisite",
        "year": 2017,
    },
    {
        "paper_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "layer": "prerequisite",
        "year": 2018,
    },
    {
        "paper_id": "2104.08696",
        "title": "Knowledge Neurons in Pretrained Transformers",
        "layer": "foundation",
        "year": 2021,
    },
    {
        "paper_id": "2209.11895",
        "title": "In-context Learning and Induction Heads",
        "layer": "foundation",
        "year": 2022,
    },
    {
        "paper_id": "2211.00593",
        "title": "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small",
        "layer": "core",
        "year": 2022,
    },
    {
        "paper_id": "2202.05262",
        "title": "Locating and Editing Factual Associations in GPT",
        "layer": "core",
        "year": 2022,
    },
    {
        "paper_id": "2304.14997",
        "title": "Towards Automated Circuit Discovery for Mechanistic Interpretability",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2312.04782",
        "title": "Sparse Probing Finds Interpretable Features in Large Language Models",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2310.01405",
        "title": "Representation Engineering: A Top-Down Approach to AI Transparency",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2310.02207",
        "title": "Language Models Represent Space and Time",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2309.08600",
        "title": "Sparse Autoencoders Find Highly Interpretable Features in Language Models",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2305.01610",
        "title": "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2301.12971",
        "title": "Discovering Latent Knowledge in Language Models Without Supervision",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2310.06824",
        "title": "Neurons in Large Language Models: Dead, N-gram, Positional",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2403.19647",
        "title": "Improving Activation Steering in Language Models with SAE-Based Feature Decomposition",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2209.10652",
        "title": "Toy Models of Superposition",
        "layer": "core",
        "year": 2022,
    },
    {
        "paper_id": "2408.05451",
        "title": "Mathematical Models of Computation in Superposition",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2303.14151",
        "title": "Double Descent Demystified: Identifying, Interpreting & Comparing Superposition",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2410.13928",
        "title": "Automatically Interpreting Millions of Features in Large Language Models",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2411.10397",
        "title": "Features that Make a Difference: Leveraging Gradients for Improved Dictionary Learning",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2412.06410",
        "title": "BatchTopK Sparse Autoencoders",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2502.17332",
        "title": "Tokenized SAEs: Disentangling SAE Reconstructions",
        "layer": "latest",
        "year": 2025,
    },
    {
        "paper_id": "2503.17547",
        "title": "Learning Multi-Level Features with Matryoshka Sparse Autoencoders",
        "layer": "latest",
        "year": 2025,
    },
    {
        "paper_id": "2304.05969",
        "title": "Localizing Model Behavior with Path Patching",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2309.16042",
        "title": "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods",
        "layer": "core",
        "year": 2023,
    },
    {
        "paper_id": "2404.15255",
        "title": "How to use and interpret activation patching",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2409.10559",
        "title": "Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers",
        "layer": "latest",
        "year": 2024,
    },
    {
        "paper_id": "2405.07987",
        "title": "The Platonic Representation Hypothesis",
        "layer": "latest",
        "year": 2024,
    },
]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Clean up whitespace in text while preserving structure."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_text(text: str) -> str:
    """Remove problematic characters."""
    if not text:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def extract_arxiv_id(entry_id: str) -> str:
    """Extract the clean arXiv ID from an entry URL."""
    match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    if match:
        return match.group(1)
    parts = entry_id.split("/abs/")
    if len(parts) > 1:
        return parts[-1].split("v")[0]
    return entry_id


def is_valid_arxiv_id(paper_id: str) -> bool:
    return bool(re.match(r"^\d{4}\.\d{4,5}$", paper_id or ""))


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_full_text_from_pdf(pdf_bytes: bytes, max_chars: int = MAX_FULLTEXT_CHARS) -> str:
    """Extract text from PDF bytes, preferring PyMuPDF and falling back to pdfplumber."""

    def _extract_with_pymupdf() -> str:
        import fitz  # PyMuPDF

        try:
            fitz.TOOLS.mupdf_display_errors(False)
            fitz.TOOLS.mupdf_display_warnings(False)
        except Exception:
            pass

        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            total_chars = 0
            for page in doc:
                try:
                    page_text = page.get_text("text")
                except Exception as exc:
                    log.debug(f"PyMuPDF page extraction skipped: {exc}")
                    continue
                if not page_text:
                    continue
                text_parts.append(page_text)
                total_chars += len(page_text)
                if total_chars >= max_chars:
                    break
            full_text = "\n".join(text_parts)
            return clean_text(full_text[:max_chars])
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    def _extract_with_pdfplumber() -> str:
        import pdfplumber

        text_parts = []
        total_chars = 0
        with io.BytesIO(pdf_bytes) as buffer:
            with pdfplumber.open(buffer) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception as exc:
                        log.debug(f"pdfplumber page extraction skipped: {exc}")
                        continue
                    if not page_text:
                        continue
                    text_parts.append(page_text)
                    total_chars += len(page_text)
                    if total_chars >= max_chars:
                        break
        full_text = "\n".join(text_parts)
        return clean_text(full_text[:max_chars])

    text = ""
    pymupdf_failed = False

    try:
        text = _extract_with_pymupdf()
    except Exception as exc:
        pymupdf_failed = True
        log.warning(f"PyMuPDF extraction failed, falling back to pdfplumber: {exc}")

    if text:
        return text

    if not pymupdf_failed:
        log.debug("PyMuPDF returned no usable text; trying pdfplumber fallback.")

    try:
        text = _extract_with_pdfplumber()
        if text:
            return text
    except Exception as exc:
        log.warning(f"PDF fallback extraction failed: {exc}")

    log.warning("PDF text extraction failed for this file.")
    return ""


def download_pdf(paper_id: str, pdf_url: str, timeout: int = PDF_TIMEOUT) -> Optional[bytes]:
    """Download a PDF from arXiv."""
    if not is_valid_arxiv_id(paper_id):
        log.warning(f"Skipping invalid arXiv ID: {paper_id}")
        return None

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    # Check local cache first
    cache_path = PDF_CACHE_DIR / f"{paper_id}.pdf"
    if cache_path.exists():
        log.debug(f"PDF cache hit: {paper_id}")
        return cache_path.read_bytes()

    try:
        resp = requests.get(pdf_url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1000:
            # Cache locally
            cache_path.write_bytes(resp.content)
            return resp.content
        if resp.status_code == 404:
            log.warning(f"PDF not ready yet: {paper_id}")
            return None
        if resp.status_code == 403:
            log.warning(f"PDF forbidden for {paper_id}")
            return None
        log.warning(f"PDF download failed ({resp.status_code}): {paper_id}")
        return None
    except Exception as e:
        log.warning(f"PDF download error for {paper_id}: {e}")
        return None


def _run_db_write_with_retry(db, paper_id: str, action: str, fn) -> bool:
    """Run a database write with reconnect/retry so one dropped connection does not abort the batch."""
    for attempt in range(1, DB_WRITE_RETRIES + 1):
        try:
            fn()
            return True
        except OperationalError as exc:
            log.warning(
                f"Database write failed for {paper_id} during {action} (attempt {attempt}/{DB_WRITE_RETRIES}): {exc}"
            )
            db.close()
            if attempt < DB_WRITE_RETRIES:
                time.sleep(2 * attempt)
                continue
            return False

    return False


def _commit_with_retry(db, paper_id: str) -> bool:
    """Commit with reconnect/retry so a transient disconnect does not stop enrichment."""
    for attempt in range(1, DB_WRITE_RETRIES + 1):
        try:
            db.commit()
            return True
        except OperationalError as exc:
            log.warning(
                f"Commit failed for {paper_id} (attempt {attempt}/{DB_WRITE_RETRIES}): {exc}"
            )
            db.close()
            if attempt < DB_WRITE_RETRIES:
                time.sleep(2 * attempt)
                continue
            return False

    return False


def _load_keyword_state() -> dict:
    if not KEYWORD_STATE_PATH.exists():
        return {"next_page": 0, "failed_pages": []}
    try:
        state = json.loads(KEYWORD_STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("next_page", 0)
        state.setdefault("failed_pages", [])
        return state
    except Exception as exc:
        log.warning(f"Failed to load keyword resume state, starting fresh: {exc}")
        return {"next_page": 0, "failed_pages": []}


def _save_keyword_state(state: dict) -> None:
    KEYWORD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = KEYWORD_STATE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(KEYWORD_STATE_PATH)


# ---------------------------------------------------------------------------
# ArXiv API helpers
# ---------------------------------------------------------------------------


def build_keyword_query(keywords: list[str] = None) -> str:
    """Build an arXiv query from keyword phrases."""
    if keywords is None:
        keywords = [
            # Core mech interp
            "mechanistic interpretability",
            "transformer circuits",
            "circuit discovery",
            "induction heads",
            "superposition neural networks",
            "activation patching",
            "sparse autoencoders interpretability",
            # Methods
            "attention head analysis",
            "causal tracing",
            "logit lens",
            "path patching",
            "causal mediation analysis",
            # Features / neurons
            "feature visualization neural networks",
            "polysemanticity",
            "monosemantic features",
            "knowledge neurons",
            # Foundational / prerequisite
            "neural network interpretability",
            "probing classifiers transformers",
            "representation learning transformers",
            "residual stream",
            # Safety / alignment
            "interpretability alignment",
            "representation engineering",
            "activation steering",
        ]
    parts = [f'all:"{kw}"' for kw in keywords]
    return " OR ".join(parts)


def fetch_arxiv_papers(query: str, start: int = 0, max_results: int = 100) -> Optional[list[dict]]:
    """Fetch papers from arXiv API with retry on rate limiting."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    max_retries = 3
    backoff_base = 30

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(ARXIV_API_URL, params=params, timeout=ARXIV_REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if attempt < max_retries:
                    wait = backoff_base * (2 ** attempt)
                    log.warning(f"arXiv rate limited (429). Retry {attempt+1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                    continue
                log.warning("arXiv rate limit exhausted after retries.")
                return None
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            time.sleep(5)  # Rate limit courtesy
            return feed.entries
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) and attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                log.warning(f"arXiv rate limited. Retry {attempt+1}/{max_retries} in {wait}s...")
                time.sleep(wait)
                continue
            log.warning(f"arXiv fetch failed: {e}")
            return None
        except requests.exceptions.Timeout as e:
            log.warning(f"arXiv fetch timed out: {e}")
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                time.sleep(wait)
                continue
            return None
        except Exception as e:
            log.warning(f"arXiv fetch failed: {e}")
            return None
    return None


def parse_arxiv_entry(entry) -> dict:
    """Parse a feedparser entry into our paper format."""
    paper_id = extract_arxiv_id(entry.id)
    if not is_valid_arxiv_id(paper_id):
        return None
    title = sanitize_text(clean_text(entry.get("title", "")))
    abstract = sanitize_text(clean_text(entry.get("summary", "")))
    authors = sanitize_text(", ".join(a.get("name", "") for a in entry.get("authors", [])))
    categories = sanitize_text(", ".join(t.get("term", "") for t in entry.get("tags", [])))
    published = entry.get("published", "")
    updated = entry.get("updated", "")

    pdf_url = ""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    year = None
    try:
        year = int(published[:4]) if published else None
    except (ValueError, IndexError):
        pass

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "pdf_url": sanitize_text(pdf_url),
        "published": published,
        "updated": updated,
        "full_text": "",
        "download_status": "pending",
        "parse_status": "pending",
        "is_seed": False,
        "layer": assign_layer(year),
        "source": "arxiv",
        "year": year,
    }


# ---------------------------------------------------------------------------
# Ingestion modes
# ---------------------------------------------------------------------------


def ingest_seed_papers(db):
    """Ingest the curated seed papers from arXiv metadata."""
    log.info(f"Ingesting {len(SEED_PAPERS)} seed papers...")

    for seed in SEED_PAPERS:
        paper_id = seed["paper_id"]

        # Check if already ingested
        if db.paper_exists(paper_id):
            # Ensure seed flag is set
            db.update_paper_field(paper_id, "layer", seed["layer"])
            log.info(f"  Seed already exists: {paper_id} — {seed['title'][:60]}")
            # Make sure is_seed is set
            with db.conn.cursor() as cur:
                cur.execute("UPDATE papers SET is_seed = TRUE WHERE paper_id = %s", (paper_id,))
            db.commit()
            continue

        # Fetch metadata from arXiv
        query = f"id:{paper_id}"
        entries = fetch_arxiv_papers(query, max_results=1)

        if entries:
            paper = parse_arxiv_entry(entries[0])
            if not paper:
                log.warning(f"  Seed entry skipped due to invalid arXiv ID: {paper_id}")
                continue
            paper["is_seed"] = True
            paper["layer"] = seed["layer"]
            db.upsert_paper(paper)
            db.commit()
            log.info(f"  ✓ Ingested seed: {paper_id} — {paper['title'][:60]}")
        else:
            # Fallback: create minimal record from seed definition
            paper = {
                "paper_id": paper_id,
                "title": seed["title"],
                "abstract": "",
                "authors": "",
                "categories": "",
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                "published": f"{seed['year']}-01-01T00:00:00Z",
                "updated": None,
                "full_text": "",
                "download_status": "pending",
                "parse_status": "pending",
                "is_seed": True,
                "layer": seed["layer"],
                "source": "arxiv",
            }
            db.upsert_paper(paper)
            db.commit()
            log.info(f"  ✓ Ingested seed (fallback): {paper_id} — {seed['title'][:60]}")

        time.sleep(3)  # arXiv rate limit

    log.info(f"Seed ingestion complete. Corpus: {db.count_papers()} papers")


def ingest_keyword_papers(
    db,
    max_pages: int = 20,
    batch_size: int = 100,
    query_override: Optional[str] = None,
):
    """Ingest papers from arXiv via keyword queries (gap-filling mode)."""
    log.info("Starting keyword-based ingestion...")
    query = query_override or f"({build_keyword_query()}) AND (cat:cs.LG OR cat:cs.CL)"
    total_added = 0
    valid = 0
    invalid = 0
    state = _load_keyword_state()
    start_page = int(state.get("next_page", 0))
    failed_pages = list(state.get("failed_pages", []))

    pending_pages = []
    if failed_pages:
        pending_pages.extend(sorted({int(page) for page in failed_pages}))
    pending_pages.extend(range(start_page, max_pages))
    seen_pages = set()

    for page in pending_pages:
        if page in seen_pages or page >= max_pages:
            continue
        seen_pages.add(page)

        if db.count_papers() >= CORPUS_TARGET_MAX:
            log.info(f"Corpus target reached ({db.count_papers()} papers)")
            break

        start = page * batch_size
        entries = fetch_arxiv_papers(query, start=start, max_results=batch_size)
        if entries is None:
            log.warning(f"Fetch failed at page {page}; saving resume state and stopping so the next run can retry it.")
            state["next_page"] = page
            state["failed_pages"] = sorted(set(failed_pages + [page]))
            _save_keyword_state(state)
            break
        if not entries:
            log.info(f"No more results at page {page}")
            state["next_page"] = page + 1
            state["failed_pages"] = [p for p in failed_pages if p != page]
            _save_keyword_state(state)
            break

        page_added = 0
        for entry in entries:
            paper = parse_arxiv_entry(entry)
            if not paper:
                invalid += 1
                continue

            valid += 1

            if not is_relevant(paper["title"], paper["abstract"]):
                continue

            if len(paper.get("abstract", "")) < 200:
                continue

            if db.paper_exists(paper["paper_id"]):
                continue

            if db.count_papers() >= CORPUS_TARGET_MAX:
                break

            db.upsert_paper(paper)
            page_added += 1
            total_added += 1

        db.commit()
        log.info(f"  Page {page}: +{page_added} papers (total corpus: {db.count_papers()})")
        log.info(f"  Valid arXiv papers: {valid}, Skipped: {invalid}")
        state["next_page"] = page + 1
        state["failed_pages"] = [p for p in failed_pages if p != page]
        _save_keyword_state(state)
        time.sleep(3)  # arXiv rate limit

    log.info(f"Keyword ingestion complete: +{total_added} papers. Corpus: {db.count_papers()}")


def enrich_full_text(
    db,
    local_store: LocalPDFStore,
    limit: int = 0,
    pdf_timeout: int = PDF_TIMEOUT,
    retry_failed: bool = False,
):
    """Download PDFs locally and extract full text."""
    papers = db.get_papers_missing_full_text(limit=limit, include_failed=retry_failed)
    log.info(f"Enriching full text for {len(papers)} papers...")

    enriched = 0
    failed = 0

    for paper in papers:
        paper_id = paper["paper_id"]
        if not is_valid_arxiv_id(paper_id):
            db.update_paper_field(paper_id, "download_status", "skipped")
            db.commit()
            continue
        pdf_url = paper.get("pdf_url", "")
        pub_year = str(paper.get("published").year) if paper.get("published") else None
        target_pdf_path = local_store.get_pdf_path(paper_id, year=pub_year)
        legacy_pdf_path = Path("data/pdfs") / f"{paper_id}.pdf"

        # Reuse already-downloaded PDFs without hitting the network again.
        if target_pdf_path.exists() and target_pdf_path.stat().st_size > 1024:
            local_path = str(target_pdf_path)
            log.info(f"PDF {paper_id} already exists locally. Skipping download.")
        elif legacy_pdf_path.exists() and legacy_pdf_path.stat().st_size > 1024:
            local_path = str(legacy_pdf_path)
        else:
            local_path = local_store.download_pdf(paper_id, pdf_url, year=pub_year, timeout=pdf_timeout)

        if not local_path:
            if _run_db_write_with_retry(db, paper_id, "download_status=skipped", lambda: db.update_paper_field(paper_id, "download_status", "skipped")):
                _commit_with_retry(db, paper_id)
            failed += 1
            continue

        if not _run_db_write_with_retry(db, paper_id, "local_pdf_path", lambda: db.update_paper_field(paper_id, "local_pdf_path", local_path)):
            failed += 1
            continue
        if not _run_db_write_with_retry(db, paper_id, "download_status=downloaded", lambda: db.update_paper_field(paper_id, "download_status", "downloaded")):
            failed += 1
            continue
        if not _commit_with_retry(db, paper_id):
            failed += 1
            continue

        # Read PDF bytes for text extraction
        pdf_bytes = local_store.read_pdf(paper_id, year=pub_year)
        if not pdf_bytes:
            _run_db_write_with_retry(db, paper_id, "parse_status=failed", lambda: db.update_paper_field(paper_id, "parse_status", "failed"))
            _commit_with_retry(db, paper_id)
            failed += 1
            continue

        # Extract full text
        full_text = extract_full_text_from_pdf(pdf_bytes)
        if full_text:
            full_text = full_text.replace('\x00', '')  # Remove NUL bytes for PostgreSQL
            
        if full_text and len(full_text) > 100:
            if not _run_db_write_with_retry(db, paper_id, "full_text", lambda: db.update_paper_field(paper_id, "full_text", full_text)):
                failed += 1
                continue

            if not _run_db_write_with_retry(db, paper_id, "parse_status=parsed", lambda: db.update_paper_field(paper_id, "parse_status", "parsed")):
                failed += 1
                continue

            if not _commit_with_retry(db, paper_id):
                failed += 1
                continue

            enriched += 1
        else:
            _run_db_write_with_retry(db, paper_id, "parse_status=failed", lambda: db.update_paper_field(paper_id, "parse_status", "failed"))
            _commit_with_retry(db, paper_id)
            failed += 1

        if enriched % 10 == 0 and enriched > 0:
            log.info(f"  Enriched {enriched} / {len(papers)} papers (failed: {failed})")

        time.sleep(1)  # Be gentle with arXiv

    log.info(f"Enrichment complete: {enriched} enriched, {failed} failed, {len(papers)} total")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ArXiv ingestion for mechanistic interpretability")
    parser.add_argument(
        "--mode", choices=["seed", "keyword", "enrich", "all"],
        default="seed",
        help="Ingestion mode: seed (curated papers), keyword (arXiv search), enrich (PDF extraction), all (full pipeline)"
    )
    parser.add_argument("--pdf-timeout", type=int, default=60, help="PDF download timeout (seconds)")
    parser.add_argument("--max-pages", type=int, default=20, help="Max pages for keyword search")
    parser.add_argument("--query", type=str, default=None, help="Override the keyword query string")
    parser.add_argument("--enrich-limit", type=int, default=0, help="Max papers to enrich (0=all)")
    parser.add_argument("--reset-keyword-state", action="store_true", help="Clear saved keyword-ingestion resume state")
    parser.add_argument("--retry-failed", action="store_true", help="Retry papers whose download or parse status previously failed")
    args = parser.parse_args()

    if args.reset_keyword_state and KEYWORD_STATE_PATH.exists():
        KEYWORD_STATE_PATH.unlink()

    db = get_db()
    db.run_migrations()
    
    local_store = LocalPDFStore()

    if args.mode in ("seed", "all"):
        ingest_seed_papers(db)

    if args.mode in ("keyword", "all"):
        ingest_keyword_papers(db, max_pages=args.max_pages, query_override=args.query)

    if args.mode in ("enrich", "all"):
        enrich_full_text(
            db,
            local_store,
            limit=args.enrich_limit,
            pdf_timeout=args.pdf_timeout,
            retry_failed=args.retry_failed,
        )

    # Print corpus summary
    health = db.get_corpus_health()
    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    print(f"  Total papers:    {health['total_papers']}")
    print(f"  Seed papers:     {health['seed_papers']}")
    print(f"  With full text:  {health['with_full_text']}")
    print(f"  Without text:    {health['without_full_text']}")
    print(f"  Layers:          {health['layers']}")
    print(f"  Era %:           {health['era_percentages']}")
    print("=" * 60 + "\n")

    db.close()


if __name__ == "__main__":
    main()
