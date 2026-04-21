"""
ingest_arxiv.py -- Fetch papers from ArXiv API and store metadata in SQLite.

Usage:
    conda run -n pytorch python ingest/ingest_arxiv.py \
        [--categories cs.AI,cs.LG] [--max-papers 5000] [--include-full-text]
"""

import argparse
import io
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_CATEGORIES = os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG")
DEFAULT_MAX_PAPERS = int(os.getenv("MAX_PAPERS", "10000"))
DEFAULT_DB_PATH = os.getenv("DB_PATH", "data/arxiv_papers.db")
DEFAULT_PDF_TIMEOUT = int(os.getenv("PDF_TIMEOUT", "30"))
DEFAULT_MAX_FULLTEXT_CHARS = int(os.getenv("MAX_FULLTEXT_CHARS", "150000"))
DEFAULT_MAX_FULLTEXT_PAPERS = int(os.getenv("MAX_FULLTEXT_PAPERS", "500"))
RESULTS_PER_PAGE = 100  # ArXiv API max per request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str):
    """Add a SQLite column if it does not already exist."""
    cols = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        conn.commit()
        log.info(f"Added missing column '{column}' to table '{table}'")


def init_db(db_path: str) -> sqlite3.Connection:
    """Create SQLite database and papers table if not exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id   TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            abstract   TEXT NOT NULL,
            authors    TEXT,
            categories TEXT,
            pdf_url    TEXT,
            published  TEXT,
            updated    TEXT,
            full_text  TEXT
        )
        """
    )
    conn.commit()

    # Ensure backward compatibility with older DB files.
    _ensure_column(conn, "papers", "full_text", "TEXT")

    return conn


def upsert_paper(conn: sqlite3.Connection, paper: dict):
    """Insert or update a paper record (idempotent by paper_id)."""
    conn.execute(
        """
        INSERT INTO papers (paper_id, title, abstract, authors, categories, pdf_url, published, updated, full_text)
        VALUES (:paper_id, :title, :abstract, :authors, :categories, :pdf_url, :published, :updated, :full_text)
        ON CONFLICT(paper_id) DO UPDATE SET
            title      = excluded.title,
            abstract   = excluded.abstract,
            authors    = excluded.authors,
            categories = excluded.categories,
            pdf_url    = excluded.pdf_url,
            published  = excluded.published,
            updated    = excluded.updated,
            full_text  = COALESCE(NULLIF(excluded.full_text, ''), papers.full_text)
        """,
        paper,
    )


# ---------------------------------------------------------------------------
# ArXiv API helpers
# ---------------------------------------------------------------------------


def build_query(categories: list[str]) -> str:
    """Build ArXiv search query string from category list."""
    cat_queries = [f"cat:{cat}" for cat in categories]
    return "+OR+".join(cat_queries)


def clean_text(text: str) -> str:
    """Clean up whitespace and newlines from ArXiv text fields."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_arxiv_id(entry_id: str) -> str:
    """Extract paper ID from ArXiv entry URL.
    e.g. 'http://arxiv.org/abs/2301.12345v1' -> '2301.12345'
    """
    match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    if match:
        return match.group(1)
    # Fallback for older format IDs
    return entry_id.split("/abs/")[-1].split("v")[0]


def parse_entry(entry) -> dict:
    """Parse a feedparser entry into a paper dict."""
    paper_id = extract_arxiv_id(entry.id)
    authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
    categories = ", ".join(t.get("term", "") for t in entry.get("tags", []))

    # Find PDF link
    pdf_url = ""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    return {
        "paper_id": paper_id,
        "title": clean_text(entry.get("title", "")),
        "abstract": clean_text(entry.get("summary", "")),
        "authors": authors,
        "categories": categories,
        "pdf_url": pdf_url,
        "published": entry.get("published", ""),
        "updated": entry.get("updated", ""),
        "full_text": None,
    }


def fetch_pdf_text(
    pdf_url: str,
    timeout: int = DEFAULT_PDF_TIMEOUT,
    max_chars: int = DEFAULT_MAX_FULLTEXT_CHARS,
) -> str:
    """Download and extract text from a PDF URL with a size cap for safety."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(
            "pypdf is required for --include-full-text. Install dependencies from requirements.txt"
        ) from e

    try:
        response = requests.get(pdf_url, timeout=timeout)
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
    except Exception as e:
        log.debug(f"Failed to download/parse PDF {pdf_url}: {e}")
        return ""

    parts = []
    total_chars = 0

    for page in reader.pages:
        page_text = clean_text(page.extract_text() or "")
        if not page_text:
            continue

        if total_chars >= max_chars:
            break

        remaining = max_chars - total_chars
        if len(page_text) > remaining:
            page_text = page_text[:remaining]

        parts.append(page_text)
        total_chars += len(page_text)

    return clean_text(" ".join(parts))


def fetch_papers(categories: list[str], max_papers: int) -> list[dict]:
    """Fetch papers from ArXiv API with pagination."""
    query = build_query(categories)
    papers = []
    start = 0

    pbar = tqdm(total=max_papers, desc="Fetching papers", unit="paper")

    while start < max_papers:
        batch_size = min(RESULTS_PER_PAGE, max_papers - start)
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"Request failed at offset {start}: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        feed = feedparser.parse(response.text)
        entries = feed.entries

        if not entries:
            log.info(f"No more entries at offset {start}. Total fetched: {len(papers)}")
            break

        for entry in entries:
            paper = parse_entry(entry)
            if paper["abstract"] and paper["title"]:
                papers.append(paper)

        pbar.update(len(entries))
        start += len(entries)

        # ArXiv API rate limit: 1 request per 3 seconds
        time.sleep(3)

    pbar.close()
    log.info(f"Fetched {len(papers)} papers total")
    return papers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Ingest papers from ArXiv API into SQLite")
    parser.add_argument(
        "--categories",
        type=str,
        default=DEFAULT_CATEGORIES,
        help="Comma-separated ArXiv categories (default: cs.AI,cs.LG)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=DEFAULT_MAX_PAPERS,
        help="Maximum number of papers to fetch",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLite database path (default: data/arxiv_papers.db)",
    )
    parser.add_argument(
        "--include-full-text",
        action="store_true",
        help="Fetch and extract full text from paper PDFs (slower, larger storage)",
    )
    parser.add_argument(
        "--pdf-timeout",
        type=int,
        default=DEFAULT_PDF_TIMEOUT,
        help="PDF download timeout in seconds",
    )
    parser.add_argument(
        "--max-fulltext-chars",
        type=int,
        default=DEFAULT_MAX_FULLTEXT_CHARS,
        help="Maximum characters to store per paper full text",
    )
    parser.add_argument(
        "--max-fulltext-papers",
        type=int,
        default=DEFAULT_MAX_FULLTEXT_PAPERS,
        help="Max papers to enrich with full text; 0 means all",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    log.info(f"Ingesting up to {args.max_papers} papers from categories: {categories}")

    # Fetch papers
    papers = fetch_papers(categories, args.max_papers)

    if not papers:
        log.error("No papers fetched. Check network / ArXiv API.")
        sys.exit(1)

    if args.include_full_text:
        full_text_limit = (
            len(papers)
            if args.max_fulltext_papers == 0
            else min(args.max_fulltext_papers, len(papers))
        )
        log.info(
            f"Fetching full text from PDFs for up to {full_text_limit} papers "
            f"(timeout={args.pdf_timeout}s, max_chars={args.max_fulltext_chars})"
        )

        enriched = 0
        for paper in tqdm(papers[:full_text_limit], desc="Fetching full text", unit="paper"):
            full_text = fetch_pdf_text(
                paper["pdf_url"],
                timeout=args.pdf_timeout,
                max_chars=args.max_fulltext_chars,
            )
            if full_text:
                paper["full_text"] = full_text
                enriched += 1

        log.info(f"Extracted full text for {enriched}/{full_text_limit} papers")

    # Store in SQLite
    conn = init_db(args.db_path)
    log.info(f"Upserting {len(papers)} papers into {args.db_path}")

    for paper in tqdm(papers, desc="Storing papers", unit="paper"):
        upsert_paper(conn, paper)

    conn.commit()

    # Summary
    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    log.info(f"Database now contains {count} papers")
    conn.close()


if __name__ == "__main__":
    main()
