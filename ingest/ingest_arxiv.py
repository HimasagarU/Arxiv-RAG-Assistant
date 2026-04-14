"""
ingest_arxiv.py — Fetch papers from ArXiv API and store metadata in SQLite.

Usage:
    conda run -n pytorch python ingest/ingest_arxiv.py [--categories cs.AI,cs.LG] [--max-papers 5000]
"""

import argparse
import os
import re
import sqlite3
import sys
import time
import logging
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

def init_db(db_path: str) -> sqlite3.Connection:
    """Create SQLite database and papers table if not exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id   TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            abstract   TEXT NOT NULL,
            authors    TEXT,
            categories TEXT,
            pdf_url    TEXT,
            published  TEXT,
            updated    TEXT
        )
    """)
    conn.commit()
    return conn


def upsert_paper(conn: sqlite3.Connection, paper: dict):
    """Insert or update a paper record (idempotent by paper_id)."""
    conn.execute("""
        INSERT INTO papers (paper_id, title, abstract, authors, categories, pdf_url, published, updated)
        VALUES (:paper_id, :title, :abstract, :authors, :categories, :pdf_url, :published, :updated)
        ON CONFLICT(paper_id) DO UPDATE SET
            title      = excluded.title,
            abstract   = excluded.abstract,
            authors    = excluded.authors,
            categories = excluded.categories,
            pdf_url    = excluded.pdf_url,
            published  = excluded.published,
            updated    = excluded.updated
    """, paper)


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
    }


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

        url = f"{ARXIV_API_URL}?search_query={query}&start={start}&max_results={batch_size}&sortBy=submittedDate&sortOrder=descending"

        try:
            response = requests.get(url, timeout=30)
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
    parser.add_argument("--categories", type=str, default=DEFAULT_CATEGORIES,
                        help="Comma-separated ArXiv categories (default: cs.AI,cs.LG)")
    parser.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS,
                        help="Maximum number of papers to fetch (default: 5000)")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                        help="SQLite database path (default: data/arxiv_papers.db)")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    log.info(f"Ingesting up to {args.max_papers} papers from categories: {categories}")

    # Fetch papers
    papers = fetch_papers(categories, args.max_papers)

    if not papers:
        log.error("No papers fetched. Check network / ArXiv API.")
        sys.exit(1)

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
