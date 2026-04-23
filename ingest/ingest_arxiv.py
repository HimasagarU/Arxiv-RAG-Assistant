"""
ingest_arxiv.py -- Fetch papers from ArXiv API and store metadata in SQLite.

Usage:
    conda run -n pytorch python ingest/ingest_arxiv.py \
    [--categories cs.AI,cs.LG] [--max-papers 5000] [--include-full-text]
    [--enrich-existing-full-text] [--start-offset 0]
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
DEFAULT_USER_AGENT = os.getenv(
    "ARXIV_USER_AGENT", "ArxivBot/1.0 (your_email@example.com)"
)
RESULTS_PER_PAGE = 50  # Intentionally conservative for API stability.
ARXIV_REQUEST_TIMEOUT = 30
ARXIV_REQUEST_MAX_RETRIES = 5
ARXIV_RETRY_INITIAL_DELAY = 5

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


def get_papers_missing_full_text(
    conn: sqlite3.Connection,
    categories: list[str] | None = None,
    limit: int = 0,
) -> list[dict]:
    """Load existing DB rows missing full_text for enrichment-only mode."""
    conn.row_factory = sqlite3.Row

    where_clauses = ["(full_text IS NULL OR TRIM(full_text) = '')"]
    params: list[object] = []

    cleaned_categories = [c.strip() for c in (categories or []) if c.strip()]
    if cleaned_categories:
        cat_filters = []
        for cat in cleaned_categories:
            cat_filters.append("categories LIKE ?")
            params.append(f"%{cat}%")
        where_clauses.append("(" + " OR ".join(cat_filters) + ")")

    query = (
        "SELECT paper_id, title, abstract, authors, categories, pdf_url, published, updated, full_text "
        "FROM papers "
        f"WHERE {' AND '.join(where_clauses)} "
        "ORDER BY published DESC"
    )

    if limit and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def enrich_existing_full_text(
    conn: sqlite3.Connection,
    categories: list[str],
    max_fulltext_papers: int,
    pdf_timeout: int,
    max_fulltext_chars: int,
) -> int:
    """Fetch full_text for existing rows missing it, without ArXiv metadata refetch."""
    limit = max_fulltext_papers if max_fulltext_papers > 0 else 0
    targets = get_papers_missing_full_text(conn, categories=categories, limit=limit)

    if not targets:
        log.info("No papers are missing full_text for selected categories.")
        return 0

    log.info(
        f"Enriching full_text for {len(targets)} existing papers "
        f"(timeout={pdf_timeout}s, max_chars={max_fulltext_chars})"
    )

    enriched = 0
    for paper in tqdm(targets, desc="Enriching existing full text", unit="paper"):
        pdf_url = (paper.get("pdf_url") or "").strip()
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{paper['paper_id']}.pdf"

        full_text = fetch_pdf_text(
            pdf_url,
            timeout=pdf_timeout,
            max_chars=max_fulltext_chars,
        )
        if not full_text:
            continue

        conn.execute(
            "UPDATE papers SET full_text = ? WHERE paper_id = ?",
            (full_text, paper["paper_id"]),
        )
        enriched += 1

        if enriched % 25 == 0:
            conn.commit()

    conn.commit()
    return enriched


# ---------------------------------------------------------------------------
# ArXiv API helpers
# ---------------------------------------------------------------------------


def build_query(categories: list[str]) -> str:
    """Build ArXiv search query string from category list."""
    cat_queries = [f"cat:{cat}" for cat in categories]
    # Keep logical operators as plain text; requests will URL-encode safely.
    return " OR ".join(cat_queries)


def clean_text(text: str) -> str:
    """Clean up whitespace and newlines from ArXiv text fields."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_text(text: str) -> str:
    """Remove non-encodable characters that can break SQLite writes."""
    if not text:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


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
        "title": sanitize_text(clean_text(entry.get("title", ""))),
        "abstract": sanitize_text(clean_text(entry.get("summary", ""))),
        "authors": sanitize_text(authors),
        "categories": sanitize_text(categories),
        "pdf_url": sanitize_text(pdf_url),
        "published": sanitize_text(entry.get("published", "")),
        "updated": sanitize_text(entry.get("updated", "")),
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
        response = requests.get(
            pdf_url,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
    except Exception as e:
        log.debug(f"Failed to download/parse PDF {pdf_url}: {e}")
        return ""

    parts = []
    total_chars = 0

    for page in reader.pages:
        page_text = sanitize_text(clean_text(page.extract_text() or ""))
        if not page_text:
            continue

        if total_chars >= max_chars:
            break

        remaining = max_chars - total_chars
        if len(page_text) > remaining:
            page_text = page_text[:remaining]

        parts.append(page_text)
        total_chars += len(page_text)

    return sanitize_text(clean_text(" ".join(parts)))


def fetch_papers(categories: list[str], max_papers: int) -> list[dict]:
    """Fetch papers from ArXiv API with pagination."""
    query = build_query(categories)
    papers = []
    start = 0
    headers = {"User-Agent": DEFAULT_USER_AGENT}

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

        response = None
        delay = ARXIV_RETRY_INITIAL_DELAY

        for attempt in range(1, ARXIV_REQUEST_MAX_RETRIES + 1):
            try:
                response = requests.get(
                    ARXIV_API_URL,
                    params=params,
                    timeout=ARXIV_REQUEST_TIMEOUT,
                    headers=headers,
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == ARXIV_REQUEST_MAX_RETRIES:
                    log.error(
                        f"Request failed at offset {start} after {attempt} attempts: {e}"
                    )
                    response = None
                    break

                log.warning(
                    f"Request failed at offset {start} (attempt {attempt}/{ARXIV_REQUEST_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= 2

        if response is None:
            log.error("Stopping fetch due to repeated request failures.")
            break

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


def _fetch_arxiv_feed(params: dict, start: int) -> str | None:
    """Fetch one ArXiv API page with retries and user-agent headers."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    response = None
    delay = ARXIV_RETRY_INITIAL_DELAY

    for attempt in range(1, ARXIV_REQUEST_MAX_RETRIES + 1):
        log.info(
            f"Requesting arXiv page start={start}, max_results={params.get('max_results')}, attempt={attempt}/{ARXIV_REQUEST_MAX_RETRIES}"
        )
        try:
            response = requests.get(
                ARXIV_API_URL,
                params=params,
                timeout=ARXIV_REQUEST_TIMEOUT,
                headers=headers,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt == ARXIV_REQUEST_MAX_RETRIES:
                log.error(
                    f"Request failed at offset {start} after {attempt} attempts: {e}"
                )
                return None

            log.warning(
                f"Request failed at offset {start} (attempt {attempt}/{ARXIV_REQUEST_MAX_RETRIES}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)
            delay *= 2

    return None


def ingest_papers_streaming(
    categories: list[str],
    max_papers: int,
    db_path: str,
    include_full_text: bool = False,
    pdf_timeout: int = DEFAULT_PDF_TIMEOUT,
    max_fulltext_chars: int = DEFAULT_MAX_FULLTEXT_CHARS,
    max_fulltext_papers: int = DEFAULT_MAX_FULLTEXT_PAPERS,
    start_offset: int = 0,
) -> tuple[int, int]:
    """Fetch, enrich, and upsert papers incrementally so progress is visible immediately."""
    query = build_query(categories)
    conn = init_db(db_path)
    fetched = 0
    stored = 0
    fulltext_enriched = 0
    start = max(0, start_offset)
    resume_base = start_offset if start_offset > 0 else 0

    if start_offset > 0:
        log.info(f"Resuming ingestion from offset {start_offset}")

    log.info(
        f"Streaming ingest started for categories={categories}, max_papers={max_papers}, "
        f"include_full_text={include_full_text}, max_fulltext_papers={max_fulltext_papers}"
    )

    while start < max_papers:
        batch_size = min(RESULTS_PER_PAGE, max_papers - start)
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        log.info(
            f"[{categories[0] if categories else 'all'}] Fetching page start={start}, batch_size={batch_size}, "
            f"stored={stored}, fetched={fetched}, full_text={fulltext_enriched}"
        )

        feed_text = _fetch_arxiv_feed(params, start)
        if feed_text is None:
            log.error("Stopping fetch due to repeated request failures.")
            break

        feed = feedparser.parse(feed_text)
        entries = feed.entries

        if not entries:
            log.info(f"No more entries at offset {start}. Total fetched: {fetched}")
            break

        for entry in entries:
            if fetched >= max_papers:
                break

            paper = parse_entry(entry)
            if not paper["abstract"] or not paper["title"]:
                continue

            fetched += 1
            overall_index = resume_base + fetched
            remaining_total = max(0, max_papers - start_offset)

            if include_full_text and (max_fulltext_papers == 0 or fulltext_enriched < max_fulltext_papers):
                log.info(
                    f"[{categories[0] if categories else 'all'}] Full-text {fetched}/{remaining_total} "
                    f"(overall {overall_index}/{max_papers}) paper_id={paper['paper_id']}"
                )
                full_text = fetch_pdf_text(
                    paper["pdf_url"],
                    timeout=pdf_timeout,
                    max_chars=max_fulltext_chars,
                )
                if full_text:
                    paper["full_text"] = full_text
                    fulltext_enriched += 1

            upsert_paper(conn, paper)
            stored += 1

            if stored % 10 == 0 or stored == 1:
                conn.commit()
                log.info(
                    f"[{categories[0] if categories else 'all'}] Stored {stored}/{remaining_total} papers "
                    f"(overall {overall_index}/{max_papers}, fetched={fetched}, full_text={fulltext_enriched})"
                )

        conn.commit()
        log.info(
            f"[{categories[0] if categories else 'all'}] Page complete: start={start}, "
            f"fetched={fetched}, stored={stored}, full_text={fulltext_enriched}, overall={resume_base + fetched}/{max_papers}"
        )

        start += len(entries)

        # ArXiv API rate limit: 1 request per 3 seconds
        time.sleep(3)

    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    log.info(
        f"Streaming ingest finished: stored={stored}, full_text={fulltext_enriched}, db_total={count}"
    )
    return stored, fulltext_enriched


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
        "--enrich-existing-full-text",
        action="store_true",
        help=(
            "Only enrich existing DB rows missing full_text; "
            "skip ArXiv metadata fetch"
        ),
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
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Resume arXiv pagination from this offset (e.g. 850 to continue after 850 papers)",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    log.info(f"Using categories: {categories}")

    if args.enrich_existing_full_text:
        conn = init_db(args.db_path)
        enriched = enrich_existing_full_text(
            conn,
            categories=categories,
            max_fulltext_papers=args.max_fulltext_papers,
            pdf_timeout=args.pdf_timeout,
            max_fulltext_chars=args.max_fulltext_chars,
        )
        count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        log.info(f"Database now contains {count} papers")
        log.info(f"Enriched full_text for {enriched} papers")
        conn.close()
        return

    log.info(f"Ingesting up to {args.max_papers} papers from categories: {categories}")
    if args.include_full_text:
        log.info(
            f"Full-text mode enabled; progress will be logged as papers are stored "
            f"(pdf_timeout={args.pdf_timeout}s, max_chars={args.max_fulltext_chars})"
        )

    stored, enriched = ingest_papers_streaming(
        categories=categories,
        max_papers=args.max_papers,
        db_path=args.db_path,
        include_full_text=args.include_full_text,
        pdf_timeout=args.pdf_timeout,
        max_fulltext_chars=args.max_fulltext_chars,
        max_fulltext_papers=args.max_fulltext_papers,
        start_offset=args.start_offset,
    )

    if stored == 0:
        log.error("No papers ingested. Check network / ArXiv API.")
        sys.exit(1)

    conn = sqlite3.connect(args.db_path)
    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    log.info(f"Database now contains {count} papers")


if __name__ == "__main__":
    main()
