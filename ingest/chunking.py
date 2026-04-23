"""
chunking.py — Chunk paper text into overlapping segments for indexing.

Usage:
    conda run -n pytorch python ingest/chunking.py [--db-path data/arxiv_papers.db]
        [--output data/chunks.jsonl] [--source abstract|full_text|auto]
"""

import argparse
import json
import os
import sqlite3
import logging
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

DEFAULT_DB_PATH = os.getenv("DB_PATH", "data/arxiv_papers.db")
DEFAULT_OUTPUT = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
DEFAULT_CHUNK_SIZE = 300  # tokens
DEFAULT_OVERLAP_FRAC = 0.2
DEFAULT_SOURCE_MODE = os.getenv("CHUNK_SOURCE_MODE", "abstract")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Get tiktoken tokenizer for token counting."""
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> list[dict]:
    """
    Split text into overlapping chunks based on token count.

    Returns list of dicts with 'text' and 'token_count'.
    """
    tokens = tokenizer.encode(text, disallowed_special=())
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        return [{"text": text, "token_count": total_tokens}]

    overlap = int(chunk_size * overlap_frac)
    step = chunk_size - overlap
    chunks = []

    for start in range(0, total_tokens, step):
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = tokenizer.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text_decoded,
            "token_count": len(chunk_tokens),
        })
        if end >= total_tokens:
            break

    return chunks


def build_chunk_source_text(paper: dict, source_mode: str = "abstract") -> str:
    """Build text that will be chunked from selected source mode."""
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    full_text = (paper.get("full_text") or "").strip()

    if source_mode == "abstract":
        body = abstract
    elif source_mode == "full_text":
        body = full_text
    elif source_mode == "auto":
        body = full_text or abstract
    else:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    if not body:
        return ""
    return f"{title}. {body}" if title else body


def resolve_chunk_source(paper: dict, source_mode: str) -> str:
    """Resolve actual source used for a chunk when mode can fallback."""
    if source_mode != "auto":
        return source_mode
    return "full_text" if (paper.get("full_text") or "").strip() else "abstract"


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether a SQLite table has a given column name."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in cols


def process_papers(
    db_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    source_mode: str = DEFAULT_SOURCE_MODE,
):
    """Read papers from SQLite, chunk selected source text, write JSONL."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    has_full_text = _has_column(conn, "papers", "full_text")
    select_fields = "paper_id, title, abstract, authors, categories"
    if has_full_text:
        select_fields += ", full_text"

    papers = conn.execute(
        f"SELECT {select_fields} FROM papers"
    ).fetchall()
    conn.close()

    if not papers:
        log.error(f"No papers found in {db_path}")
        return 0

    tokenizer = get_tokenizer()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    skipped_no_source = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(papers, desc="Chunking papers", unit="paper"):
            paper = dict(row)
            source_text = build_chunk_source_text(paper, source_mode=source_mode)
            if not source_text:
                skipped_no_source += 1
                continue

            chunks = chunk_text(source_text, tokenizer, chunk_size, overlap_frac)
            chunk_source = resolve_chunk_source(paper, source_mode)

            for i, chunk in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{paper['paper_id']}_chunk_{i}",
                    "paper_id": paper["paper_id"],
                    "chunk_text": chunk["text"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "categories": paper["categories"],
                    "token_count": chunk["token_count"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_source": chunk_source,
                }
                f.write(json.dumps(chunk_record) + "\n")
                total_chunks += 1

    log.info(
        f"Created {total_chunks} chunks from {len(papers)} papers "
        f"(source_mode={source_mode}, skipped_without_source={skipped_no_source}) → {output_path}"
    )
    return total_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunk paper text for indexing")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                        help="SQLite database path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output JSONL file path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Chunk size in tokens (default: 300)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRAC,
                        help="Overlap fraction (default: 0.2)")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE_MODE,
                        choices=["abstract", "full_text", "auto"],
                        help="Source text mode: abstract, full_text, or auto (prefer full_text)")
    args = parser.parse_args()

    total = process_papers(
        args.db_path,
        args.output,
        args.chunk_size,
        args.overlap,
        source_mode=args.source,
    )
    if total == 0:
        log.error("No chunks created. Check database content.")
    else:
        log.info(f"Done. {total} chunks ready for indexing.")


if __name__ == "__main__":
    main()
