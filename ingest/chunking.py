"""
chunking.py — Chunk paper abstracts into overlapping segments for indexing.

Usage:
    conda run -n pytorch python ingest/chunking.py [--db-path data/arxiv_papers.db] [--output data/chunks.jsonl]
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
    tokens = tokenizer.encode(text)
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


def process_papers(
    db_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
):
    """Read papers from SQLite, chunk abstracts, write JSONL."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    papers = conn.execute(
        "SELECT paper_id, title, abstract, authors, categories FROM papers"
    ).fetchall()
    conn.close()

    if not papers:
        log.error(f"No papers found in {db_path}")
        return 0

    tokenizer = get_tokenizer()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for paper in tqdm(papers, desc="Chunking papers", unit="paper"):
            # Combine title + abstract for richer chunks
            full_text = f"{paper['title']}. {paper['abstract']}"
            chunks = chunk_text(full_text, tokenizer, chunk_size, overlap_frac)

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
                }
                f.write(json.dumps(chunk_record) + "\n")
                total_chunks += 1

    log.info(f"Created {total_chunks} chunks from {len(papers)} papers → {output_path}")
    return total_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunk paper abstracts for indexing")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                        help="SQLite database path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output JSONL file path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Chunk size in tokens (default: 300)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRAC,
                        help="Overlap fraction (default: 0.2)")
    args = parser.parse_args()

    total = process_papers(args.db_path, args.output, args.chunk_size, args.overlap)
    if total == 0:
        log.error("No chunks created. Check database content.")
    else:
        log.info(f"Done. {total} chunks ready for indexing.")


if __name__ == "__main__":
    main()
