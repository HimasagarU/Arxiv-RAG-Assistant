"""
build_bm25.py — Build offline BM25 index and metadata artifacts.

This script fetches all text chunks from PostgreSQL, tokenizes them,
builds a rank_bm25 BM25Okapi index, and saves it to disk along with
lightweight metadata mapping files.

Run this locally to generate artifacts before deploying to HF Spaces.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict

import joblib
from rank_bm25 import BM25Okapi
import psycopg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Output paths
DATA_DIR = Path("data")
BM25_PATH = DATA_DIR / "bm25_v1.pkl"
CHUNKS_META_PATH = DATA_DIR / "chunks_meta.jsonl"
CHUNKS_TEXT_PATH = DATA_DIR / "chunks_text.jsonl"
PAPERS_META_PATH = DATA_DIR / "papers_meta.json"

def tokenize(text: str) -> List[str]:
    """Simple lowercase + punctuation stripping tokenizer."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("Connecting to PostgreSQL...")
    db = get_db()
    
    # We will build the corpus and metadata incrementally to manage memory
    corpus_tokens = []
    
    log.info("Fetching chunks and papers from database...")
    # Fetch all chunks joined with paper metadata
    sql = """
        SELECT
            c.chunk_id,
            c.paper_id,
            c.chunk_text,
            c.chunk_type,
            c.modality,
            c.section_hint,
            c.layer,
            c.chunk_source,
            c.token_count,
            c.chunk_index,
            c.total_chunks,
            c.title AS chunk_title,
            c.authors AS chunk_authors,
            c.categories AS chunk_categories,
            p.title AS paper_title,
            p.authors AS paper_authors,
            p.categories AS paper_categories,
            p.published
        FROM chunks c
        JOIN papers p ON p.paper_id = c.paper_id
        ORDER BY c.chunk_id ASC
    """
    
    # Track papers to create papers_meta.json
    papers_meta = {}
    
    with db.conn.cursor() as cur:
        cur.execute(sql)
        
        # We write JSONL line by line
        with open(CHUNKS_META_PATH, "w", encoding="utf-8") as f_meta, \
             open(CHUNKS_TEXT_PATH, "w", encoding="utf-8") as f_text:
             
            for i, row in enumerate(cur):
                chunk_id = row["chunk_id"]
                paper_id = row["paper_id"]
                
                # Tokenize text
                text = row.get("chunk_text", "")
                corpus_tokens.append(tokenize(text))
                
                # Extract paper metadata
                if paper_id not in papers_meta:
                    papers_meta[paper_id] = {
                        "title": row.get("paper_title", ""),
                        "authors": row.get("paper_authors", ""),
                        "categories": row.get("paper_categories", ""),
                        "published": str(row.get("published", "")) if row.get("published") else None,
                        "layer": row.get("layer", "core")
                    }
                
                # Chunk metadata (no full text)
                meta = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": row.get("chunk_title", "") or row.get("paper_title", ""),
                    "authors": row.get("chunk_authors", "") or row.get("paper_authors", ""),
                    "categories": row.get("chunk_categories", "") or row.get("paper_categories", ""),
                    "chunk_type": row.get("chunk_type", "text"),
                    "modality": row.get("modality", "text"),
                    "section_hint": row.get("section_hint", "other"),
                    "layer": row.get("layer", "core"),
                    "chunk_index": row.get("chunk_index", 0),
                    "total_chunks": row.get("total_chunks", 1),
                    "chunk_source": row.get("chunk_source", "full_text")
                }
                
                f_meta.write(json.dumps(meta) + "\n")
                f_text.write(json.dumps({"chunk_id": chunk_id, "text": text}) + "\n")
                
                if (i + 1) % 10000 == 0:
                    log.info(f"Processed {i + 1} chunks...")

    log.info(f"Total chunks processed: {len(corpus_tokens)}")
    
    # Save papers_meta.json
    log.info(f"Saving papers metadata ({len(papers_meta)} papers)...")
    with open(PAPERS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(papers_meta, f, indent=2)

    # Build BM25 index
    log.info("Building BM25Okapi index (this may take a moment)...")
    bm25 = BM25Okapi(corpus_tokens)
    
    log.info("Saving BM25 index to disk with compression...")
    joblib.dump(bm25, BM25_PATH, compress=3)
    
    log.info("Artifact generation complete!")
    log.info("Next steps: Zip these artifacts and upload them to your Cloudflare R2 bucket.")
    log.info(f"Artifacts generated in {DATA_DIR}:")
    for p in [BM25_PATH, CHUNKS_META_PATH, CHUNKS_TEXT_PATH, PAPERS_META_PATH]:
        size_mb = p.stat().st_size / (1024 * 1024)
        log.info(f" - {p.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
