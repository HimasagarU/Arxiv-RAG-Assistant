"""
build_bm25.py — Build and serialize a BM25 index from chunks.

Usage:
    conda run -n pytorch python index/build_bm25.py [--chunks data/chunks.jsonl]
"""

import argparse
import json
import os
import pickle
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

DEFAULT_CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
DEFAULT_BM25_PATH = os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")

# Simple English stopwords
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she",
    "they", "me", "us", "him", "her", "them", "my", "our", "your",
    "his", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "not", "no", "nor", "as", "if", "then", "so",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "all", "also", "am", "any", "because", "before", "between", "both",
    "each", "few", "more", "most", "other", "own", "same", "some",
    "such", "only", "into", "over", "under", "up", "down", "out",
])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens, optional stopword removal."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# Build BM25
# ---------------------------------------------------------------------------

def build_bm25_index(chunks_path: str, output_path: str):
    """Build BM25Okapi index from chunks JSONL and serialize."""
    # Load chunks
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    log.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Tokenize all documents
    log.info("Tokenizing documents...")
    chunk_ids = []
    tokenized_corpus = []

    for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk"):
        tokens = tokenize(chunk["chunk_text"])
        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk["chunk_id"])

    # Build BM25 index
    log.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Serialize (save index + chunk_ids mapping)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    index_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "tokenize_fn": "lowercase_alphanumeric_no_stopwords",
    }

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"BM25 index saved → {output_path} ({file_size_mb:.1f} MB)")
    log.info(f"Index contains {len(chunk_ids)} documents")
    return bm25, chunk_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from chunks")
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS_PATH,
                        help="Path to chunks JSONL file")
    parser.add_argument("--output", type=str, default=DEFAULT_BM25_PATH,
                        help="Output pickle file path")
    args = parser.parse_args()

    build_bm25_index(args.chunks, args.output)
    log.info("Done. BM25 index built successfully.")


if __name__ == "__main__":
    main()
