"""
chunking.py — Full-text chunking for the mechanistic interpretability corpus.

Creates retrieval chunks from paper full text only.

Usage:
    conda run -n pytorch python ingest/chunking.py [--source auto] [--reset]
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import tiktoken
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
DEFAULT_CHUNK_SIZE = 450
DEFAULT_OVERLAP_FRAC = 0.15


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def get_tokenizer():
    """Get the tiktoken tokenizer for chunk sizing."""
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Text chunk helpers
# ---------------------------------------------------------------------------


SECTION_HEADER_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s*\.?\s*(Introduction|Related Work|Background|Method|Methods|"
    r"Methodology|Experiments|Results|Discussion|Conclusion|Conclusions|"
    r"Evaluation|Analysis|Ablation|Abstract|Appendix|Preliminaries)\b",
    re.IGNORECASE | re.MULTILINE,
)


def detect_section_hint(text: str) -> str:
    """Detect the section a text chunk belongs to."""
    match = SECTION_HEADER_RE.search(text)
    if match:
        name = match.group(2).lower()
        mapping = {
            "introduction": "introduction",
            "related work": "related_work",
            "background": "background",
            "preliminaries": "background",
            "method": "method", "methods": "method", "methodology": "method",
            "experiments": "experiments", "experiment": "experiments",
            "evaluation": "experiments",
            "results": "results", "analysis": "results", "ablation": "results",
            "discussion": "discussion",
            "conclusion": "conclusion", "conclusions": "conclusion",
            "abstract": "abstract",
            "appendix": "appendix",
        }
        return mapping.get(name, "other")
    return "other"


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> list[dict]:
    """Split text into overlapping token-based chunks with section hints."""
    if not text or not text.strip():
        return []
    if not 0 <= overlap_frac < 1:
        raise ValueError("overlap_frac must be in the range [0, 1).")

    tokens = tokenizer.encode(text, disallowed_special=())
    if not tokens:
        return []

    overlap_tokens = max(0, int(chunk_size * overlap_frac))
    if overlap_tokens >= chunk_size:
        overlap_tokens = max(0, chunk_size - 1)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens)

        section_hint = detect_section_hint(chunk_text_str)

        chunks.append({
            "chunk_text": chunk_text_str.strip(),
            "token_count": len(chunk_tokens),
            "section_hint": section_hint,
        })

        if end >= len(tokens):
            break
        start = end - overlap_tokens

    return chunks


# ---------------------------------------------------------------------------
# Source text builder
# ---------------------------------------------------------------------------


def _strip_non_retrieval_sections(text: str) -> str:
    """Remove references, bibliography, acknowledgements, and appendix sections.
    
    These sections don't contribute useful retrieval signal and add noise.
    """
    # Patterns that mark the start of non-retrieval sections
    cut_patterns = [
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?(?:References|Bibliography|Works Cited)\s*\n",
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?Acknowledg(?:e)?ments?\s*\n",
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?Appendix\s*(?:[A-Z])?\s*\n",
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?Supplementary\s+Materials?\s*\n",
    ]
    
    earliest_cut = len(text)
    for pattern in cut_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < earliest_cut:
            earliest_cut = match.start()
    
    if earliest_cut < len(text):
        text = text[:earliest_cut].rstrip()
    
    return text


def build_chunk_source_text(paper: dict, source_mode: str = "auto") -> str:
    """Build the text to chunk from a paper record."""
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    full_text = (paper.get("full_text") or "").strip()

    if source_mode == "abstract":
        base = abstract
    elif source_mode == "full_text":
        if not full_text:
            return ""
        base = full_text
    elif source_mode == "auto":
        base = full_text if full_text else abstract
    else:
        base = full_text if full_text else abstract

    if not base:
        return ""

    # Strip references, acknowledgements, appendix before chunking
    if len(base) > 500:  # Only for full text, not abstracts
        base = _strip_non_retrieval_sections(base)

    return f"{title}. {base}" if title else base


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------


def chunk_paper(
    paper: dict,
    tokenizer,
    source_mode: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> list[dict]:
    """Chunk a single paper into text chunks."""
    paper_id = paper["paper_id"]
    title = paper.get("title", "")
    authors = paper.get("authors", "")
    categories = paper.get("categories", "")
    layer = paper.get("layer", "core")

    all_chunks = []

    # Text chunks from full text / abstract
    source_text = build_chunk_source_text(paper, source_mode)
    if source_text:
        text_chunks = chunk_text(source_text, tokenizer, chunk_size, overlap_frac)
        chunk_source = source_mode
        if source_mode == "auto":
            chunk_source = "full_text" if paper.get("full_text", "").strip() else "abstract"

        for idx, tc in enumerate(text_chunks):
            chunk_id = f"{paper_id}_text_{idx}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "chunk_type": "text",
                "modality": "text",
                "chunk_text": tc["chunk_text"],
                "section_hint": tc["section_hint"],
                "page_start": None,
                "page_end": None,
                "token_count": tc["token_count"],
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "chunk_source": chunk_source,
                "layer": layer,
                "artifact_meta": {},
                # Extra metadata for JSONL compat
                "title": title,
                "authors": authors,
                "categories": categories,
            })

    return all_chunks


def run_chunking(
    source_mode: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    limit: int = 0,
    reset: bool = False,
):
    """Run chunking for all papers in the corpus."""
    db = get_db()
    db.run_migrations()
    tokenizer = get_tokenizer()

    papers = db.get_all_papers(limit=limit)
    log.info(f"Chunking {len(papers)} papers (source={source_mode}, size={chunk_size}, overlap={overlap_frac})")

    if reset:
        log.info("Reset flag set: clearing existing chunks before rebuild.")
        db.delete_all_chunks()
        db.commit()

    chunks_path = Path(CHUNKS_PATH)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    type_counts = {"text": 0}

    with open(chunks_path, "w", encoding="utf-8") as f:
        for idx, paper in enumerate(papers):
            paper_chunks = chunk_paper(paper, tokenizer, source_mode, chunk_size, overlap_frac)

            if not paper_chunks:
                continue

            for chunk in paper_chunks:
                # Write to JSONL
                jsonl_record = {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "chunk_type": chunk["chunk_type"],
                    "modality": chunk["modality"],
                    "chunk_text": chunk["chunk_text"],
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", ""),
                    "categories": chunk.get("categories", ""),
                    "section_hint": chunk.get("section_hint", "other"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "token_count": chunk["token_count"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "chunk_source": chunk.get("chunk_source", "full_text"),
                    "layer": chunk.get("layer", "core"),
                }
                f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")

                # Insert to PostgreSQL
                db.insert_chunk(chunk)

                type_counts[chunk["chunk_type"]] = type_counts.get(chunk["chunk_type"], 0) + 1
                total_chunks += 1

            if (idx + 1) % 50 == 0:
                db.commit()
                log.info(f"  Chunked {idx + 1}/{len(papers)} papers ({total_chunks} chunks)")

    db.commit()

    log.info(f"\nChunking complete:")
    log.info(f"  Total chunks:    {total_chunks}")
    log.info(f"  By type:         {type_counts}")
    log.info(f"  JSONL:           {chunks_path}")
    log.info(f"  PostgreSQL:      {db.count_chunks()} rows")

    db.close()
    return total_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Full-text chunking")
    parser.add_argument("--source", choices=["abstract", "full_text", "auto"], default="auto",
                        help="Text source mode")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Target chunk size in tokens")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRAC,
                        help="Overlap fraction between chunks")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to chunk (0=all)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all existing chunks before rebuilding")
    args = parser.parse_args()

    run_chunking(
        source_mode=args.source,
        chunk_size=args.chunk_size,
        overlap_frac=args.overlap,
        limit=args.limit,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
