"""
chunking.py — Chunk paper text into overlapping segments for indexing.

Supports paragraph-aware chunking and section-header detection
for richer downstream retrieval.

Usage:
    conda run -n pytorch python ingest/chunking.py [--db-path data/arxiv_papers.db]
        [--output data/chunks.jsonl] [--source abstract|full_text|auto]
"""

import argparse
import json
import os
import re
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
DEFAULT_CHUNK_SIZE = 512  # tokens — larger to preserve complete paragraphs
DEFAULT_OVERLAP_FRAC = 0.15
DEFAULT_SOURCE_MODE = os.getenv("CHUNK_SOURCE_MODE", "auto")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

# Regex patterns for common section headers (LaTeX-style, Markdown-style, numbered)
_SECTION_PATTERNS = [
    # Numbered sections: "1. Introduction", "2 Method", "3.1 Dataset"
    re.compile(
        r"^\s*\d+(?:\.\d+)*\.?\s+"
        r"(introduction|related\s+work|background|preliminaries|methodology|method|methods|"
        r"approach|model|architecture|framework|algorithm|experiments?|"
        r"results?|evaluation|discussion|conclusion|conclusions|"
        r"abstract|summary|overview|limitations|future\s+work|"
        r"training|implementation|setup|analysis|ablation)",
        re.IGNORECASE,
    ),
    # Unnumbered headers at line start
    re.compile(
        r"^(introduction|related\s+work|background|preliminaries|methodology|method|methods|"
        r"approach|model\s+architecture|our\s+approach|proposed\s+method|"
        r"experiments?|experimental\s+setup|results?\s+and\s+discussion|results?|"
        r"evaluation|discussion|conclusion|conclusions|"
        r"abstract|summary|overview|limitations|future\s+work|"
        r"training\s+details?|implementation\s+details?|"
        r"reward\s+model|policy\s+optimization)[\s:.\-]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Map detected header text → canonical section_hint
_SECTION_MAP = {
    "introduction": "intro",
    "abstract": "abstract",
    "summary": "abstract",
    "overview": "intro",
    "related work": "background",
    "background": "background",
    "preliminaries": "background",
    "methodology": "method",
    "method": "method",
    "methods": "method",
    "approach": "method",
    "our approach": "method",
    "proposed method": "method",
    "model architecture": "method",
    "model": "method",
    "architecture": "method",
    "framework": "method",
    "algorithm": "method",
    "training": "method",
    "training details": "method",
    "implementation": "method",
    "implementation details": "method",
    "reward model": "method",
    "policy optimization": "method",
    "experiment": "results",
    "experiments": "results",
    "experimental setup": "results",
    "results": "results",
    "results and discussion": "results",
    "evaluation": "results",
    "analysis": "results",
    "ablation": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "conclusion",
    "future work": "conclusion",
    "setup": "method",
}


def detect_section_hint(text: str) -> str:
    """
    Detect which section of a paper a chunk likely belongs to.
    Returns one of: intro, background, method, results, discussion,
    conclusion, abstract, or 'other'.
    """
    # Check the first few lines of the chunk for a header
    first_lines = text[:300].strip()

    for pattern in _SECTION_PATTERNS:
        match = pattern.search(first_lines)
        if match:
            # Get the captured group (the section name)
            section_text = match.group(1) if match.lastindex else match.group(0)
            section_text = section_text.strip().lower()
            section_text = re.sub(r'[\s:.\-]+$', '', section_text)

            # Look up canonical name
            for key, hint in _SECTION_MAP.items():
                if key in section_text:
                    return hint

    # Heuristic fallback: check for keywords in the text itself
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["we propose", "our method", "our approach", "algorithm ", "pipeline"]):
        return "method"
    if any(kw in text_lower for kw in ["table ", "figure ", "accuracy", "f1 score", "benchmark", "dataset"]):
        return "results"
    if any(kw in text_lower for kw in ["in this paper", "we introduce", "we present", "this work"]):
        return "intro"

    return "other"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Get tiktoken tokenizer for token counting."""
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> list[dict]:
    """
    Split text into overlapping chunks based on token count.
    Pure token-window fallback for text without paragraph structure.

    Returns list of dicts with 'text', 'token_count', and 'section_hint'.
    """
    tokens = tokenizer.encode(text, disallowed_special=())
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        return [{
            "text": text,
            "token_count": total_tokens,
            "section_hint": detect_section_hint(text),
        }]

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
            "section_hint": detect_section_hint(chunk_text_decoded),
        })
        if end >= total_tokens:
            break

    return chunks


def chunk_text_paragraphs(
    text: str,
    tokenizer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> list[dict]:
    """
    Paragraph-aware chunking: split on double-newlines first, then merge
    paragraphs into chunks up to the token limit. Preserves paragraph
    boundaries for more coherent chunks.

    Falls back to token-window chunking for text without paragraph breaks.
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If no paragraph structure detected, fall back to token-window
    if len(paragraphs) <= 1:
        return chunk_text(text, tokenizer, chunk_size, overlap_frac)

    # Merge paragraphs into chunks respecting token limits
    chunks = []
    current_paragraphs = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para, disallowed_special=()))

        # If a single paragraph exceeds the chunk size, chunk it with token-window
        if para_tokens > chunk_size:
            # Flush current buffer first
            if current_paragraphs:
                chunk_text_str = "\n\n".join(current_paragraphs)
                chunks.append({
                    "text": chunk_text_str,
                    "token_count": current_tokens,
                    "section_hint": detect_section_hint(chunk_text_str),
                })
                current_paragraphs = []
                current_tokens = 0

            # Chunk the large paragraph with token-window
            sub_chunks = chunk_text(para, tokenizer, chunk_size, overlap_frac)
            chunks.extend(sub_chunks)
            continue

        # Would adding this paragraph exceed the limit?
        if current_tokens + para_tokens > chunk_size and current_paragraphs:
            # Flush current buffer
            chunk_text_str = "\n\n".join(current_paragraphs)
            chunks.append({
                "text": chunk_text_str,
                "token_count": current_tokens,
                "section_hint": detect_section_hint(chunk_text_str),
            })

            # Start new buffer with overlap: keep last paragraph for context
            overlap_para = current_paragraphs[-1] if current_paragraphs else ""
            overlap_tokens = len(tokenizer.encode(overlap_para, disallowed_special=())) if overlap_para else 0
            current_paragraphs = [overlap_para] if overlap_para else []
            current_tokens = overlap_tokens

        current_paragraphs.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_paragraphs:
        chunk_text_str = "\n\n".join(current_paragraphs)
        actual_tokens = len(tokenizer.encode(chunk_text_str, disallowed_special=()))
        chunks.append({
            "text": chunk_text_str,
            "token_count": actual_tokens,
            "section_hint": detect_section_hint(chunk_text_str),
        })

    return chunks


def build_chunk_source_text(paper: dict, source_mode: str = "auto") -> str:
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

            chunk_source = resolve_chunk_source(paper, source_mode)

            # Use paragraph-aware chunking for full_text, token-window for abstracts
            if chunk_source == "full_text":
                chunks = chunk_text_paragraphs(source_text, tokenizer, chunk_size, overlap_frac)
            else:
                chunks = chunk_text(source_text, tokenizer, chunk_size, overlap_frac)

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
                    "section_hint": chunk.get("section_hint", "other"),
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
                        help="Chunk size in tokens (default: 512)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRAC,
                        help="Overlap fraction (default: 0.15)")
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
