"""
build_bm25.py -- Build BM25 and lightweight metadata artifacts from local chunks.

This script now prefers the local `data/chunks.jsonl` corpus so artifact rebuilds
do not depend on PostgreSQL containing the full legacy corpus.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List

import joblib
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.chunking import build_contextual_text
from index.lexical_text import build_lexical_index_text
from utils.artifact_schema import ARTIFACT_SCHEMA_VERSION, write_artifact_manifest
from utils.section_labels import normalize_section_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", str(DATA_DIR / "chunks.jsonl")))
BM25_PATH = DATA_DIR / "bm25_v1.pkl"
CHUNKS_META_PATH = DATA_DIR / "chunks_meta.jsonl"
CHUNKS_TEXT_PATH = DATA_DIR / "chunks_text.jsonl"
PAPERS_META_PATH = DATA_DIR / "papers_meta.json"


def tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _load_existing_papers_meta() -> dict:
    if PAPERS_META_PATH.exists():
        with open(PAPERS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    tmp_path.replace(final_path)


def _contextual_text_for_chunk(chunk: dict) -> str:
    contextual_text = chunk.get("contextual_text")
    if contextual_text:
        return contextual_text
    return build_contextual_text(
        title=chunk.get("title", ""),
        authors=chunk.get("authors", ""),
        categories=chunk.get("categories", ""),
        section_hint=chunk.get("section_hint", "other"),
        chunk_text=chunk.get("chunk_text", ""),
        chunk_index=chunk.get("chunk_index", 0),
        total_chunks=chunk.get("total_chunks", 1),
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks corpus: {CHUNKS_PATH}")

    existing_papers_meta = _load_existing_papers_meta()
    papers_meta = dict(existing_papers_meta)
    corpus_tokens = []

    tmp_meta = CHUNKS_META_PATH.with_suffix(CHUNKS_META_PATH.suffix + ".tmp")
    tmp_text = CHUNKS_TEXT_PATH.with_suffix(CHUNKS_TEXT_PATH.suffix + ".tmp")
    tmp_papers = PAPERS_META_PATH.with_suffix(PAPERS_META_PATH.suffix + ".tmp")
    tmp_bm25 = BM25_PATH.with_suffix(BM25_PATH.suffix + ".tmp")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f_in, \
         open(tmp_meta, "w", encoding="utf-8") as f_meta, \
         open(tmp_text, "w", encoding="utf-8") as f_text:

        for i, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk["chunk_id"]
            paper_id = chunk["paper_id"]
            raw_text = chunk.get("chunk_text", "")
            sec_norm = normalize_section_label(chunk.get("section_hint", "other"))
            chunk_norm = {**chunk, "section_hint": sec_norm}
            contextual_text = _contextual_text_for_chunk(chunk_norm)
            chunk_for_lex = {**chunk_norm, "chunk_text": raw_text}
            lexical_doc = build_lexical_index_text(chunk_for_lex)

            corpus_tokens.append(tokenize(lexical_doc))

            existing = papers_meta.get(paper_id, {})
            pub_date = chunk.get("published") or existing.get("published")
            papers_meta[paper_id] = {
                "title": chunk.get("title", "") or existing.get("title", ""),
                "authors": chunk.get("authors", "") or existing.get("authors", ""),
                "categories": chunk.get("categories", "") or existing.get("categories", ""),
                "published": pub_date,
                "layer": chunk.get("layer", existing.get("layer", "core")),
            }

            meta = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title": chunk.get("title", ""),
                "authors": chunk.get("authors", ""),
                "categories": chunk.get("categories", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
                "modality": chunk.get("modality", "text"),
                "section_hint": sec_norm,
                "layer": chunk.get("layer", "core"),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "chunk_source": chunk.get("chunk_source", "full_text"),
            }
            f_meta.write(json.dumps(meta) + "\n")
            f_text.write(json.dumps({
                "chunk_id": chunk_id,
                "text": raw_text,
                "contextual_text": contextual_text,
                "lexical_index_text": lexical_doc,
            }) + "\n")

            if i % 10000 == 0:
                log.info(f"Processed {i} chunks...")

    log.info(f"Total chunks processed: {len(corpus_tokens)}")
    log.info(f"Saving papers metadata ({len(papers_meta)} papers)...")
    with open(tmp_papers, "w", encoding="utf-8") as f:
        json.dump(papers_meta, f, indent=2)

    log.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus_tokens)

    log.info("Saving BM25 index...")
    joblib.dump(bm25, tmp_bm25, compress=3)

    _atomic_replace(tmp_meta, CHUNKS_META_PATH)
    _atomic_replace(tmp_text, CHUNKS_TEXT_PATH)
    _atomic_replace(tmp_papers, PAPERS_META_PATH)
    _atomic_replace(tmp_bm25, BM25_PATH)

    write_artifact_manifest(
        DATA_DIR,
        extra={
            "bm25_path": str(BM25_PATH.name),
            "chunks_meta": str(CHUNKS_META_PATH.name),
            "chunks_text": str(CHUNKS_TEXT_PATH.name),
            "papers_meta": str(PAPERS_META_PATH.name),
            "chunk_count": len(corpus_tokens),
            "paper_count": len(papers_meta),
            "lexical_fields": ["title", "authors", "categories", "section", "body"],
        },
    )

    log.info("Artifact generation complete from local chunks corpus.")
    for p in [BM25_PATH, CHUNKS_META_PATH, CHUNKS_TEXT_PATH, PAPERS_META_PATH]:
        size_mb = p.stat().st_size / (1024 * 1024)
        log.info(f" - {p.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
