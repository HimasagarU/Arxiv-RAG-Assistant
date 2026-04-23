"""
build_chroma.py — Compute embeddings and build Chroma vector index.

Usage:
    conda run -n pytorch python index/build_chroma.py [--chunks data/chunks.jsonl]
"""

import argparse
import json
import os
import logging
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

DEFAULT_CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "arxiv_chunks"
BATCH_SIZE = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunks(path: str) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def build_chroma_index(
    chunks: list[dict],
    model_name: str,
    chroma_dir: str,
    save_embeddings: bool = True,
):
    """Compute embeddings and insert into Chroma."""
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Extract texts
    texts = [c["chunk_text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    # Compute embeddings in batches
    log.info(f"Computing embeddings for {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    all_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log.info(f"Embedding shape: {all_embeddings.shape}")

    # Save embeddings backup
    if save_embeddings:
        emb_path = os.path.join(os.path.dirname(chroma_dir), "embeddings.npy")
        np.save(emb_path, all_embeddings)
        log.info(f"Saved embeddings backup → {emb_path}")

    # Build Chroma collection
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete existing collection if exists (rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches (Chroma max batch size ~5000)
    chroma_batch = 4096
    for start in tqdm(range(0, len(chunks), chroma_batch), desc="Upserting to Chroma"):
        end = min(start + chroma_batch, len(chunks))
        batch_ids = ids[start:end]
        batch_embeddings = all_embeddings[start:end].tolist()
        batch_documents = texts[start:end]
        batch_metadatas = [
            {
                "paper_id": c["paper_id"],
                "title": c["title"],
                "authors": c["authors"][:500],  # Truncate long author lists
                "categories": c["categories"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "chunk_source": c.get("chunk_source", "abstract"),
                "section_hint": c.get("section_hint", "other"),
                "total_chunks": c.get("total_chunks", 1),
            }
            for c in chunks[start:end]
        ]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )

    log.info(f"Chroma collection '{COLLECTION_NAME}' has {collection.count()} documents")
    return collection


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Chroma vector index from chunks")
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS_PATH,
                        help="Path to chunks JSONL file")
    parser.add_argument("--chroma-dir", type=str, default=DEFAULT_CHROMA_DIR,
                        help="Chroma persistent directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Sentence-transformers model name")
    parser.add_argument("--no-save-embeddings", action="store_true",
                        help="Skip saving numpy embeddings backup")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        log.error("No chunks to index. Run chunking first.")
        return

    build_chroma_index(chunks, args.model, args.chroma_dir,
                       save_embeddings=not args.no_save_embeddings)
    log.info("Done. Chroma index built successfully.")


if __name__ == "__main__":
    main()
