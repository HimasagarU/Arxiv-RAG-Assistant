"""
build_qdrant.py — Build the text-only Qdrant collection.

Collection:
    - arxiv_text: text chunks from paper full text / abstract

Uses BGE-large-en-v1.5 embeddings (1024-dim) with cosine distance
and tuned HNSW (m=32, ef_construct=400) against Qdrant Cloud.

Usage:
    conda run -n pytorch python index/build_qdrant.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
BATCH_SIZE = int(os.getenv("QDRANT_EMBED_BATCH_SIZE", "64"))
UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "16"))
UPSERT_RETRIES = int(os.getenv("QDRANT_UPSERT_RETRIES", "5"))
UPSERT_BACKOFF_SECONDS = float(os.getenv("QDRANT_UPSERT_BACKOFF_SECONDS", "2.0"))

# HNSW tuning for high recall
HNSW_M = 32               # Higher = better recall, more memory
HNSW_EF_CONSTRUCT = 400   # Higher = better index quality, slower build

COLLECTION_NAME = "arxiv_text"


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a deterministic UUID for Qdrant."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def load_chunks(chunks_path: str) -> list[dict]:
    """Load chunks from JSONL."""
    chunks = []
    total = 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            total += 1

    log.info(f"Loaded {total} chunks from {chunks_path}")
    log.info(f"  text: {len(chunks)} chunks")
    return chunks


def create_collection(client: QdrantClient, name: str, vector_size: int):
    """Create a Qdrant collection with tuned HNSW and payload indexes."""
    # Delete if exists
    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=HNSW_M,
                ef_construct=HNSW_EF_CONSTRUCT,
            ),
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=10000,  # Start indexing after 10k points
        ),
    )

    log.info(f"Created collection: {name} (HNSW m={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCT})")


def build_collection(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    model: SentenceTransformer,
):
    """Build a single Qdrant collection from chunks."""
    if not chunks:
        log.info(f"  No chunks for {collection_name}, skipping")
        return

    vector_size = model.get_sentence_embedding_dimension()
    create_collection(client, collection_name, vector_size)

    log.info(f"  Indexing {len(chunks)} chunks into {collection_name}...")

    def upsert_with_retry(points_batch: list[PointStruct], batch_label: str):
        for attempt in range(1, UPSERT_RETRIES + 1):
            try:
                client.upsert(collection_name=collection_name, points=points_batch)
                return
            except Exception as exc:
                log.warning(
                    f"Qdrant upsert failed for {batch_label} (attempt {attempt}/{UPSERT_RETRIES}): {exc}"
                )
                if attempt >= UPSERT_RETRIES:
                    raise
                time.sleep(UPSERT_BACKOFF_SECONDS * attempt)

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc=collection_name):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]

        texts = [c["chunk_text"] for c in batch]

        # BGE models do NOT use prefix for indexing
        encode_texts = texts

        embeddings = model.encode(
            encode_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []
        for i, c in enumerate(batch):
            point_id = chunk_id_to_uuid(c["chunk_id"])
            payload = {
                "chunk_id": c.get("chunk_id", ""),
                "paper_id": c.get("paper_id", ""),
                "title": (c.get("title", "") or "")[:500],
                "authors": (c.get("authors", "") or "")[:300],
                "categories": c.get("categories", ""),
                "chunk_type": c.get("chunk_type", "text"),
                "modality": c.get("modality", "text"),
                "section_hint": c.get("section_hint", "other"),
                "layer": c.get("layer", "core"),
                "token_count": c.get("token_count", 0),
                "chunk_index": c.get("chunk_index", 0),
                "total_chunks": c.get("total_chunks", 1),
                "chunk_source": c.get("chunk_source", "full_text"),
                "chunk_text": c.get("chunk_text", ""),
            }
            # Add page info if available
            if c.get("page_start") is not None:
                payload["page_start"] = c["page_start"]
            if c.get("page_end") is not None:
                payload["page_end"] = c["page_end"]

            points.append(PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload,
            ))

        for upsert_start in range(0, len(points), UPSERT_BATCH_SIZE):
            upsert_batch = points[upsert_start:upsert_start + UPSERT_BATCH_SIZE]
            batch_label = f"chunks {batch_start + upsert_start}-{batch_start + upsert_start + len(upsert_batch) - 1}"
            upsert_with_retry(upsert_batch, batch_label)

    count = client.get_collection(collection_name).points_count
    log.info(f"  {collection_name}: {count} points indexed")


def main():
    chunks_path = Path(CHUNKS_PATH)
    if not chunks_path.exists():
        log.error(f"Chunks file not found: {chunks_path}")
        log.info("Run 'python ingest/chunking.py' first.")
        return

    if not QDRANT_URL:
        log.error("QDRANT_URL is required for cloud deployment.")
        return

    # Load model
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Initialize Qdrant Cloud
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    log.info(f"Qdrant Cloud initialized at: {QDRANT_URL}")
    log.info(f"Upsert batch size: {UPSERT_BATCH_SIZE}, retries: {UPSERT_RETRIES}")

    # Load chunks
    chunks = load_chunks(str(chunks_path))

    # Build single collection
    build_collection(client, COLLECTION_NAME, chunks, model)

    # Summary
    log.info("\n" + "=" * 50)
    log.info("INDEX BUILD SUMMARY")
    log.info("=" * 50)
    try:
        info = client.get_collection(COLLECTION_NAME)
        log.info(f"  {COLLECTION_NAME:20s}: {info.points_count:,} points")
    except Exception:
        log.info(f"  {COLLECTION_NAME:20s}: 0 points")
    log.info(f"  Embedding model: {EMBEDDING_MODEL}")
    log.info(f"  HNSW config:     m={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCT}")
    log.info(f"  Qdrant url:      {QDRANT_URL}")
    log.info("=" * 50)

    client.close()


if __name__ == "__main__":
    main()
