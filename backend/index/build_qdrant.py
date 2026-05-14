"""
build_qdrant.py — Build Qdrant collections for chunk and document-level retrieval.

Optimized for low-VRAM GPUs (e.g. RTX 3050 Ti 4GB).

Collections:
    - arxiv_text: chunk vectors + payload (dense retrieval + payload recovery)
    - arxiv_docs: one vector per paper from title + abstract (parent in parent–child retrieval)

Chunk build strategy:
    1) Encode lean chunk texts in bounded windows (low peak VRAM).
    2) Upload each window to Qdrant with retries and timing logs.

Document build strategy:
    1) Build one core text per paper (title + abstract).
    2) Batch-encode papers, then batch-upload to arxiv_docs.

Usage:
    python -m index.build_qdrant --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- OpenMP Crash Workaround ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.chunking import build_contextual_text  # noqa: E402
from utils.artifact_schema import write_artifact_manifest  # noqa: E402
from utils.ids import chunk_id_to_uuid, paper_id_to_uuid  # noqa: E402
from utils.runtime import resolve_embedding_model  # noqa: E402
from utils.section_labels import normalize_section_label  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")

EMBEDDING_MODEL = resolve_embedding_model()

# ---------------------------------------------------------------------------
# 4GB RTX 3050 Ti Optimized Defaults
# ---------------------------------------------------------------------------

BATCH_SIZE = int(os.getenv("QDRANT_EMBED_BATCH_SIZE", "64"))
UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128"))
UPSERT_RETRIES = int(os.getenv("QDRANT_UPSERT_RETRIES", "5"))
UPSERT_BACKOFF_SECONDS = float(os.getenv("QDRANT_UPSERT_BACKOFF_SECONDS", "2.0"))
CHUNK_ENCODE_WINDOW = int(os.getenv("QDRANT_CHUNK_ENCODE_WINDOW", "512"))
DOC_EMBED_BATCH = int(os.getenv("QDRANT_DOC_EMBED_BATCH", "128"))
MAX_SEQ_LENGTH = int(os.getenv("QDRANT_MAX_SEQ_LENGTH", "384"))

# HNSW indexing
HNSW_M = 32
HNSW_EF_CONSTRUCT = 400

COLLECTION_CHUNKS = "arxiv_text"
COLLECTION_DOCS = "arxiv_docs"
EMBED_TEXT_VERSION = "lean_title_section_chunk_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunks(chunks_path: str) -> list[dict]:
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    log.info("Loaded %s chunks from %s", len(chunks), chunks_path)
    return chunks


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _chunk_embedding_text(chunk: dict) -> str:
    """
    Lean embedding text for chunk-level retrieval.

    Keep only:
      - title
      - normalized section
      - chunk text

    The richer contextual text remains in the payload for downstream LLM use.
    """
    title = _normalize_text(chunk.get("title", ""))
    section = normalize_section_label(chunk.get("section_hint", "other")).replace("_", " ").title()
    chunk_text = _normalize_text(chunk.get("chunk_text", ""))

    if not chunk_text:
        chunk_text = _normalize_text(chunk.get("contextual_text", ""))

    if title and chunk_text:
        return f"Title: {title}\nSection: {section}\n\n{chunk_text}"
    if title:
        return f"Title: {title}\nSection: {section}"
    return f"Section: {section}\n\n{chunk_text}".strip()


def _contextual_text_for_chunk(chunk: dict) -> str:
    """
    Rich payload text stored in Qdrant for later LLM grounding / inspection.
    """
    contextual_text = chunk.get("contextual_text")
    if contextual_text:
        return contextual_text
    sec = normalize_section_label(chunk.get("section_hint", "other"))
    return build_contextual_text(
        title=chunk.get("title", ""),
        authors=chunk.get("authors", ""),
        categories=chunk.get("categories", ""),
        section_hint=sec,
        chunk_text=chunk.get("chunk_text", ""),
        chunk_index=chunk.get("chunk_index", 0),
        total_chunks=chunk.get("total_chunks", 1),
    )


def _paper_core_embedding_text(plist: list[dict]) -> str:
    """Title + abstract for document-level dense retrieval."""
    if not plist:
        return ""

    ordered = sorted(plist, key=lambda c: int(c.get("chunk_index", 0)))
    first = ordered[0]
    title = _normalize_text(first.get("title", ""))
    abstract = _normalize_text(first.get("paper_abstract", ""))

    if not abstract:
        parts = []
        for c in ordered:
            if normalize_section_label(c.get("section_hint", "")) == "abstract":
                t = _normalize_text(c.get("chunk_text", ""))
                if t:
                    parts.append(t)
        abstract = " ".join(parts).strip()

    if len(abstract) > 10000:
        abstract = abstract[:10000]

    if title and abstract:
        return f"Title: {title}\n\nAbstract: {abstract}"[:12000]
    return (title or abstract)[:12000]


def create_collection(client: QdrantClient, name: str, vector_size: int, resume: bool = False) -> None:
    if resume:
        try:
            if client.collection_exists(name):
                log.info("Collection %s exists. Resuming without deletion.", name)
                return
        except Exception:
            pass

    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        ),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
    )
    log.info("Created collection: %s (HNSW m=%s, ef_construct=%s)", name, HNSW_M, HNSW_EF_CONSTRUCT)


def _get_existing_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """Fetch all existing point IDs in a collection to allow resuming."""
    existing_ids: set[str] = set()
    try:
        if not client.collection_exists(collection_name):
            return existing_ids

        info = client.get_collection(collection_name)
        total_points = info.points_count
        log.info(
            "Fetching %s existing IDs from %s to skip re-indexing...",
            total_points,
            collection_name,
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)

        next_offset = None
        with tqdm(total=total_points, desc=f"Scanning {collection_name}") as pbar:
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=5000,
                    offset=next_offset,
                    with_payload=False,
                    with_vectors=False,
                )
                existing_ids.update(p.id for p in points)
                pbar.update(len(points))
                if next_offset is None:
                    break

        log.info("Found %s existing points in %s.", len(existing_ids), collection_name)
    except Exception as e:
        log.warning("Could not fetch existing IDs for %s: %s", collection_name, e)
    return existing_ids


def _upload_points_with_retry(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    *,
    batch_size: int,
    parallel: int,
    wait: bool,
) -> float:
    """
    Upload points with retry/backoff and return elapsed seconds.
    Uses upload_points because it supports batched parallel ingestion.
    """
    start = time.time()
    last_exc: Exception | None = None

    for attempt in range(1, UPSERT_RETRIES + 1):
        try:
            client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=batch_size,
                parallel=parallel,
                max_retries=1,
                wait=wait,
            )
            return time.time() - start
        except Exception as exc:
            last_exc = exc
            if attempt >= UPSERT_RETRIES:
                break

            sleep_s = UPSERT_BACKOFF_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "Upload failed for %s (attempt %s/%s): %s. Retrying in %.1fs...",
                collection_name,
                attempt,
                UPSERT_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Chunk collection
# ---------------------------------------------------------------------------

def build_chunk_collection(
    client: QdrantClient,
    chunks: list[dict],
    model: SentenceTransformer,
    resume: bool = False,
) -> None:
    if not chunks:
        log.info("No chunks for %s, skipping", COLLECTION_CHUNKS)
        return

    vector_size = model.get_sentence_embedding_dimension()
    create_collection(client, COLLECTION_CHUNKS, vector_size, resume=resume)

    existing_ids = _get_existing_ids(client, COLLECTION_CHUNKS) if resume else set()
    chunks_to_process = [c for c in chunks if chunk_id_to_uuid(c.get("chunk_id", "")) not in existing_ids]

    if not chunks_to_process:
        log.info("All chunks are already indexed in %s. Skipping.", COLLECTION_CHUNKS)
        return

    n = len(chunks_to_process)
    log.info(
        "arxiv_text: encoding + uploading in windows of %s chunks (%s total, %s skipped)...",
        CHUNK_ENCODE_WINDOW,
        n,
        len(chunks) - n,
    )

    encode_times: list[float] = []
    upload_times: list[float] = []

    for win_start in range(0, n, CHUNK_ENCODE_WINDOW):
        batch = chunks_to_process[win_start: win_start + CHUNK_ENCODE_WINDOW]
        texts = [_chunk_embedding_text(c) for c in batch]

        log.info(
            "  [encode] local embeddings %s-%s of %s",
            win_start + 1,
            win_start + len(batch),
            n,
        )

        encode_t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(batch) > 256,
        )
        encode_s = time.time() - encode_t0
        encode_times.append(encode_s)

        points: list[PointStruct] = []
        for i, c in enumerate(batch):
            point_id = chunk_id_to_uuid(c["chunk_id"])
            sec = normalize_section_label(c.get("section_hint", "other"))
            payload = {
                "chunk_id": c.get("chunk_id", ""),
                "paper_id": c.get("paper_id", ""),
                "title": _normalize_text(c.get("title", ""))[:500],
                "authors": _normalize_text(c.get("authors", ""))[:300],
                "categories": c.get("categories", ""),
                "chunk_type": c.get("chunk_type", "text"),
                "modality": c.get("modality", "text"),
                "section_hint": sec,
                "layer": c.get("layer", "core"),
                "token_count": c.get("token_count", 0),
                "chunk_index": c.get("chunk_index", 0),
                "total_chunks": c.get("total_chunks", 1),
                "chunk_source": c.get("chunk_source", "full_text"),
                "chunk_text": c.get("chunk_text", ""),
                "contextual_text": _contextual_text_for_chunk(c),
                "embedding_text_version": EMBED_TEXT_VERSION,
            }
            if c.get("page_start") is not None:
                payload["page_start"] = c["page_start"]
            if c.get("page_end") is not None:
                payload["page_end"] = c["page_end"]

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
            )

        log.info("  [upload] pushing %s points to Qdrant (parallel upsert)...", len(points))
        upload_s = _upload_points_with_retry(
            client,
            COLLECTION_CHUNKS,
            points,
            batch_size=UPSERT_BATCH_SIZE,
            parallel=4,
            wait=True,
        )
        upload_times.append(upload_s)

        log.info(
            "  [timing] window %s-%s: encode=%.1fs upload=%.1fs total=%.1fs",
            win_start + 1,
            win_start + len(batch),
            encode_s,
            upload_s,
            encode_s + upload_s,
        )

    count = client.get_collection(COLLECTION_CHUNKS).points_count
    log.info("%s: %s points indexed", COLLECTION_CHUNKS, count)

    if encode_times:
        log.info(
            "%s encode avg=%.1fs min=%.1fs max=%.1fs",
            COLLECTION_CHUNKS,
            sum(encode_times) / len(encode_times),
            min(encode_times),
            max(encode_times),
        )
    if upload_times:
        log.info(
            "%s upload avg=%.1fs min=%.1fs max=%.1fs",
            COLLECTION_CHUNKS,
            sum(upload_times) / len(upload_times),
            min(upload_times),
            max(upload_times),
        )


# ---------------------------------------------------------------------------
# Document collection
# ---------------------------------------------------------------------------

def build_document_collection(
    client: QdrantClient,
    chunks: list[dict],
    model: SentenceTransformer,
    resume: bool = False,
) -> None:
    """One embedding per paper from title + abstract."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        pid = c.get("paper_id")
        if pid:
            groups[pid].append(c)

    if not groups:
        log.warning("No paper_id groups; skipping %s", COLLECTION_DOCS)
        return

    vector_size = model.get_sentence_embedding_dimension()
    create_collection(client, COLLECTION_DOCS, vector_size, resume=resume)

    existing_ids = _get_existing_ids(client, COLLECTION_DOCS) if resume else set()
    groups_to_process = {
        pid: plist for pid, plist in groups.items()
        if paper_id_to_uuid(pid) not in existing_ids
    }

    if not groups_to_process:
        log.info("All documents are already indexed in %s. Skipping.", COLLECTION_DOCS)
        return

    log.info("arxiv_docs: %s parent vectors (title+abstract)...", len(groups_to_process))
    log.info("Step 1/2: Generating doc embeddings locally using title and abstract...")

    points: list[PointStruct] = []
    doc_encode_times: list[float] = []

    items = list(groups_to_process.items())
    for batch_start in tqdm(range(0, len(items), DOC_EMBED_BATCH), desc="Doc Embeddings"):
        batch_items = items[batch_start: batch_start + DOC_EMBED_BATCH]
        doc_texts = [_paper_core_embedding_text(plist) for _, plist in batch_items]

        encode_t0 = time.time()
        embs = model.encode(
            doc_texts,
            batch_size=min(BATCH_SIZE, DOC_EMBED_BATCH),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        encode_s = time.time() - encode_t0
        doc_encode_times.append(encode_s)

        for idx, (paper_id, plist) in enumerate(batch_items):
            first = plist[0]
            abstract_chunks = [
                c.get("chunk_text", "")
                for c in plist
                if normalize_section_label(c.get("section_hint", "")) == "abstract"
            ]
            abstract_text = " ".join(abstract_chunks) if abstract_chunks else first.get("paper_abstract") or first.get("chunk_text", "")
            abstract_text = _normalize_text(abstract_text)

            payload = {
                "paper_id": paper_id,
                "title": _normalize_text(first.get("title", ""))[:500],
                "authors": _normalize_text(first.get("authors", ""))[:300],
                "categories": first.get("categories", ""),
                "layer": first.get("layer", "core"),
                "chunk_count": len(plist),
                "abstract": abstract_text[:4000],
                "embedding_text_version": EMBED_TEXT_VERSION,
            }

            vec = embs[idx]
            vlist = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            points.append(
                PointStruct(
                    id=paper_id_to_uuid(paper_id),
                    vector=vlist,
                    payload=payload,
                )
            )

        log.info(
            "  [doc encode] batch %s-%s of %s papers: %.1fs",
            batch_start + 1,
            batch_start + len(batch_items),
            len(items),
            encode_s,
        )

    log.info("Step 2/2: Uploading %s parent vectors to Qdrant using parallel connections...", len(points))
    upload_s = _upload_points_with_retry(
        client,
        COLLECTION_DOCS,
        points,
        batch_size=UPSERT_BATCH_SIZE,
        parallel=4,
        wait=True,
    )

    count = client.get_collection(COLLECTION_DOCS).points_count
    log.info("%s: %s points indexed", COLLECTION_DOCS, count)

    if doc_encode_times:
        log.info(
            "%s encode avg=%.1fs min=%.1fs max=%.1fs",
            COLLECTION_DOCS,
            sum(doc_encode_times) / len(doc_encode_times),
            min(doc_encode_times),
            max(doc_encode_times),
        )
    log.info("%s upload total=%.1fs", COLLECTION_DOCS, upload_s)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_corpus_version(data_dir: Path) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = data_dir / "corpus_version.txt"
    path.write_text(
        f"corpus_build_utc={stamp}\n"
        f"embedding_model={EMBEDDING_MODEL}\n"
        f"collections={COLLECTION_CHUNKS},{COLLECTION_DOCS}\n"
        f"embedding_text_version={EMBED_TEXT_VERSION}\n",
        encoding="utf-8",
    )
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(resume: bool = False) -> None:
    chunks_path = Path(CHUNKS_PATH)
    if not chunks_path.exists():
        log.error("Chunks file not found: %s", chunks_path)
        log.info("Run chunking first.")
        return

    if not QDRANT_URL:
        log.error("QDRANT_URL is required.")
        return

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    try:
        model.max_seq_length = MAX_SEQ_LENGTH
        log.info("Embedding max_seq_length: %s", model.max_seq_length)
    except Exception:
        log.warning("Could not set model.max_seq_length; continuing with model default.")

    log.info("Embedding dimension: %s", model.get_sentence_embedding_dimension())

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    log.info("Qdrant client ready: %s", QDRANT_URL)

    chunks = load_chunks(str(chunks_path))
    build_chunk_collection(client, chunks, model, resume=resume)
    build_document_collection(client, chunks, model, resume=resume)

    data_dir = chunks_path.parent
    _write_corpus_version(data_dir)
    write_artifact_manifest(
        data_dir,
        extra={
            "qdrant_collections": [COLLECTION_CHUNKS, COLLECTION_DOCS],
            "chunk_encode_window": CHUNK_ENCODE_WINDOW,
            "embed_batch_size": BATCH_SIZE,
            "upsert_batch_size": UPSERT_BATCH_SIZE,
            "doc_embed_batch": DOC_EMBED_BATCH,
            "max_seq_length": MAX_SEQ_LENGTH,
            "embedding_text_version": EMBED_TEXT_VERSION,
            "gpu_target": "rtx_3050_ti_4gb_optimized",
        },
    )

    log.info("\n%s", "=" * 50)
    log.info("INDEX BUILD SUMMARY")
    log.info("%s", "=" * 50)
    for name in (COLLECTION_CHUNKS, COLLECTION_DOCS):
        try:
            info = client.get_collection(name)
            log.info("  %-20s: %s points", name, f"{info.points_count:,}")
        except Exception:
            log.info("  %-20s: (unavailable)", name)
    log.info("  Embedding model: %s", EMBEDDING_MODEL)
    log.info("  Qdrant url:      %s", QDRANT_URL)
    log.info("  Embedding text:  %s", EMBED_TEXT_VERSION)
    log.info("%s", "=" * 50)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant chunk and document collections.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume indexing by skipping existing IDs in Qdrant.",
    )
    args = parser.parse_args()
    main(resume=args.resume)
