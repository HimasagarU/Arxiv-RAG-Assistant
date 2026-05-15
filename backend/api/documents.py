"""
documents.py — Document addition and ingestion status endpoints.

Endpoints:
    POST /documents/add              — Submit a paper for async ingestion
    GET  /documents/status/{job_id}  — Poll ingestion status
    GET  /documents                  — List user's added documents

The background ingestion task reuses the existing ingest pipeline
(ingest_arxiv.py) and stores PDFs in Cloudflare R2.
"""

import json
import logging
import os
import re
import sys
import threading
import types
import time
import hashlib
import zipfile
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.cache import bump_query_cache_buster_sync
from db.app_database import get_app_db
from db.app_models import DocumentJob, User
from index.lexical_text import build_lexical_index_text
from utils.ids import chunk_id_to_uuid, normalize_arxiv_paper_id, paper_id_to_uuid
from utils.metadata_normalize import normalize_published
from utils.section_labels import normalize_section_label

load_dotenv()

log = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

JOB_ACTIVE_STATUSES = frozenset({"queued", "downloading", "chunking", "embedding"})
JOB_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
ARXIV_EXPORT_API = "https://export.arxiv.org/api/query"
ARTIFACT_BUNDLE_FILES = ("bm25_v1.pkl", "chunks_meta.jsonl", "chunks_text.jsonl", "papers_meta.json")
_artifact_update_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AddDocumentRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=4, max_length=64,
                          description="ArXiv paper ID (e.g., '2301.12345')")
    pdf_url: Optional[str] = Field(default=None, max_length=512)


class DocumentJobResponse(BaseModel):
    id: str
    arxiv_id: str
    title: Optional[str]
    status: str
    error_message: Optional[str]
    chunks_created: int
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_to_response(job: DocumentJob) -> DocumentJobResponse:
    return DocumentJobResponse(
        id=str(job.id),
        arxiv_id=job.arxiv_id,
        title=job.title,
        status=job.status,
        error_message=job.error_message,
        chunks_created=job.chunks_created,
        created_at=job.created_at.isoformat() if job.created_at else "",
        updated_at=job.updated_at.isoformat() if job.updated_at else "",
    )


def _ensure_backend_imports() -> None:
    """Make the backend root and ingest package importable at runtime."""
    backend_root = Path(__file__).resolve().parent.parent
    backend_root_str = str(backend_root)
    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)

    ingest_dir = backend_root / "ingest"
    if ingest_dir.is_dir() and "ingest" not in sys.modules:
        ingest_pkg = types.ModuleType("ingest")
        ingest_pkg.__path__ = [str(ingest_dir)]
        sys.modules["ingest"] = ingest_pkg


def _get_job_status(job_id: str) -> Optional[str]:
    """Return the current persisted status for a job, or None if unavailable."""
    app_db_url = os.getenv("APP_DATABASE_URL", "")
    if not app_db_url:
        return None

    import psycopg
    from psycopg.rows import dict_row

    try:
        with psycopg.connect(app_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM document_jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
                if row:
                    return row["status"]
    except Exception as e:
        log.warning(f"Failed to read job {job_id} status: {e}")
    return None


def _is_cancelled(job_id: str) -> bool:
    return _get_job_status(job_id) == "cancelled"


def _wait_for_retriever(timeout_s: float = 180.0, poll_s: float = 2.0):
    """Wait until global HybridRetriever is ready (embed_model + Qdrant client)."""
    import time as time_mod

    from api.app import _state

    deadline = time_mod.time() + timeout_s
    while time_mod.time() < deadline:
        r = _state.get("retriever")
        if r is not None and getattr(r, "embed_model", None) and getattr(r, "qdrant_client", None):
            return r
        time_mod.sleep(poll_s)
    return None


def _qdrant_has_paper_chunks(qdrant_client, collection: str, paper_id: str) -> bool:
    """Return True if at least one point exists for this paper_id in the collection."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        flt = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])
        pts, _ = qdrant_client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(pts)
    except Exception as exc:
        log.warning("Qdrant scroll check failed for %s: %s", paper_id, exc)
        return False


def _standalone_embed_and_qdrant():
    """Build SentenceTransformer + Qdrant when the global retriever is not yet initialized."""
    import torch
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from utils.runtime import resolve_embedding_model

    url = (os.getenv("QDRANT_URL") or "").strip()
    if not url:
        return None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(resolve_embedding_model(), device=device)
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return model, client


def _rebuild_bm25_safe():
    """Rebuild BM25 artifacts from ``chunks.jsonl`` (same as CLI ``index.build_bm25``)."""
    import importlib

    try:
        mod = importlib.import_module("index.build_bm25")
        mod.main()
        log.info("[Ingest] BM25 artifact rebuild finished.")
    except Exception as e:
        log.error("[Ingest] BM25 rebuild failed: %s", e)
        raise


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", (text or "").lower()).split()


def _chunk_meta_record(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "paper_id": chunk["paper_id"],
        "title": chunk.get("title", ""),
        "authors": chunk.get("authors", ""),
        "categories": chunk.get("categories", ""),
        "chunk_type": chunk.get("chunk_type", "text"),
        "modality": chunk.get("modality", "text"),
        "section_hint": normalize_section_label(chunk.get("section_hint", "other")),
        "layer": chunk.get("layer", "core"),
        "chunk_index": chunk.get("chunk_index", 0),
        "total_chunks": chunk.get("total_chunks", 1),
        "chunk_source": chunk.get("chunk_source", "full_text"),
    }


def _chunk_text_record(chunk: dict) -> dict:
    chunk_norm = {
        **chunk,
        "section_hint": normalize_section_label(chunk.get("section_hint", "other")),
    }
    lexical_doc = build_lexical_index_text(chunk_norm)
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk.get("chunk_text", ""),
        "contextual_text": chunk.get("contextual_text", chunk.get("chunk_text", "")),
        "lexical_index_text": lexical_doc,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _upload_artifact_bundle_to_r2(data_dir: Path) -> bool:
    """Zip refreshed retrieval artifacts and upload the bundle/checksum to R2."""
    from ingest.r2_storage import R2Storage

    r2 = R2Storage()
    if not r2.is_available:
        log.warning("[Ingest] R2 unavailable; refreshed BM25 artifacts remain local only.")
        return False

    zip_name = os.getenv("ARTIFACT_ZIP_NAME", "artifacts_v1.zip")
    zip_path = data_dir / zip_name
    sha_path = data_dir / f"{zip_name}.sha256"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name in ARTIFACT_BUNDLE_FILES:
            path = data_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Cannot upload artifact bundle; missing {path}")
            zipf.write(path, arcname=name)

    digest = _sha256_file(zip_path)
    sha_path.write_text(digest + "\n", encoding="utf-8")

    zip_key = r2.upload_bytes(zip_name, zip_path.read_bytes(), content_type="application/zip")
    sha_key = r2.upload_bytes(sha_path.name, sha_path.read_bytes(), content_type="text/plain")
    if zip_key and sha_key:
        log.info("[Ingest] Uploaded refreshed artifact bundle to R2: %s", zip_name)
        return True
    log.warning("[Ingest] Artifact bundle upload to R2 was incomplete.")
    return False


def _refresh_bm25_artifacts_with_new_chunks(paper_meta: dict, chunks: list[dict]) -> bool:
    """Append new chunks to the full local R2 artifact set, rebuild BM25, and upload it."""
    from rank_bm25 import BM25Okapi
    from utils.artifact_schema import write_artifact_manifest

    with _artifact_update_lock:
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        data_dir.mkdir(parents=True, exist_ok=True)

        required = [data_dir / "chunks_meta.jsonl", data_dir / "chunks_text.jsonl", data_dir / "papers_meta.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            log.warning("[Ingest] Cannot refresh BM25 artifacts; missing full local artifacts: %s", ", ".join(missing))
            return False

        text_by_chunk: dict[str, dict] = {}
        with open(data_dir / "chunks_text.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid:
                    text_by_chunk[cid] = rec

        new_chunk_ids = {c["chunk_id"] for c in chunks}
        meta_records: list[dict] = []
        with open(data_dir / "chunks_meta.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("chunk_id") not in new_chunk_ids:
                    meta_records.append(rec)

        for chunk in chunks:
            text_by_chunk[chunk["chunk_id"]] = _chunk_text_record(chunk)
            meta_records.append(_chunk_meta_record(chunk))

        papers_path = data_dir / "papers_meta.json"
        with open(papers_path, "r", encoding="utf-8") as f:
            papers_meta = json.load(f)
        paper_id = paper_meta["paper_id"]
        existing_paper = papers_meta.get(paper_id, {})
        papers_meta[paper_id] = {
            **existing_paper,
            "title": paper_meta.get("title", existing_paper.get("title", "")),
            "authors": paper_meta.get("authors", existing_paper.get("authors", "")),
            "categories": paper_meta.get("categories", existing_paper.get("categories", "")),
            "published": normalize_published(paper_meta.get("published")) or existing_paper.get("published", ""),
            "layer": paper_meta.get("layer", existing_paper.get("layer", "core")),
            "pdf_url": paper_meta.get("pdf_url", existing_paper.get("pdf_url", "")),
        }

        corpus_tokens: list[list[str]] = []
        tmp_meta = (data_dir / "chunks_meta.jsonl").with_suffix(".jsonl.tmp")
        tmp_text = (data_dir / "chunks_text.jsonl").with_suffix(".jsonl.tmp")
        tmp_papers = papers_path.with_suffix(".json.tmp")
        tmp_bm25 = (data_dir / "bm25_v1.pkl").with_suffix(".pkl.tmp")

        with open(tmp_meta, "w", encoding="utf-8") as f_meta, open(tmp_text, "w", encoding="utf-8") as f_text:
            for meta in meta_records:
                cid = meta.get("chunk_id")
                text_rec = text_by_chunk.get(cid, {})
                lexical_doc = text_rec.get("lexical_index_text") or text_rec.get("contextual_text") or text_rec.get("text", "")
                corpus_tokens.append(_tokenize_for_bm25(lexical_doc))
                f_meta.write(json.dumps(meta, default=str) + "\n")
                f_text.write(json.dumps(text_rec, default=str) + "\n")

        with open(tmp_papers, "w", encoding="utf-8") as f:
            json.dump(papers_meta, f, indent=2, default=str)

        log.info("[Ingest] Rebuilding BM25 over %s full artifact chunks...", len(corpus_tokens))
        import joblib
        joblib.dump(BM25Okapi(corpus_tokens), tmp_bm25, compress=3)

        tmp_meta.replace(data_dir / "chunks_meta.jsonl")
        tmp_text.replace(data_dir / "chunks_text.jsonl")
        tmp_papers.replace(papers_path)
        tmp_bm25.replace(data_dir / "bm25_v1.pkl")

        write_artifact_manifest(
            data_dir,
            extra={
                "bm25_path": "bm25_v1.pkl",
                "chunks_meta": "chunks_meta.jsonl",
                "chunks_text": "chunks_text.jsonl",
                "papers_meta": "papers_meta.json",
                "chunk_count": len(corpus_tokens),
                "paper_count": len(papers_meta),
                "lexical_fields": ["title", "authors", "categories", "section", "body"],
            },
        )

        _upload_artifact_bundle_to_r2(data_dir)
        log.info("[Ingest] BM25 artifacts refreshed with %s new chunks.", len(chunks))
        return True


def _hot_reload_bm25_into_retriever():
    try:
        from api.app import _state

        r = _state.get("retriever")
        if not r:
            return
        import joblib

        bm25_path = Path(os.getenv("DATA_DIR", "data")) / "bm25_v1.pkl"
        if bm25_path.is_file():
            bm25 = joblib.load(bm25_path)
            bm25_docs = getattr(bm25, "corpus_size", None)
            if not isinstance(bm25_docs, int):
                doc_len = getattr(bm25, "doc_len", None)
                bm25_docs = len(doc_len) if doc_len is not None else 0
            meta_docs = len(getattr(r, "chunks_meta", []) or [])
            if bm25_docs and meta_docs and bm25_docs != meta_docs:
                r.bm25 = None
                r.bm25_dirty = True
                log.warning(
                    "[Ingest] Refusing BM25 hot-reload: bm25 has %s docs but chunks_meta has %s rows.",
                    bm25_docs,
                    meta_docs,
                )
                return
            r.bm25 = bm25
            r.bm25_dirty = False
            log.info("[Ingest] Hot-reloaded BM25 from %s", bm25_path)
    except Exception as ex:
        log.warning("[Ingest] BM25 hot-reload skipped: %s", ex)


def _fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata from arXiv with useful failure diagnostics."""
    import feedparser
    import requests

    params = {"id_list": arxiv_id}
    last_error = ""
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                ARXIV_EXPORT_API,
                params=params,
                timeout=30,
                headers={"User-Agent": "ArxivRagAssistant/1.0 (paper ingestion)"},
            )
            if resp.status_code == 429 and attempt < 3:
                wait_s = 5 * attempt
                log.warning(
                    "[Ingest] arXiv metadata rate limited for %s; retrying in %ss",
                    arxiv_id,
                    wait_s,
                )
                time.sleep(wait_s)
                continue
            if resp.status_code != 200:
                last_error = f"arXiv metadata HTTP {resp.status_code}"
                log.warning("[Ingest] %s for %s", last_error, arxiv_id)
                continue

            feed = feedparser.parse(resp.text)
            if feed.bozo:
                bozo_msg = str(getattr(feed, "bozo_exception", ""))[:200]
                if bozo_msg:
                    log.warning("[Ingest] arXiv feed parse warning for %s: %s", arxiv_id, bozo_msg)

            entries = getattr(feed, "entries", []) or []
            if entries:
                entry = entries[0]
                entry_id = (entry.get("id") or "").lower()
                if arxiv_id in entry_id:
                    return {
                        "title": entry.get("title", "").replace("\n", " ").strip(),
                        "abstract": entry.get("summary", "").strip(),
                        "authors": ", ".join(a.get("name", "") for a in entry.get("authors", [])),
                        "categories": ", ".join(t.get("term", "") for t in entry.get("tags", [])),
                        "published_raw": entry.get("published") or entry.get("updated") or "",
                    }
                last_error = f"arXiv returned a different entry ({entry_id or 'missing id'})"
            else:
                last_error = "arXiv metadata feed returned no entries"
        except requests.exceptions.Timeout:
            last_error = "arXiv metadata request timed out"
            log.warning("[Ingest] %s for %s", last_error, arxiv_id)
        except Exception as exc:
            last_error = f"arXiv metadata request failed: {exc}"
            log.warning("[Ingest] %s", last_error)

    # Last-resort metadata from the abs page. This avoids false "not found" reports
    # when the export API lags or is blocked by a proxy, while still failing clearly.
    try:
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        resp = requests.get(
            abs_url,
            timeout=30,
            headers={"User-Agent": "ArxivRagAssistant/1.0 (paper ingestion)"},
        )
        if resp.status_code == 200 and f"arXiv:{arxiv_id}" in resp.text:
            title_match = re.search(r"<h1[^>]*class=\"title[^>]*>\s*<span[^>]*>Title:</span>\s*(.*?)</h1>", resp.text, re.S)
            abstract_match = re.search(r"<blockquote[^>]*class=\"abstract[^>]*>\s*<span[^>]*>Abstract:</span>\s*(.*?)</blockquote>", resp.text, re.S)
            authors_match = re.search(r"<div[^>]*class=\"authors[^>]*>\s*<span[^>]*>Authors:</span>\s*(.*?)</div>", resp.text, re.S)
            cats_match = re.search(r"<td[^>]*class=\"tablecell subjects\"[^>]*>\s*(.*?)</td>", resp.text, re.S)

            def _clean_html(value: str) -> str:
                value = re.sub(r"<[^>]+>", " ", value or "")
                value = re.sub(r"\s+", " ", value)
                return value.strip()

            return {
                "title": _clean_html(title_match.group(1)) if title_match else arxiv_id,
                "abstract": _clean_html(abstract_match.group(1)) if abstract_match else "",
                "authors": _clean_html(authors_match.group(1)) if authors_match else "",
                "categories": _clean_html(cats_match.group(1)) if cats_match else "",
                "published_raw": "",
            }
        if resp.status_code == 404:
            raise ValueError(f"Paper {arxiv_id} was not found on arXiv abs page.")
        last_error = f"{last_error}; abs page HTTP {resp.status_code}"
    except ValueError:
        raise
    except Exception as exc:
        last_error = f"{last_error}; abs page fallback failed: {exc}"

    raise RuntimeError(last_error or f"Could not fetch arXiv metadata for {arxiv_id}")


def _download_arxiv_pdf(arxiv_id: str, pdf_url: Optional[str], year: Optional[str] = None) -> tuple[bytes, str]:
    import requests

    try:
        from storage.local_pdf_store import LocalPDFStore

        local_store = LocalPDFStore()
        local_path = local_store.download_pdf(
            arxiv_id,
            pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            year=year,
            timeout=int(os.getenv("PDF_TIMEOUT", "60")),
        )
        if local_path:
            cached = Path(local_path).read_bytes()
            if cached and len(cached) >= 1024:
                log.info("[Ingest] Using local PDF cache for %s: %s", arxiv_id, local_path)
                return cached, pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except Exception as exc:
        log.warning("[Ingest] Local PDF store path failed for %s; direct download fallback: %s", arxiv_id, exc)

    candidates = []
    if pdf_url:
        candidates.append(pdf_url)
    candidates.extend([
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{arxiv_id}",
    ])

    seen = set()
    last_error = ""
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        try:
            log.info("[Ingest] Downloading PDF for %s from %s", arxiv_id, url)
            resp = requests.get(
                url,
                timeout=120,
                headers={"User-Agent": "ArxivRagAssistant/1.0 (paper ingestion)"},
            )
            content_type = (resp.headers.get("content-type") or "").lower()
            if resp.status_code == 200 and len(resp.content or b"") >= 1024:
                if "pdf" not in content_type and not resp.content.startswith(b"%PDF"):
                    last_error = f"download from {url} was not a PDF ({content_type or 'unknown content type'})"
                    log.warning("[Ingest] %s", last_error)
                    continue
                return resp.content, url
            last_error = f"PDF download failed from {url} (HTTP {resp.status_code}, {len(resp.content or b'')} bytes)"
            log.warning("[Ingest] %s", last_error)
        except requests.exceptions.Timeout:
            last_error = f"PDF download timed out from {url}"
            log.warning("[Ingest] %s", last_error)
        except Exception as exc:
            last_error = f"PDF download failed from {url}: {exc}"
            log.warning("[Ingest] %s", last_error)

    raise RuntimeError(last_error or f"Could not download PDF for {arxiv_id}")

def _run_ingestion(job_id: str, arxiv_id: str, pdf_url: Optional[str] = None):
    """
    Background thread: downloads, chunks, embeds a paper.
    Updates the DocumentJob status at each stage.

    This runs in a separate thread with its own sync DB session
    because FastAPI BackgroundTasks don't support async well
    on HuggingFace Spaces.
    """
    _ensure_backend_imports()

    import psycopg
    from psycopg.rows import dict_row

    arxiv_id = normalize_arxiv_paper_id(arxiv_id)
    if not arxiv_id or not re.match(r"^\d{4}\.\d{4,5}$", arxiv_id):
        log.error("[Ingest] Invalid arXiv id after normalization: %s", arxiv_id)
        return

    app_db_url = os.getenv("APP_DATABASE_URL", "")
    if not app_db_url:
        log.error("APP_DATABASE_URL not set, cannot update job status.")
        return

    def _update_job(status: str, error: Optional[str] = None,
                    title: Optional[str] = None, chunks: int = 0):
        """Update job status in the app database (sync)."""
        try:
            with psycopg.connect(app_db_url, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    sql = """
                        UPDATE document_jobs
                        SET status = %s, error_message = %s,
                            updated_at = NOW()
                    """
                    params = [status, error]
                    if title:
                        sql += ", title = %s"
                        params.append(title)
                    if chunks > 0:
                        sql += ", chunks_created = %s"
                        params.append(chunks)
                    sql += " WHERE id = %s"
                    params.append(job_id)
                    cur.execute(sql, params)
                conn.commit()
        except Exception as e:
            log.error(f"Failed to update job {job_id}: {e}")

    try:
        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} was cancelled before start.")
            return

        # Stage 1: Downloading
        _update_job("downloading")
        log.info(f"[Ingest] Downloading paper {arxiv_id}...")

        # Fetch metadata from ArXiv API. Use HTTPS; some hosted proxies block plain HTTP
        # and an empty/blocked response used to be mislabeled as "not found".
        try:
            metadata = _fetch_arxiv_metadata(arxiv_id)
        except ValueError as exc:
            _update_job("failed", error=str(exc))
            return
        except Exception as exc:
            _update_job("failed", error=f"Could not fetch arXiv metadata: {str(exc)[:400]}")
            return

        paper_title = metadata["title"]
        abstract = metadata["abstract"]
        authors = metadata["authors"]
        categories = metadata["categories"]
        published_raw = metadata["published_raw"]
        published_iso = normalize_published(published_raw)

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled after metadata fetch.")
            return

        try:
            pub_year = published_iso[:4] if published_iso else None
            pdf_bytes, actual_pdf_url = _download_arxiv_pdf(arxiv_id, pdf_url, year=pub_year)
        except Exception as exc:
            _update_job("failed", error=str(exc)[:500])
            return

        _update_job("downloading", title=paper_title)

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled after PDF download.")
            return

        # Upload PDF to R2
        try:
            from ingest.r2_storage import R2Storage
            r2 = R2Storage()
            if r2.is_available:
                r2.upload_pdf(arxiv_id, pdf_bytes)
                log.info(f"[Ingest] PDF uploaded to R2: {arxiv_id}")
        except Exception as e:
            log.warning(f"R2 upload failed (non-fatal): {e}")

        # Stage 2: Chunking
        _update_job("chunking")
        log.info(f"[Ingest] Chunking paper {arxiv_id}...")

        # Extract text from PDF using the same extractor as the offline ingest path
        # (PyMuPDF first, pdfplumber fallback).
        from ingest.ingest_arxiv import extract_full_text_from_pdf

        full_text = extract_full_text_from_pdf(pdf_bytes)
        if full_text:
            full_text = full_text.replace("\x00", "")

        if not full_text.strip():
            _update_job("failed", error="Could not extract text from PDF.")
            return

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled after text extraction.")
            return

        # Chunk the text using the existing pipeline
        from ingest.chunking import chunk_paper, get_tokenizer
        tokenizer = get_tokenizer()

        paper_dict = {
            "paper_id": arxiv_id,
            "title": paper_title,
            "authors": authors,
            "categories": categories,
            "full_text": full_text,
            "abstract": abstract,
            "published": published_iso or published_raw,
            "layer": "core",
        }
        chunks = chunk_paper(paper_dict, tokenizer)

        if not chunks:
            _update_job("failed", error="No chunks produced from text.")
            return

        log.info(f"[Ingest] Produced {len(chunks)} chunks for {arxiv_id}")

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled after chunking.")
            return

        # Stage 3: Embedding + storing
        _update_job("embedding")
        log.info(f"[Ingest] Embedding {len(chunks)} chunks for {arxiv_id}...")

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled before persistence.")
            return

        # Store paper in Neon DB (corpus)
        from db.database import get_db
        db = get_db()
        db.run_migrations()
        db.upsert_paper({
            "paper_id": arxiv_id,
            "title": paper_title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": actual_pdf_url,
            "published": published_iso,
            "updated": None,
            "full_text": "",
            "download_status": "downloaded",
            "parse_status": "parsed",
            "is_seed": False,
            "layer": "core",
            "source": "user_upload",
        })
        for ch in chunks:
            db.insert_chunk(ch)
        db.commit()

        # Persist chunks locally for offline rebuilds
        try:
            chunks_path = Path(os.getenv("DATA_DIR", "data")) / "chunks.jsonl"
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chunks_path, "a", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, default=str) + "\n")
            log.info(f"[Ingest] Appended {len(chunks)} chunks to {chunks_path}")
        except Exception as e:
            log.error(f"[Ingest] Failed to append chunks to offline JSONL: {e}")

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled after local persistence.")
            return

        # Embed and upsert to Qdrant (wait for retriever or use standalone clients)
        from api.retrieval import COLLECTION_TEXT, COLLECTION_DOCS
        from index.build_qdrant import _chunk_embedding_text, _paper_core_embedding_text
        from qdrant_client.models import PointStruct

        retriever = _wait_for_retriever(timeout_s=float(os.getenv("INGEST_RETRIEVER_WAIT_S", "180")))
        embed_model = None
        qdrant_client = None
        collections_for_docs = set()

        if retriever is not None:
            embed_model = retriever.embed_model
            qdrant_client = retriever.qdrant_client
            collections_for_docs = set(retriever.collections.keys()) if retriever.collections else set()
            log.info("[Ingest] Using global HybridRetriever for embedding/Qdrant.")
        else:
            log.warning("[Ingest] Retriever not ready after wait; loading standalone embedder for this job.")
            embed_model, qdrant_client = _standalone_embed_and_qdrant()
            if embed_model is None or qdrant_client is None:
                _update_job(
                    "failed",
                    error="QDRANT_URL not set or retriever never became ready; cannot embed. "
                    "Ensure the API finished loading indexes and Qdrant is configured.",
                )
                return
            try:
                cols = qdrant_client.get_collections().collections
                collections_for_docs = {c.name for c in cols}
            except Exception:
                collections_for_docs = set()

        qdrant_ok = False
        try:
            texts = [_chunk_embedding_text(c) for c in chunks]

            INGEST_EMBED_BATCH_SIZE = int(os.getenv("INGEST_EMBED_BATCH_SIZE", "32"))
            all_embeddings = []
            log.info(f"[Ingest] Embedding {len(texts)} chunks in batches of {INGEST_EMBED_BATCH_SIZE}...")

            for i in range(0, len(texts), INGEST_EMBED_BATCH_SIZE):
                batch_texts = texts[i : i + INGEST_EMBED_BATCH_SIZE]
                batch_embs = embed_model.encode(
                    batch_texts,
                    batch_size=INGEST_EMBED_BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                all_embeddings.append(batch_embs)

            embeddings = np.vstack(all_embeddings)

            points = []
            for i, chunk in enumerate(chunks):
                point_id = chunk_id_to_uuid(chunk["chunk_id"])
                sec = normalize_section_label(chunk.get("section_hint", "other"))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embeddings[i].tolist(),
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "paper_id": chunk["paper_id"],
                            "title": chunk.get("title", ""),
                            "authors": chunk.get("authors", ""),
                            "categories": chunk.get("categories", ""),
                            "chunk_text": chunk["chunk_text"],
                            "contextual_text": chunk.get("contextual_text", chunk["chunk_text"]),
                            "chunk_type": chunk.get("chunk_type", "text"),
                            "modality": "text",
                            "section_hint": sec,
                            "layer": chunk.get("layer", "core"),
                            "token_count": chunk.get("token_count", 0),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "total_chunks": chunk.get("total_chunks", 1),
                            "chunk_source": chunk.get("chunk_source", "full_text"),
                        },
                    )
                )

            batch_size = 100
            for j in range(0, len(points), batch_size):
                qdrant_client.upsert(
                    collection_name=COLLECTION_TEXT,
                    points=points[j : j + batch_size],
                )

            log.info(f"[Ingest] Upserted {len(points)} vectors to Qdrant collection {COLLECTION_TEXT}")
            qdrant_ok = True

            # Parent–child: one vector per paper in arxiv_docs when collection exists
            if COLLECTION_DOCS in collections_for_docs:
                try:
                    doc_text = _paper_core_embedding_text(chunks)
                    if doc_text.strip():
                        demb = embed_model.encode(
                            [doc_text],
                            batch_size=1,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        doc_vec = demb[0].tolist()
                        doc_point = PointStruct(
                            id=paper_id_to_uuid(arxiv_id),
                            vector=doc_vec,
                            payload={
                                "paper_id": arxiv_id,
                                "title": (paper_title or "")[:500],
                                "authors": (authors or "")[:300],
                                "categories": categories or "",
                                "layer": "core",
                                "chunk_count": len(chunks),
                                "abstract": (abstract or "")[:4000],
                            },
                        )
                        qdrant_client.upsert(collection_name=COLLECTION_DOCS, points=[doc_point])
                        log.info("[Ingest] Upserted parent vector into %s", COLLECTION_DOCS)
                except Exception as doc_exc:
                    log.warning("[Ingest] arxiv_docs upsert skipped: %s", doc_exc)

        except Exception as e:
            log.exception("[Ingest] Qdrant upsert failed: %s", e)
            _update_job("failed", error=f"Vector store update failed: {str(e)[:500]}")
            return

        if not qdrant_ok:
            _update_job("failed", error="No vectors were written to Qdrant.")
            return

        # Update in-memory HybridRetriever metadata + BM25 alignment (when available)
        try:
            from api.app import _state
            if _state.get("retriever") is not None:
                _state["retriever"].add_paper(paper_dict, chunks, persist=False)
                log.info("[Ingest] Updated in-memory HybridRetriever metadata for %s", arxiv_id)
        except Exception as e:
            log.warning("[Ingest] In-memory retriever update failed (non-fatal): %s", e)

        if _is_cancelled(job_id):
            log.info(f"[Ingest] Job {job_id} cancelled after vector store updates.")
            return

        _update_job("done", title=paper_title, chunks=len(chunks))
        log.info(f"[Ingest] Paper {arxiv_id} ingestion complete ({len(chunks)} chunks).")
        bump_query_cache_buster_sync()

        if os.getenv("INGEST_UPDATE_BM25_ARTIFACTS", "true").lower() in ("1", "true", "yes"):
            def _refresh_bm25_artifacts():
                try:
                    refreshed = _refresh_bm25_artifacts_with_new_chunks(paper_dict, chunks)
                    if refreshed:
                        _hot_reload_bm25_into_retriever()
                except Exception as exc:
                    log.warning("[Ingest] BM25/R2 artifact refresh failed: %s", exc)

            threading.Thread(target=_refresh_bm25_artifacts, daemon=True).start()
        else:
            log.info(
                "[Ingest] Skipping BM25/R2 artifact refresh after on-demand ingest "
                "(INGEST_UPDATE_BM25_ARTIFACTS=false)."
            )

    except Exception as e:
        log.error(f"[Ingest] Fatal error for {arxiv_id}: {e}")
        _update_job("failed", error=str(e)[:500])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/add", response_model=DocumentJobResponse, status_code=202)
async def add_document(
    body: AddDocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """Submit a paper for async ingestion. Returns a job ID for status polling."""
    canonical = normalize_arxiv_paper_id(body.arxiv_id)
    if not canonical or not re.match(r"^\d{4}\.\d{4,5}$", canonical):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ArXiv paper id. Use the form YYYY.NNNNN (e.g. 2301.12345).",
        )

    # Skip ingestion only when corpus already has chunks in Postgres and (if Qdrant is configured) vectors exist.
    try:
        from api.app import _state
        from api.retrieval import COLLECTION_TEXT
        from db.database import get_db

        neon_db = get_db()
        chunk_count = neon_db.count_chunks_for_paper(canonical)
        retriever = _state.get("retriever")
        qdrant_configured = bool((os.getenv("QDRANT_URL") or "").strip())
        has_vectors = False
        if qdrant_configured and retriever and getattr(retriever, "qdrant_client", None):
            has_vectors = _qdrant_has_paper_chunks(
                retriever.qdrant_client, COLLECTION_TEXT, canonical
            )
        elif qdrant_configured and not retriever:
            # Server still booting — do not claim "already indexed"
            has_vectors = False

        if chunk_count > 0 and (not qdrant_configured or has_vectors):
            existing_title = None
            if retriever and retriever.papers_meta.get(canonical):
                existing_title = retriever.papers_meta[canonical].get("title", "Untitled")
            else:
                row = neon_db.get_paper(canonical)
                if row:
                    existing_title = row.get("title", "Untitled")
            if existing_title:
                job = DocumentJob(
                    user_id=current_user.id,
                    arxiv_id=canonical,
                    pdf_url=body.pdf_url,
                    status="done",
                    title=existing_title,
                    chunks_created=chunk_count,
                )
                db.add(job)
                await db.flush()
                log.info("Document already indexed; skipping ingestion: %s", canonical)
                return _job_to_response(job)
    except HTTPException:
        raise
    except Exception as e:
        log.warning("Could not check for existing indexed paper: %s", e)

    # Check for duplicate in-progress jobs (app DB)
    existing = await db.execute(
        select(DocumentJob).where(
            DocumentJob.user_id == current_user.id,
            DocumentJob.arxiv_id == canonical,
            DocumentJob.status.in_(list(JOB_ACTIVE_STATUSES)),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Paper {canonical} is already being processed.",
        )

    job = DocumentJob(
        user_id=current_user.id,
        arxiv_id=canonical,
        pdf_url=body.pdf_url,
    )
    db.add(job)
    await db.flush()

    # Launch background ingestion in a thread (HF Spaces doesn't support
    # true async background tasks well)
    thread = threading.Thread(
        target=_run_ingestion,
        args=(str(job.id), canonical, body.pdf_url),
        daemon=True,
    )
    thread.start()

    log.info(f"Document ingestion queued: {canonical} (job={job.id})")
    return _job_to_response(job)


@router.get("/status/{job_id}", response_model=DocumentJobResponse)
async def get_job_status(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """Poll the status of a document ingestion job."""
    result = await db.execute(
        select(DocumentJob).where(
            DocumentJob.id == job_id,
            DocumentJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_to_response(job)


@router.get("", response_model=list[DocumentJobResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """List all documents submitted by the current user."""
    result = await db.execute(
        select(DocumentJob)
        .where(DocumentJob.user_id == current_user.id)
        .order_by(DocumentJob.created_at.desc())
        .limit(100)
    )
    jobs = result.scalars().all()
    return [_job_to_response(j) for j in jobs]


@router.post("/cancel/{job_id}", response_model=DocumentJobResponse)
async def cancel_document(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """Cancel an in-progress document ingestion job."""
    result = await db.execute(
        select(DocumentJob).where(
            DocumentJob.id == job_id,
            DocumentJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status in JOB_TERMINAL_STATUSES:
        return _job_to_response(job)

    job.status = "cancelled"
    job.error_message = "Cancelled by user."
    await db.flush()
    # Ensure attributes (timestamps) are loaded on the instance before returning
    try:
        await db.refresh(job)
    except Exception:
        # Best-effort refresh; if it fails, still return the current job state
        pass
    return _job_to_response(job)
