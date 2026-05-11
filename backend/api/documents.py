"""
documents.py — Document addition and ingestion status endpoints.

Endpoints:
    POST /documents/add              — Submit a paper for async ingestion
    GET  /documents/status/{job_id}  — Poll ingestion status
    GET  /documents                  — List user's added documents

The background ingestion task reuses the existing ingest pipeline
(ingest_arxiv.py) and stores PDFs in Cloudflare R2.
"""

import logging
import os
import time
import threading
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from db.app_database import get_app_db, async_session_factory
from db.app_models import DocumentJob, User

load_dotenv()

log = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


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


# ---------------------------------------------------------------------------
# Background ingestion task
# ---------------------------------------------------------------------------

def _run_ingestion(job_id: str, arxiv_id: str, pdf_url: Optional[str] = None):
    """
    Background thread: downloads, chunks, embeds a paper.
    Updates the DocumentJob status at each stage.

    This runs in a separate thread with its own sync DB session
    because FastAPI BackgroundTasks don't support async well
    on HuggingFace Spaces.
    """
    backend_root = str(Path(__file__).resolve().parent.parent)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    import psycopg
    from psycopg.rows import dict_row

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
        # Stage 1: Downloading
        _update_job("downloading")
        log.info(f"[Ingest] Downloading paper {arxiv_id}...")

        import requests
        import feedparser

        # Fetch metadata from ArXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = requests.get(api_url, timeout=30)
        feed = feedparser.parse(resp.text)

        if not feed.entries:
            _update_job("failed", error="Paper not found on ArXiv.")
            return

        entry = feed.entries[0]
        paper_title = entry.get("title", "").replace("\n", " ").strip()
        abstract = entry.get("summary", "").strip()
        authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
        categories = ", ".join(t.get("term", "") for t in entry.get("tags", []))

        # Download PDF
        actual_pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_resp = requests.get(actual_pdf_url, timeout=120)
        if pdf_resp.status_code != 200:
            _update_job("failed", error=f"PDF download failed (HTTP {pdf_resp.status_code}).")
            return

        pdf_bytes = pdf_resp.content
        _update_job("downloading", title=paper_title)

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

        # Extract text from PDF
        import fitz  # PyMuPDF
        import io

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()

        if not full_text.strip():
            _update_job("failed", error="Could not extract text from PDF.")
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
            "layer": "core",
        }
        chunks = chunk_paper(paper_dict, tokenizer)

        if not chunks:
            _update_job("failed", error="No chunks produced from text.")
            return

        log.info(f"[Ingest] Produced {len(chunks)} chunks for {arxiv_id}")

        # Stage 3: Embedding + storing
        _update_job("embedding")
        log.info(f"[Ingest] Embedding {len(chunks)} chunks for {arxiv_id}...")

        # Store paper in Neon DB
        from db.database import get_db
        db = get_db()
        db.upsert_paper({
            "paper_id": arxiv_id,
            "title": paper_title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": actual_pdf_url,
            "full_text": full_text[:200000],
            "download_status": "downloaded",
            "parse_status": "parsed",
            "is_seed": False,
            "layer": "core",
            "source": "user_upload",
        })

        # Store chunks in Neon DB
        db.insert_chunks_batch(chunks)
        db.commit()

        # Embed and upsert to Qdrant
        try:
            from api.app import _state
            if _state["retriever"] is not None:
                retriever = _state["retriever"]
                from qdrant_client.models import PointStruct
                from uuid import uuid5, NAMESPACE_URL

                texts = [c["chunk_text"] for c in chunks]
                embeddings = retriever.embed_model.encode(
                    texts, batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                points = []
                for i, chunk in enumerate(chunks):
                    point_id = str(uuid5(NAMESPACE_URL, chunk["chunk_id"]))
                    points.append(PointStruct(
                        id=point_id,
                        vector=embeddings[i].tolist(),
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "paper_id": chunk["paper_id"],
                            "title": chunk.get("title", ""),
                            "authors": chunk.get("authors", ""),
                            "categories": chunk.get("categories", ""),
                            "chunk_text": chunk["chunk_text"],
                            "chunk_type": chunk.get("chunk_type", "text"),
                            "modality": "text",
                            "section_hint": chunk.get("section_hint", "other"),
                            "layer": chunk.get("layer", "core"),
                            "token_count": chunk.get("token_count", 0),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "total_chunks": chunk.get("total_chunks", 1),
                            "chunk_source": chunk.get("chunk_source", "full_text"),
                        },
                    ))

                # Batch upsert to Qdrant
                from api.retrieval import COLLECTION_TEXT
                batch_size = 100
                for j in range(0, len(points), batch_size):
                    retriever.qdrant_client.upsert(
                        collection_name=COLLECTION_TEXT,
                        points=points[j:j + batch_size],
                    )

                log.info(f"[Ingest] Upserted {len(points)} vectors to Qdrant for {arxiv_id}")
                
                # Update in-memory metadata and BM25 index
                retriever.add_paper(paper_dict, chunks)
                log.info(f"[Ingest] Updated in-memory HybridRetriever (BM25 + metadata) for {arxiv_id}")
        except Exception as e:
            log.warning(f"Qdrant/BM25 update failed (non-fatal): {e}")

        # Stage 4: Done
        _update_job("done", title=paper_title, chunks=len(chunks))
        log.info(f"[Ingest] Paper {arxiv_id} ingestion complete ({len(chunks)} chunks).")

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
    # Check if paper already exists in the corpus
    try:
        existing_title = None
        from api.app import _state
        if _state.get("retriever") and _state["retriever"].papers_meta.get(body.arxiv_id):
            existing_title = _state["retriever"].papers_meta[body.arxiv_id].get('title', 'Untitled')
        else:
            from db.database import get_db
            neon_db = get_db()
            existing_paper = neon_db.get_paper(body.arxiv_id)
            neon_db.close()
            if existing_paper:
                existing_title = existing_paper.get('title', 'Untitled')
                
        if existing_title:
            # Paper already in corpus, bypass ingestion and return instantly
            job = DocumentJob(
                user_id=current_user.id,
                arxiv_id=body.arxiv_id,
                pdf_url=body.pdf_url,
                status="done",
                title=existing_title,
                chunks_created=0,
            )
            db.add(job)
            await db.flush()
            log.info(f"Document already in corpus, bypassing ingestion: {body.arxiv_id}")
            return _job_to_response(job)
    except Exception as e:
        log.warning(f"Could not check for existing paper: {e}")

    # Check for duplicate in-progress jobs (app DB)
    existing = await db.execute(
        select(DocumentJob).where(
            DocumentJob.user_id == current_user.id,
            DocumentJob.arxiv_id == body.arxiv_id,
            DocumentJob.status.in_(["queued", "downloading", "chunking", "embedding"]),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Paper {body.arxiv_id} is already being processed.",
        )

    job = DocumentJob(
        user_id=current_user.id,
        arxiv_id=body.arxiv_id,
        pdf_url=body.pdf_url,
    )
    db.add(job)
    await db.flush()

    # Launch background ingestion in a thread (HF Spaces doesn't support
    # true async background tasks well)
    thread = threading.Thread(
        target=_run_ingestion,
        args=(str(job.id), body.arxiv_id, body.pdf_url),
        daemon=True,
    )
    thread.start()

    log.info(f"Document ingestion queued: {body.arxiv_id} (job={job.id})")
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
