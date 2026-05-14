"""CLI orchestrator for offline ingestion and indexing workflows."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.fetch_data import fetch_and_extract
from db.database import get_db
from ingest.chunking import run_chunking
from ingest.ingest_arxiv import ingest_keyword_papers, ingest_seed_papers, enrich_full_text
from index import build_bm25, build_qdrant
from storage.local_pdf_store import LocalPDFStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DERIVED_ARTIFACTS = [
    DATA_DIR / "chunks.jsonl",
    DATA_DIR / "chunks_meta.jsonl",
    DATA_DIR / "chunks_text.jsonl",
    DATA_DIR / "papers_meta.json",
    DATA_DIR / "bm25_v1.pkl",
]
QDRANT_COLLECTIONS = ["arxiv_text", "arxiv_docs"]


def _confirm_or_exit(message: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not click.confirm(message, default=False):
        raise click.Abort()


def _delete_artifacts(keep_metadata: bool = True) -> None:
    for path in DERIVED_ARTIFACTS:
        if keep_metadata and path.name == "papers_meta.json":
            continue
        if path.exists():
            path.unlink()
            log.info("Deleted %s", path)


def _reset_qdrant_collections() -> None:
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        log.warning("QDRANT_URL not set; skipping Qdrant reset.")
        return
    client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
    for name in QDRANT_COLLECTIONS:
        try:
            client.delete_collection(name)
            log.info("Deleted Qdrant collection: %s", name)
        except Exception as exc:
            log.info("Qdrant collection %s not deleted (%s)", name, exc)
    client.close()


@click.group()
def cli() -> None:
    """ArXiv RAG offline orchestration CLI."""


@cli.command()
@click.option("--mode", type=click.Choice(["seed", "keyword", "enrich", "all"], case_sensitive=False), default="all")
@click.option("--max-pages", type=int, default=20, show_default=True)
@click.option("--query", "query_override", type=str, default=None, help="Override the keyword query.")
@click.option("--pdf-timeout", type=int, default=60, show_default=True)
@click.option("--enrich-limit", type=int, default=0, show_default=True)
@click.option("--retry-failed", is_flag=True)
def ingest(
    mode: str,
    max_pages: int,
    query_override: str | None,
    pdf_timeout: int,
    enrich_limit: int,
    retry_failed: bool,
) -> None:
    """Ingest papers from ArXiv (seed/keyword/enrich)."""
    db = get_db()
    db.run_migrations()
    local_store = LocalPDFStore()

    if mode in {"seed", "all"}:
        ingest_seed_papers(db)

    if mode in {"keyword", "all"}:
        ingest_keyword_papers(db, max_pages=max_pages, query_override=query_override)

    if mode in {"enrich", "all"}:
        enrich_full_text(
            db,
            local_store,
            limit=enrich_limit,
            pdf_timeout=pdf_timeout,
            retry_failed=retry_failed,
        )

    db.close()


@cli.command()
@click.option("--source", type=click.Choice(["auto", "full_text", "abstract"], case_sensitive=False), default="auto")
@click.option("--strategy", type=click.Choice(["section-sentence", "token"], case_sensitive=False), default="section-sentence")
@click.option("--reset", is_flag=True)
@click.option("--limit", type=int, default=0)
@click.option("--offline", is_flag=True, default=True, show_default=True,
              help="Write chunks to JSONL only (skip Neon insert). Chunks are stored in Qdrant via 'index'.")
def chunk(source: str, strategy: str, reset: bool, limit: int, offline: bool) -> None:
    """Chunk papers using the production chunker."""
    run_chunking(source_mode=source, strategy=strategy, reset=reset, limit=limit, offline=offline)


@cli.command()
@click.option("--target", type=click.Choice(["qdrant", "bm25", "both"], case_sensitive=False), default="both")
@click.option("--resume", is_flag=True, help="Resume Qdrant indexing without deleting existing data.")
def index(target: str, resume: bool) -> None:
    """Build indexes from local chunk artifacts."""
    if target in {"qdrant", "both"}:
        build_qdrant.main(resume=resume)
    if target in {"bm25", "both"}:
        build_bm25.main()


@cli.command()
@click.option("--keep-pdfs/--drop-pdfs", default=True, show_default=True)
@click.option("--rebuild", is_flag=True, help="Rebuild chunks and indexes after reset.")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts.")
def reset(keep_pdfs: bool, rebuild: bool, yes: bool) -> None:
    """Reset derived artifacts while preserving PDFs by default."""
    _confirm_or_exit(
        "This will delete derived artifacts and reset Qdrant collections. Continue?",
        assume_yes=yes,
    )

    if not keep_pdfs:
        for pdf_dir in [DATA_DIR / "pdfs", DATA_DIR / "arxiv_pdfs"]:
            if pdf_dir.exists():
                shutil.rmtree(pdf_dir)
                log.info("Deleted PDFs in %s", pdf_dir)

    _delete_artifacts(keep_metadata=True)
    _reset_qdrant_collections()

    db = get_db()
    db.run_migrations()
    db.delete_all_chunks()
    db.commit()
    db.close()

    if rebuild:
        run_chunking(source_mode="auto", strategy="section-sentence", reset=True, offline=True)
        build_bm25.main()
        build_qdrant.main()


@cli.command()
def fetch_artifacts() -> None:
    """Fetch artifact bundle from R2 (if configured)."""
    fetch_and_extract()


@cli.command()
def health() -> None:
    """Print corpus health summary from Neon."""
    db = get_db()
    db.run_migrations()
    health = db.get_corpus_health()
    db.close()
    click.echo(json.dumps(health, indent=2))


@cli.command()
@click.option("--limit", type=int, default=0, help="Limit chunk checks (0=all).")
def audit(limit: int) -> None:
    """Audit local artifacts vs Qdrant vs Neon metadata."""
    chunks_path = DATA_DIR / "chunks.jsonl"
    papers_meta_path = DATA_DIR / "papers_meta.json"

    if not chunks_path.exists() or not papers_meta_path.exists():
        raise click.ClickException("Missing chunks.jsonl or papers_meta.json in data directory.")

    with open(papers_meta_path, "r", encoding="utf-8") as f:
        papers_meta = json.load(f)

    chunk_ids = []
    chunk_paper_ids = set()
    with open(chunks_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_ids.append(row.get("chunk_id"))
            chunk_paper_ids.add(row.get("paper_id"))
            if limit and idx >= limit:
                break

    missing_papers = sorted(pid for pid in chunk_paper_ids if pid not in papers_meta)

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise click.ClickException("QDRANT_URL not set; cannot audit Qdrant.")

    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
    qdrant_chunk_ids = set()
    qdrant_paper_ids = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name="arxiv_text",
            with_payload=True,
            limit=500,
            offset=next_offset,
        )
        for point in points:
            payload = point.payload or {}
            cid = payload.get("chunk_id")
            pid = payload.get("paper_id")
            if cid:
                qdrant_chunk_ids.add(cid)
            if pid:
                qdrant_paper_ids.add(pid)
        if next_offset is None:
            break

    missing_chunks = [cid for cid in chunk_ids if cid and cid not in qdrant_chunk_ids]

    db = get_db()
    db.run_migrations()
    with db.conn.cursor() as cur:
        cur.execute("SELECT paper_id FROM papers")
        neon_paper_ids = {row["paper_id"] for row in cur.fetchall()}
    db.close()

    missing_neon = sorted(pid for pid in qdrant_paper_ids if pid not in neon_paper_ids)

    report = {
        "chunks_checked": len(chunk_ids),
        "papers_in_chunks": len(chunk_paper_ids),
        "missing_papers_meta": missing_papers[:50],
        "missing_qdrant_chunks": missing_chunks[:50],
        "missing_neon_papers": missing_neon[:50],
    }
    click.echo(json.dumps(report, indent=2))


@cli.command("sync-metadata")
@click.option("--skip-existing", is_flag=True, help="Skip papers already in Neon.")
def sync_metadata(skip_existing: bool) -> None:
    """Sync paper metadata from local artifacts into Neon PostgreSQL."""
    from db.metadata_sync import sync_papers_from_artifacts

    db = get_db()
    db.run_migrations()
    result = sync_papers_from_artifacts(db, DATA_DIR, skip_existing=skip_existing)
    click.echo(json.dumps(result, indent=2))
    click.echo(f"Total papers in Neon: {db.count_papers()}")
    db.close()


@cli.command("neon-report")
def neon_report() -> None:
    """Print Neon metadata and coverage report."""
    db = get_db()
    db.run_migrations()
    report = db.neon_metadata_report(DATA_DIR)
    db.close()
    click.echo(json.dumps(report, indent=2, default=str))


@cli.command("neon")
@click.argument("action", type=click.Choice(["truncate-chunks"], case_sensitive=False))
def neon(action: str) -> None:
    """Neon database maintenance commands."""
    db = get_db()
    db.run_migrations()
    if action == "truncate-chunks":
        db.truncate_chunks_table()
        click.echo(f"Chunks truncated. Papers retained: {db.count_papers()}")
    db.close()


@cli.command("pipeline")
@click.argument(
    "stage",
    type=click.Choice(["parse", "chunk", "bm25", "qdrant", "sync", "full"], case_sensitive=False),
)
@click.option("--keep-papers-meta", is_flag=True)
@click.option("--chunk-strategy", default="section-sentence")
@click.option("--chunk-source", default="auto")
@click.option("--enrich-limit", type=int, default=0)
@click.option("--pdf-timeout", type=int, default=60)
@click.option("--neon-chunks", is_flag=True, help="Insert chunks into Neon during chunk stage.")
@click.option("--reset-qdrant", is_flag=True)
@click.option("--qdrant-resume", is_flag=True)
@click.option("--yes", is_flag=True, help="Confirm destructive stage=full")
def pipeline_cmd(
    stage: str,
    keep_papers_meta: bool,
    chunk_strategy: str,
    chunk_source: str,
    enrich_limit: int,
    pdf_timeout: int,
    neon_chunks: bool,
    reset_qdrant: bool,
    qdrant_resume: bool,
    yes: bool,
):
    """Run one offline rebuild stage (see ingest/pipeline.py)."""
    from ingest.pipeline import run_stage, _write_pipeline_manifest

    if stage == "full" and not yes:
        raise click.ClickException("stage=full requires --yes (truncates chunks and rebuilds indexes).")

    run_stage(
        stage,
        data_dir=DATA_DIR,
        keep_papers_meta=keep_papers_meta,
        chunk_strategy=chunk_strategy,
        chunk_source=chunk_source,
        enrich_limit=enrich_limit,
        pdf_timeout=pdf_timeout,
        neon_chunk_rows=neon_chunks,
        reset_qdrant=reset_qdrant,
        qdrant_resume=qdrant_resume,
    )
    _write_pipeline_manifest(DATA_DIR, stage)


if __name__ == "__main__":
    cli()
