"""Offline corpus rebuild: retain PDFs, re-parse, re-chunk, re-embed, rebuild BM25, sync Neon paper metadata."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


def discover_paper_ids_with_local_pdfs(data_dir: Path) -> list[str]:
    ids: set[str] = set()
    for root in (data_dir / "pdfs", data_dir / "arxiv_pdfs"):
        if not root.exists():
            continue
        for p in root.rglob("*.pdf"):
            try:
                if p.stat().st_size > 1024:
                    ids.add(p.stem)
            except OSError:
                continue
    return sorted(ids)


def clear_derived_files(data_dir: Path, *, keep_papers_meta: bool = False) -> None:
    targets = [
        data_dir / "chunks.jsonl",
        data_dir / "chunks_meta.jsonl",
        data_dir / "chunks_text.jsonl",
        data_dir / "bm25_v1.pkl",
    ]
    if not keep_papers_meta:
        targets.append(data_dir / "papers_meta.json")
    for path in targets:
        if path.exists():
            path.unlink()
            log.info("Removed %s", path)


def reset_qdrant_collections(collection_names: list[str] | None = None) -> None:
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        log.warning("QDRANT_URL not set; skipping Qdrant reset.")
        return
    names = collection_names or ["arxiv_text", "arxiv_docs"]
    client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
    for name in names:
        try:
            client.delete_collection(name)
            log.info("Deleted Qdrant collection: %s", name)
        except Exception as exc:
            log.info("Qdrant collection %s not deleted (%s)", name, exc)
    client.close()


def run_rebuild_pipeline(
    *,
    data_dir: Path | None = None,
    keep_papers_meta: bool = False,
    chunk_strategy: str = "section-sentence",
    chunk_source: str = "auto",
    enrich_limit: int = 0,
    pdf_timeout: int = 60,
    neon_chunk_rows: bool = True,
    reset_qdrant: bool = True,
) -> None:
    from db.database import get_db
    from db import metadata_sync
    from ingest.chunking import run_chunking
    from ingest.ingest_arxiv import enrich_full_text
    from index import build_bm25, build_qdrant
    from storage.local_pdf_store import LocalPDFStore

    root = data_dir or Path(os.getenv("DATA_DIR", "data"))
    root.mkdir(parents=True, exist_ok=True)

    db = get_db()
    db.run_migrations()

    clear_derived_files(root, keep_papers_meta=keep_papers_meta)
    db.truncate_chunks_table()

    ids = discover_paper_ids_with_local_pdfs(root)
    cleared = db.clear_full_text_for_paper_ids(ids)
    log.info("Prepared %s papers with local PDFs for re-parse (full_text cleared: %s rows).", len(ids), cleared)

    local_store = LocalPDFStore()
    enrich_full_text(
        db,
        local_store,
        limit=enrich_limit,
        pdf_timeout=pdf_timeout,
        retry_failed=False,
    )
    db.close()

    run_chunking(
        source_mode=chunk_source,
        strategy=chunk_strategy,
        reset=True,
        limit=0,
        offline=not neon_chunk_rows,
    )

    build_bm25.main()
    if reset_qdrant:
        reset_qdrant_collections()
    build_qdrant.main()

    db2 = get_db()
    db2.run_migrations()
    metadata_sync.sync_papers_from_artifacts(db2, root, skip_existing=False)
    db2.close()
    from utils.artifact_schema import write_artifact_manifest

    write_artifact_manifest(root, extra={"pipeline": "full_offline_rebuild"})
    log.info("Pipeline rebuild finished.")


def run_stage(
    stage: str,
    *,
    data_dir: Path | None = None,
    keep_papers_meta: bool = False,
    chunk_strategy: str = "section-sentence",
    chunk_source: str = "auto",
    enrich_limit: int = 0,
    pdf_timeout: int = 60,
    neon_chunk_rows: bool = True,
    reset_qdrant: bool = False,
    qdrant_resume: bool = False,
) -> None:
    """Run a single pipeline stage: parse | chunk | bm25 | qdrant | sync | full."""
    root = data_dir or Path(os.getenv("DATA_DIR", "data"))
    root.mkdir(parents=True, exist_ok=True)
    stage = stage.lower().strip()

    if stage == "parse":
        from db.database import get_db
        from ingest.ingest_arxiv import enrich_full_text
        from storage.local_pdf_store import LocalPDFStore

        db = get_db()
        db.run_migrations()
        ids = discover_paper_ids_with_local_pdfs(root)
        cleared = db.clear_full_text_for_paper_ids(ids)
        log.info("Re-parse: cleared full_text for %s papers (%s rows).", len(ids), cleared)
        local_store = LocalPDFStore()
        enrich_full_text(
            db,
            local_store,
            limit=enrich_limit,
            pdf_timeout=pdf_timeout,
            retry_failed=False,
        )
        db.close()
        log.info("Stage parse complete.")
        return

    if stage == "chunk":
        from ingest.chunking import run_chunking

        run_chunking(
            source_mode=chunk_source,
            strategy=chunk_strategy,
            reset=True,
            limit=0,
            offline=not neon_chunk_rows,
        )
        log.info("Stage chunk complete.")
        return

    if stage == "bm25":
        from index import build_bm25

        build_bm25.main()
        log.info("Stage bm25 complete.")
        return

    if stage == "qdrant":
        from index import build_qdrant

        if reset_qdrant:
            reset_qdrant_collections()
        build_qdrant.main(resume=qdrant_resume)
        log.info("Stage qdrant complete.")
        return

    if stage == "sync":
        from db.database import get_db
        from db import metadata_sync

        db = get_db()
        db.run_migrations()
        metadata_sync.sync_papers_from_artifacts(db, root, skip_existing=False)
        db.close()
        log.info("Stage sync complete.")
        return

    if stage == "full":
        run_rebuild_pipeline(
            data_dir=root,
            keep_papers_meta=keep_papers_meta,
            chunk_strategy=chunk_strategy,
            chunk_source=chunk_source,
            enrich_limit=enrich_limit,
            pdf_timeout=pdf_timeout,
            neon_chunk_rows=neon_chunk_rows,
            reset_qdrant=True,
        )
        return

    raise ValueError(f"Unknown stage: {stage}")


def _write_pipeline_manifest(root: Path, stage: str) -> None:
    from utils.artifact_schema import write_artifact_manifest

    write_artifact_manifest(root, extra={"last_pipeline_stage": stage})


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Offline corpus pipeline stages")
    parser.add_argument(
        "stage",
        choices=["parse", "chunk", "bm25", "qdrant", "sync", "full"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--keep-papers-meta", action="store_true")
    parser.add_argument("--chunk-strategy", default="section-sentence")
    parser.add_argument("--chunk-source", default="auto")
    parser.add_argument("--enrich-limit", type=int, default=0)
    parser.add_argument("--pdf-timeout", type=int, default=60)
    parser.add_argument("--neon-chunks", action="store_true", help="Insert chunks into Neon (default offline JSONL)")
    parser.add_argument("--reset-qdrant", action="store_true", help="Delete Qdrant collections before qdrant stage")
    parser.add_argument("--qdrant-resume", action="store_true", help="Resume Qdrant indexing without deleting collections")
    parser.add_argument("--i-understand-destructive-full", action="store_true", help="Required for stage=full")
    args = parser.parse_args()

    root = Path(args.data_dir) if args.data_dir else None
    if args.stage == "full" and not args.i_understand_destructive_full:
        log.error("Refusing stage=full without --i-understand-destructive-full (truncates chunks and rebuilds indexes).")
        raise SystemExit(2)

    run_stage(
        args.stage,
        data_dir=root,
        keep_papers_meta=args.keep_papers_meta,
        chunk_strategy=args.chunk_strategy,
        chunk_source=args.chunk_source,
        enrich_limit=args.enrich_limit,
        pdf_timeout=args.pdf_timeout,
        neon_chunk_rows=args.neon_chunks,
        reset_qdrant=args.reset_qdrant,
        qdrant_resume=args.qdrant_resume,
    )
    if root:
        _write_pipeline_manifest(root, args.stage)
    else:
        _write_pipeline_manifest(Path(os.getenv("DATA_DIR", "data")), args.stage)


if __name__ == "__main__":
    main_cli()
