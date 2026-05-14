"""Sync paper rows in PostgreSQL from local `papers_meta.json` (no full-text backfill)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_local_pdf_path(paper_id: str, pdf_roots: list[Path]) -> str:
    for root in pdf_roots:
        if not root.exists():
            continue
        direct = root / f"{paper_id}.pdf"
        if direct.exists():
            return str(direct)
        nested = next(root.rglob(f"{paper_id}.pdf"), None)
        if nested:
            return str(nested)
    return ""


def sync_papers_from_artifacts(
    db,
    data_dir: Path,
    *,
    skip_existing: bool = False,
) -> dict[str, int]:
    """
    Upsert `papers` from `papers_meta.json` plus optional chunk counts from `chunks_meta.jsonl`.

    Returns counts: inserted, updated, skipped.
    """
    papers_meta_path = data_dir / "papers_meta.json"
    chunks_meta_path = data_dir / "chunks_meta.jsonl"

    if not papers_meta_path.exists():
        raise FileNotFoundError(f"Missing artifact file: {papers_meta_path}")

    papers_meta = _load_json(papers_meta_path)
    if not isinstance(papers_meta, dict):
        raise ValueError("papers_meta.json must be a JSON object keyed by paper_id")

    chunk_counts: dict[str, int] = {}
    if chunks_meta_path.exists():
        with open(chunks_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                pid = entry.get("paper_id", "")
                if pid:
                    chunk_counts[pid] = chunk_counts.get(pid, 0) + 1

    pdf_roots = [data_dir / "pdfs", data_dir / "arxiv_pdfs"]

    try:
        from ingest.ingest_arxiv import SEED_PAPERS

        seed_ids = {seed["paper_id"] for seed in SEED_PAPERS}
    except Exception:
        seed_ids = set()

    inserted = 0
    updated = 0
    skipped = 0

    for paper_id, meta in papers_meta.items():
        if skip_existing and db.paper_exists(paper_id):
            skipped += 1
            continue

        existed = db.paper_exists(paper_id)
        local_pdf_path = discover_local_pdf_path(paper_id, pdf_roots)
        paper_record = {
            "paper_id": paper_id,
            "title": meta.get("title", ""),
            "abstract": meta.get("abstract", "") or "",
            "authors": meta.get("authors", "") or "",
            "categories": meta.get("categories", "") or "",
            "published": meta.get("published"),
            "full_text": "",
            "pdf_url": "",
            "download_status": "artifact_only",
            "parse_status": "artifact_only",
            "local_pdf_path": local_pdf_path,
            "quality_score": 1.0 if chunk_counts.get(paper_id, 0) > 0 else 0.5,
            "is_seed": paper_id in seed_ids,
            "layer": meta.get("layer", "core") or "core",
            "source": "artifact_import",
            "semantic_scholar_id": "",
        }
        db.upsert_paper(paper_record)
        if existed:
            updated += 1
        else:
            inserted += 1

    db.commit()
    log.info(
        "Artifact sync complete: inserted=%s updated=%s skipped=%s total_papers_in_db=%s",
        inserted,
        updated,
        skipped,
        db.count_papers(),
    )
    return {"inserted": inserted, "updated": updated, "skipped": skipped}
