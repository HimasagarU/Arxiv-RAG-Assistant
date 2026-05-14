"""
citation_expander.py — Seed-driven citation expansion via Semantic Scholar API.

Provides backward (references) and forward (citations) expansion
for seed papers, with relevance filtering and layer tagging.

Usage:
    conda run -n pytorch python ingest/citation_expander.py
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Optional
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
CORPUS_TARGET_MAX = int(os.getenv("CORPUS_TARGET_MAX", "5000"))
RESUME_STATE_PATH = Path(os.getenv("CITATION_EXPANDER_STATE", "data/citation_expander_state.json"))

# Rate limiting: free tier = 100 requests / 5 min
S2_REQUEST_DELAY = 4.0  # seconds between requests
S2_MAX_RETRIES = 3
S2_BACKOFF_BASE = 30  # seconds; retries at 30s, 60s, 120s


def _load_resume_state() -> dict:
    if not RESUME_STATE_PATH.exists():
        return {"seeds": {}}
    try:
        return json.loads(RESUME_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"Failed to load resume state, starting fresh: {exc}")
        return {"seeds": {}}


def _save_resume_state(state: dict) -> None:
    RESUME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RESUME_STATE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(RESUME_STATE_PATH)


def _seed_record(state: dict, seed_paper_id: str) -> dict:
    seeds = state.setdefault("seeds", {})
    record = seeds.setdefault(seed_paper_id, {
        "s2_id": "",
        "lookup_status": "pending",
        "references_status": "pending",
        "citations_status": "pending",
        "updated_at": None,
    })
    return record


def _record_complete(record: dict) -> bool:
    return (
        record.get("lookup_status") == "done"
        and record.get("references_status") == "done"
        and record.get("citations_status") == "done"
    )


def _record_has_failure(record: dict) -> bool:
    return any(
        record.get(field) == "failed"
        for field in ("lookup_status", "references_status", "citations_status")
    )


def _touch_record(record: dict) -> None:
    record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


def _s2_request(url: str, params: dict, timeout: int = 30) -> Optional[dict]:
    """Make an S2 API request with disk caching and exponential backoff on 429."""
    import hashlib
    import json
    from pathlib import Path
    
    # Simple disk cache
    cache_dir = Path("data/s2_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{url}?{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    for attempt in range(S2_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=_s2_headers(), timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                cache_file.write_text(json.dumps(data))
                return data
            elif resp.status_code == 429:
                if attempt < S2_MAX_RETRIES:
                    wait = S2_BACKOFF_BASE * (2 ** attempt)
                    log.warning(f"S2 rate limited (429). Retry {attempt+1}/{S2_MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
                    continue
                log.warning(f"S2 rate limit exhausted after {S2_MAX_RETRIES} retries: {url}")
                return None
            elif resp.status_code == 404:
                return None
            else:
                log.warning(f"S2 request failed ({resp.status_code}): {url}")
                return None
        except Exception as e:
            log.warning(f"S2 request error: {e}")
            if attempt < S2_MAX_RETRIES:
                time.sleep(S2_BACKOFF_BASE)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Relevance keywords
# ---------------------------------------------------------------------------

RELEVANCE_KEYWORDS = [
    # Core mechanistic interpretability
    "mechanistic interpretability", "transformer circuits", "circuit analysis",
    "circuit discovery", "computational graph",
    # Attention & internal analysis
    "attention heads", "attention patterns", "induction heads",
    "intermediate activations", "internal representations", "residual stream",
    "qk circuit", "ov circuit",
    # Feature / neuron analysis
    "superposition", "feature decomposition", "feature visualization",
    "polysemanticity", "monosemantic", "neuron analysis",
    "sparse autoencoders", "dictionary learning",
    # Interpretability methods
    "activation patching", "causal tracing", "logit lens", "tuned lens",
    "ablation study", "path patching", "causal mediation",
    "probing classifiers", "probing classifier",
    # Foundational / prerequisite
    "transformer architecture", "self-attention mechanism", "multi-head attention",
    "representation learning", "knowledge neurons",
    "layer normalization", "positional encoding",
    # Safety & alignment
    "interpretability for alignment", "deceptive alignment",
    # Broader terms (relaxed filter — catches more legitimate papers)
    "interpretability", "explainability", "neural network analysis",
    "transformer", "attention mechanism", "language model",
    "feature extraction", "latent representation", "hidden states",
    "model understanding", "model analysis", "model behavior",
    "knowledge representation", "neural circuits",
]

# Compile patterns for fast matching
_KEYWORD_PATTERNS = [re.compile(re.escape(kw), re.IGNORECASE) for kw in RELEVANCE_KEYWORDS]


def is_relevant(title: str, abstract: str = "") -> bool:
    """Check if a paper is relevant based on keyword matching.
    
    Matches against title first (stricter), then abstract (broader).
    A title-only match requires 1 keyword. An abstract-only match
    still requires 1 keyword.
    """
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    combined = f"{title_lower} {abstract_lower}"
    return any(p.search(combined) for p in _KEYWORD_PATTERNS)


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def assign_layer(year: Optional[int], is_seed: bool = False, is_reference: bool = False) -> str:
    """Assign a layer tag based on year and provenance."""
    if year is None:
        return "core"
    if year <= 2019:
        return "prerequisite"
    if year <= 2022:
        return "foundation" if not is_reference else "prerequisite"
    if year <= 2024:
        return "core"
    return "latest"


# ---------------------------------------------------------------------------
# Semantic Scholar API helpers
# ---------------------------------------------------------------------------


def _s2_headers() -> dict:
    headers = {"Accept": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return headers


def s2_lookup_paper(arxiv_id: str) -> Optional[dict]:
    """Look up a paper by arXiv ID on Semantic Scholar (with retry)."""
    url = f"{S2_API_BASE}/paper/ARXIV:{arxiv_id}"
    params = {"fields": "paperId,externalIds,title,abstract,year,authors,citationCount,referenceCount,openAccessPdf"}
    return _s2_request(url, params, timeout=15)


def s2_get_references(s2_paper_id: str, limit: int = 500) -> Optional[list[dict]]:
    """Get references (backward citations) of a paper."""
    url = f"{S2_API_BASE}/paper/{s2_paper_id}/references"
    params = {
        "fields": "paperId,externalIds,title,abstract,year,authors,openAccessPdf",
        "limit": min(limit, 1000),
    }
    result = _s2_request(url, params, timeout=30)
    if result:
        data = result.get("data", [])
        return [item.get("citedPaper", {}) for item in data if item.get("citedPaper")]
    return None


def s2_get_citations(s2_paper_id: str, limit: int = 500) -> Optional[list[dict]]:
    """Get citations (forward) - papers that cite this paper."""
    url = f"{S2_API_BASE}/paper/{s2_paper_id}/citations"
    params = {
        "fields": "paperId,externalIds,title,abstract,year,authors,openAccessPdf",
        "limit": min(limit, 1000),
    }
    result = _s2_request(url, params, timeout=30)
    if result:
        data = result.get("data", [])
        return [item.get("citingPaper", {}) for item in data if item.get("citingPaper")]
    return None


def _extract_arxiv_id(external_ids: dict) -> str:
    """Extract arXiv ID from Semantic Scholar externalIds."""
    arxiv_id = external_ids.get("ArXiv", "")
    if arxiv_id:
        return arxiv_id
    # Fallback: try DOI-based extraction
    return ""


def _s2_paper_to_db_dict(s2_paper: dict, layer: str, source: str = "semantic_scholar") -> Optional[dict]:
    """Convert a Semantic Scholar paper dict to our DB format."""
    external_ids = s2_paper.get("externalIds", {}) or {}
    arxiv_id = _extract_arxiv_id(external_ids)
    s2_id = s2_paper.get("paperId", "")
    title = (s2_paper.get("title") or "").strip()

    if not title:
        return None

    # Keep arXiv IDs only; drop Semantic Scholar hashes entirely.
    if not arxiv_id:
        return None
    paper_id = arxiv_id

    authors_list = s2_paper.get("authors") or []
    authors = ", ".join(a.get("name", "") for a in authors_list if a.get("name"))

    year = s2_paper.get("year")
    published = f"{year}-01-01T00:00:00Z" if year else None

    pdf_url = ""
    if arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    elif s2_paper.get("openAccessPdf"):
        pdf_url = s2_paper["openAccessPdf"].get("url", "")

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": (s2_paper.get("abstract") or "").strip(),
        "authors": authors,
        "categories": "",
        "pdf_url": pdf_url,
        "published": published,
        "updated": None,
        "full_text": "",
        "download_status": "pending",
        "parse_status": "pending",
        "pdf_r2_key": "",
        "quality_score": 0.0,
        "is_seed": False,
        "layer": layer,
        "source": source,
        "semantic_scholar_id": s2_id,
    }


# ---------------------------------------------------------------------------
# Expansion logic
# ---------------------------------------------------------------------------


def expand_seed_references(db, seed_paper_id: str, s2_paper_id: str):
    """Backward expansion: ingest references of a seed paper."""
    log.info(f"Expanding references for seed {seed_paper_id} (S2: {s2_paper_id})")
    references = s2_get_references(s2_paper_id)
    if references is None:
        log.warning(f"  Failed to fetch references for {seed_paper_id}; will retry next run")
        return 0, False
    log.info(f"  Found {len(references)} references")

    accepted = 0
    skipped_relevance = 0
    skipped_no_title = 0
    for ref in references:
        title = (ref.get("title") or "").strip()
        abstract = (ref.get("abstract") or "").strip()

        if not title:
            skipped_no_title += 1
            continue

        if not is_relevant(title, abstract):
            skipped_relevance += 1
            continue

        year = ref.get("year")
        layer = assign_layer(year, is_reference=True)
        paper_dict = _s2_paper_to_db_dict(ref, layer=layer)
        if not paper_dict:
            continue

        # Check corpus cap
        if db.count_papers() >= CORPUS_TARGET_MAX:
            log.info(f"  Corpus cap reached ({CORPUS_TARGET_MAX}), stopping reference expansion")
            break

        db.upsert_paper(paper_dict)
        db.insert_citation_edge(seed_paper_id, paper_dict["paper_id"], "reference")
        accepted += 1

    db.commit()
    log.info(f"  Accepted {accepted} references for {seed_paper_id}")
    if skipped_relevance or skipped_no_title:
        log.info(f"  Rejected: {skipped_relevance} relevance, {skipped_no_title} no-title")
    return accepted, True


# Soft layer quotas: throttle (skip every Nth paper) when over limit,
# but NEVER fully block. This prevents latest papers from overwhelming
# the corpus while still allowing important citations through.
LAYER_SOFT_QUOTA = {
    "latest": 0.55,       # Start throttling at 55%
    "core": 0.35,         # Start throttling at 35%
    "foundation": 1.0,    # Never throttle foundation
    "prerequisite": 1.0,  # Never throttle prerequisite
}


def _check_layer_quota(db, layer: str) -> bool:
    """Soft quota check: returns True to accept, False to skip.
    
    When a layer exceeds its soft quota, only every 3rd paper is accepted
    (throttle mode). Foundation and prerequisite are never throttled.
    """
    max_frac = LAYER_SOFT_QUOTA.get(layer, 1.0)
    if max_frac >= 1.0:
        return True  # Never throttle this layer
    
    total = db.count_papers() or 1
    health = db.get_corpus_health()
    layer_count = health.get("layers", {}).get(layer, 0)
    current_frac = layer_count / total
    
    if current_frac < max_frac:
        return True  # Under quota, accept
    
    # Over quota: accept every 3rd paper (throttle, don't block)
    return (layer_count % 3) == 0


def expand_seed_citations(db, seed_paper_id: str, s2_paper_id: str):
    """Forward expansion: ingest papers citing a seed paper."""
    log.info(f"Expanding citations for seed {seed_paper_id} (S2: {s2_paper_id})")
    citations = s2_get_citations(s2_paper_id)
    if citations is None:
        log.warning(f"  Failed to fetch citations for {seed_paper_id}; will retry next run")
        return 0, False
    log.info(f"  Found {len(citations)} citing papers")

    accepted = 0
    skipped_quota = 0
    skipped_relevance = 0
    skipped_no_title = 0
    for cit in citations:
        title = (cit.get("title") or "").strip()
        abstract = (cit.get("abstract") or "").strip()

        if not title:
            skipped_no_title += 1
            continue

        if not is_relevant(title, abstract):
            skipped_relevance += 1
            continue

        year = cit.get("year")
        layer = assign_layer(year)

        # Soft quota: throttle when over limit, but don't fully block
        if not _check_layer_quota(db, layer):
            skipped_quota += 1
            continue

        paper_dict = _s2_paper_to_db_dict(cit, layer=layer)
        if not paper_dict:
            continue

        if db.count_papers() >= CORPUS_TARGET_MAX:
            log.info(f"  Corpus cap reached ({CORPUS_TARGET_MAX}), stopping citation expansion")
            break

        db.upsert_paper(paper_dict)
        db.insert_citation_edge(seed_paper_id, paper_dict["paper_id"], "citation")
        accepted += 1

    db.commit()
    log.info(f"  Accepted {accepted} citations for {seed_paper_id}")
    if skipped_relevance or skipped_quota or skipped_no_title:
        log.info(f"  Rejected: {skipped_relevance} relevance, {skipped_quota} quota, {skipped_no_title} no-title")
    return accepted, True


def expand_all_seeds(db, resume: bool = True, reset_state: bool = False):
    """Run backward + forward expansion for all seed papers."""
    seeds = db.get_seed_papers()
    if not seeds:
        log.warning("No seed papers found. Ingest seeds first.")
        return

    state = {"seeds": {}}
    if resume and not reset_state:
        state = _load_resume_state()
    elif reset_state and RESUME_STATE_PATH.exists():
        RESUME_STATE_PATH.unlink()

    work_items = []
    for seed in seeds:
        paper_id = seed["paper_id"]
        record = _seed_record(state, paper_id)
        if _record_complete(record):
            continue
        work_items.append((0 if _record_has_failure(record) else 1, paper_id, seed, record))

    work_items.sort(key=lambda item: (item[0], item[1]))

    log.info(f"Expanding citations for {len(work_items)} pending seed papers...")
    total_refs = 0
    total_cits = 0

    for _, paper_id, seed, record in work_items:
        paper_id = seed["paper_id"]
        s2_id = seed.get("semantic_scholar_id", "")

        if record.get("s2_id"):
            s2_id = record["s2_id"]

        # Look up S2 ID if not stored
        if not s2_id and record.get("lookup_status") != "done":
            db.commit()
            s2_data = s2_lookup_paper(paper_id)
            if s2_data:
                s2_id = s2_data.get("paperId", "")
                if s2_id:
                    db.update_paper_field(paper_id, "semantic_scholar_id", s2_id)
                    db.commit()
                    record["s2_id"] = s2_id
                    record["lookup_status"] = "done"
                    _touch_record(record)
                    _save_resume_state(state)
            time.sleep(S2_REQUEST_DELAY)

        if not s2_id:
            log.warning(f"  No S2 ID for {paper_id}, skipping expansion")
            record["lookup_status"] = "failed"
            _touch_record(record)
            _save_resume_state(state)
            continue

        if record.get("lookup_status") != "done":
            record["s2_id"] = s2_id
            record["lookup_status"] = "done"
            _touch_record(record)
            _save_resume_state(state)

        # Backward expansion
        if record.get("references_status") != "done":
            db.commit()
            refs_added, refs_ok = expand_seed_references(db, paper_id, s2_id)
            total_refs += refs_added
            record["references_status"] = "done" if refs_ok else "failed"
            _touch_record(record)
            _save_resume_state(state)
            if not refs_ok:
                continue
        time.sleep(S2_REQUEST_DELAY)

        # Forward expansion
        if record.get("citations_status") != "done":
            db.commit()
            cits_added, cits_ok = expand_seed_citations(db, paper_id, s2_id)
            total_cits += cits_added
            record["citations_status"] = "done" if cits_ok else "failed"
            _touch_record(record)
            _save_resume_state(state)
            if not cits_ok:
                continue
        time.sleep(S2_REQUEST_DELAY)

        if _record_complete(record):
            _touch_record(record)
            _save_resume_state(state)

        # Check corpus cap
        if db.count_papers() >= CORPUS_TARGET_MAX:
            log.info(f"Corpus target reached ({db.count_papers()} papers). Stopping expansion.")
            break

    log.info(f"Expansion complete: +{total_refs} references, +{total_cits} citations")
    log.info(f"Corpus now has {db.count_papers()} papers")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Expand seed papers via Semantic Scholar citations")
    parser.add_argument("--seeds-only", action="store_true", help="Only look up S2 IDs for seeds, don't expand")
    parser.add_argument("--reset-state", action="store_true", help="Clear resume state and start expansion from scratch")
    args = parser.parse_args()

    db = get_db()
    db.run_migrations()

    if args.seeds_only:
        seeds = db.get_seed_papers()
        for s in seeds:
            if not s.get("semantic_scholar_id"):
                db.commit()
                s2 = s2_lookup_paper(s["paper_id"])
                if s2:
                    db.update_paper_field(s["paper_id"], "semantic_scholar_id", s2.get("paperId", ""))
                    db.commit()
                    log.info(f"  {s['paper_id']} -> S2: {s2.get('paperId', 'NOT FOUND')}")
                time.sleep(S2_REQUEST_DELAY)
        return

    expand_all_seeds(db, resume=True, reset_state=args.reset_state)

    # Print corpus health
    health = db.get_corpus_health()
    log.info(f"Corpus health: {health}")
    db.close()


if __name__ == "__main__":
    main()
