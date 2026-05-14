"""
timeline_balancer.py — Enforce balanced temporal coverage in the corpus.

Checks era distribution and runs targeted arXiv keyword queries
to fill gaps in underrepresented time periods.

Usage:
    conda run -n pytorch python ingest/timeline_balancer.py [--fill-gaps]
"""

import argparse
import logging
import os
import re
import sys
import time

import feedparser
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db
from ingest.citation_expander import is_relevant, assign_layer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
CORPUS_TARGET_MAX = int(os.getenv("CORPUS_TARGET_MAX", "5000"))
CORPUS_TARGET_MIN = int(os.getenv("CORPUS_TARGET_MIN", "500"))

# Target era percentages (30% early / 40% middle / 30% recent)
ERA_TARGETS = {
    "early_2017_2020": {"min_pct": 20, "max_pct": 35, "year_start": 2017, "year_end": 2020},
    "middle_2021_2023": {"min_pct": 30, "max_pct": 45, "year_start": 2021, "year_end": 2023},
    "recent_2024_plus": {"min_pct": 20, "max_pct": 35, "year_start": 2024, "year_end": 2026},
}

# Gap-filling keywords grouped by era relevance
GAP_FILL_KEYWORDS = {
    "early_2017_2020": [
        # Foundational transformer work
        "transformer architecture",
        "self-attention mechanism",
        "multi-head attention",
        "attention mechanism neural networks",
        # Representation / interpretability foundations
        "representation learning",
        "neural network interpretability",
        "feature visualization",
        "knowledge distillation",
        "attention heads analysis",
        "layer normalization",
        "probing classifiers",
        "knowledge neurons",
        "BERT analysis",
        "GPT-2 analysis",
        "ablation study neural networks",
    ],
    "middle_2021_2023": [
        # Core mech interp emergence
        "mechanistic interpretability",
        "transformer circuits",
        "induction heads",
        "superposition neural networks",
        "activation patching",
        "causal tracing",
        "circuit discovery",
        "logit lens",
        "path patching",
        "indirect object identification",
        "residual stream analysis",
        "knowledge editing",
        "causal mediation analysis",
        "automated circuit discovery",
        "representation engineering",
        "linear probing transformers",
    ],
    "recent_2024_plus": [
        "sparse autoencoders interpretability",
        "monosemantic features",
        "mechanistic interpretability",
        "circuit analysis transformers",
        "dictionary learning neural networks",
        "interpretability alignment",
        "polysemanticity",
        "activation steering",
        "feature circuits",
        "scaling monosemanticity",
        "SAE features",
    ],
}

# Maximum retries for arXiv API
ARXIV_MAX_RETRIES = 3
ARXIV_BACKOFF_BASE = 30  # seconds


def check_balance(db) -> dict:
    """Check current era distribution and identify gaps."""
    eras = db.get_era_distribution()
    total = sum(eras.values()) or 1
    total_papers = db.count_papers()

    report = {
        "total_papers": total_papers,
        "eras": {},
        "gaps": [],
        "balanced": True,
    }

    for era_name, targets in ERA_TARGETS.items():
        count = eras.get(era_name, 0)
        pct = round(count / total * 100, 1) if total > 0 else 0
        is_under = pct < targets["min_pct"]
        is_over = pct > targets["max_pct"]

        report["eras"][era_name] = {
            "count": count,
            "pct": pct,
            "target_min": targets["min_pct"],
            "target_max": targets["max_pct"],
            "status": "under" if is_under else ("over" if is_over else "ok"),
        }

        if is_under:
            # Estimate how many papers needed to reach min_pct
            needed = max(1, int(total * targets["min_pct"] / 100) - count)
            report["gaps"].append({
                "era": era_name,
                "needed": needed,
                "year_start": targets["year_start"],
                "year_end": targets["year_end"],
            })
            report["balanced"] = False

    return report


def _build_arxiv_queries_for_era(era_name: str) -> list[str]:
    """Build targeted arXiv queries for a specific era to maximize retrieval."""
    queries = {
        "early_2017_2020": ['all:"transformer"', 'all:"neural network"', 'all:"representation learning"', 'all:"attention mechanism"'],
        "middle_2021_2023": ['all:"large language model"', 'all:"transformer"', 'all:"interpretability"', 'all:"in-context learning"'],
        "recent_2024_plus": ['all:"large language model"', 'all:"mechanistic interpretability"', 'all:"sparse autoencoder"', 'all:"circuit discovery"'],
    }
    return queries.get(era_name, ['all:"transformer"', 'all:"neural network"'])


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _extract_arxiv_id(entry_id: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    if match:
        return match.group(1)
    return entry_id.split("/abs/")[-1].split("v")[0]


def fill_gap(db, gap: dict, max_per_gap: int = 500, force_fill: bool = False) -> int:
    """Fill a specific era gap by querying arXiv with year-scoped keywords."""
    era = gap["era"]
    needed = min(gap["needed"], max_per_gap)
    year_start = gap["year_start"]
    year_end = gap["year_end"]

    log.info(f"Filling gap for {era}: need ~{needed} papers ({year_start}-{year_end})")
    if force_fill:
        log.info("  FORCE-FILL mode active: bypassing strict relevance filters")

    queries = _build_arxiv_queries_for_era(era)
    fetched = 0

    for base_query in queries:
        if fetched >= needed:
            break

        # FIX 1: Add year constraint to query
        query = f"({base_query}) AND submittedDate:[{year_start}01010000 TO {year_end}12312359]"
        start = 0

        # FIX 3: Reduce start loop explosion (per query)
        while fetched < needed and start < needed * 2:
            params = {
                "search_query": query,
                "start": start,
                "max_results": 50,
                "sortBy": "submittedDate",
                # FIX 2: Sort ascending to get older papers first
                "sortOrder": "ascending",
            }

            try:
                resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    for retry in range(ARXIV_MAX_RETRIES):
                        wait = ARXIV_BACKOFF_BASE * (2 ** retry)
                        log.warning(f"arXiv rate limited (429). Retry {retry+1}/{ARXIV_MAX_RETRIES} in {wait}s...")
                        time.sleep(wait)
                        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
                        if resp.status_code != 429:
                            break
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
            except Exception as e:
                log.warning(f"arXiv query failed: {e}")
                break

            if not feed.entries:
                break
                
            # FIX 4: Add debug logging
            log.info(f"  Fetched {len(feed.entries)} entries from arXiv for query: {base_query}")

            accepted_count = 0
            for entry in feed.entries:
                paper_id = _extract_arxiv_id(entry.id)
                title = _sanitize_text(_clean_text(entry.get("title", "")))
                abstract = _sanitize_text(_clean_text(entry.get("summary", "")))
                published = entry.get("published", "")

                # Check year range
                try:
                    pub_year = int(published[:4]) if published else 0
                except (ValueError, IndexError):
                    pub_year = 0

                if pub_year < year_start or pub_year > year_end:
                    continue

                # Check relevance (bypass if force_fill)
                if not force_fill and not is_relevant(title, abstract):
                    continue

                # Skip if already in corpus
                if db.paper_exists(paper_id):
                    continue

                # Check corpus cap
                if db.count_papers() >= CORPUS_TARGET_MAX:
                    log.info("Corpus cap reached during gap-fill")
                    return fetched

                # Build paper dict
                authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
                categories = ", ".join(t.get("term", "") for t in entry.get("tags", []))

                pdf_url = ""
                for link in entry.get("links", []):
                    if link.get("type") == "application/pdf":
                        pdf_url = link.get("href", "")
                        break
                if not pdf_url:
                    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

                layer = assign_layer(pub_year)

                paper = {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": _sanitize_text(authors),
                    "categories": _sanitize_text(categories),
                    "pdf_url": _sanitize_text(pdf_url),
                    "published": published,
                    "updated": entry.get("updated", ""),
                    "full_text": "",
                    "download_status": "pending",
                    "parse_status": "pending",
                    "is_seed": False,
                    "layer": layer,
                    "source": "arxiv",
                }

                db.upsert_paper(paper)
                fetched += 1
                accepted_count += 1

                if fetched % 10 == 0:
                    db.commit()
                    log.info(f"  Gap-fill {era}: {fetched}/{needed} papers")

            log.info(f"  Accepted {accepted_count} entries from this batch")
            start += 50
            time.sleep(2.0)  # Gentle delay for arXiv

    if fetched > 0:
        db.commit()

    log.info(f"  Gap-fill complete for {era}: added {fetched} papers")
    return fetched


def _s2_gap_fill(db, gap: dict, max_papers: int = 200) -> int:
    """Gap-fill via Semantic Scholar keyword search with pagination."""
    from ingest.citation_expander import _s2_request, _s2_paper_to_db_dict

    era = gap["era"]
    keywords = GAP_FILL_KEYWORDS.get(era, [])[:4]  # Limit to 4 to save S2 API quota
    year_start = gap["year_start"]
    year_end = gap["year_end"]

    log.info(f"  S2 gap-fill for {era} ({year_start}-{year_end}), {len(keywords)} keywords...")

    fetched = 0
    for kw in keywords:
        if fetched >= max_papers:
            break

        # Paginate through S2 results
        for offset in [0, 100]:
            if fetched >= max_papers:
                break

            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": kw,
                "fields": "paperId,externalIds,title,abstract,year,authors",
                "limit": 100,
                "offset": offset,
                "year": f"{year_start}-{year_end}",
            }

            result = _s2_request(url, params, timeout=30)
            if not result:
                time.sleep(5)
                continue

            data = result.get("data", [])
            if not data:
                break

            for paper_data in data:
                title = (paper_data.get("title") or "").strip()
                abstract = (paper_data.get("abstract") or "").strip()

                if not title or not is_relevant(title, abstract):
                    continue

                year = paper_data.get("year")
                if year and (year < year_start or year > year_end):
                    continue

                layer = assign_layer(year)
                paper_dict = _s2_paper_to_db_dict(paper_data, layer=layer, source="s2_gap_fill")
                if not paper_dict:
                    continue

                if db.paper_exists(paper_dict["paper_id"]):
                    continue

                if db.count_papers() >= CORPUS_TARGET_MAX:
                    db.commit()
                    return fetched

                db.upsert_paper(paper_dict)
                fetched += 1

            db.commit()
            time.sleep(5)  # S2 rate limit between pages

        if fetched % 20 == 0 and fetched > 0:
            log.info(f"    S2 {era}: {fetched}/{max_papers} papers so far")

    log.info(f"  S2 gap-fill added {fetched} papers for {era}")
    return fetched


def fill_all_gaps(db, force_fill: bool = False) -> int:
    """Identify and fill all era gaps. Uses arXiv + S2 together."""
    report = check_balance(db)
    if report["balanced"]:
        log.info("Corpus is already balanced across eras.")
        return 0

    total_filled = 0
    for gap in report["gaps"]:
        # Try arXiv first (arXiv is faster, less rate limited, more stable)
        filled = fill_gap(db, gap, force_fill=force_fill)
        total_filled += filled

        # FIX 5: Skip S2 if arXiv returns 0 (prevents hammering S2 when query/API is busted)
        if filled == 0:
            log.warning(f"  Skipping S2 for {gap['era']} due to empty arXiv result")
            continue

        # Always supplement with S2 (not just fallback)
        remaining = gap["needed"] - filled
        if remaining > 0:
            s2_filled = _s2_gap_fill(db, gap, max_papers=min(remaining, 200))
            total_filled += s2_filled

    # Print updated health
    health = db.get_corpus_health()
    log.info(f"After gap-filling: {health}")
    return total_filled


def print_balance_report(db):
    """Pretty-print the timeline balance report."""
    report = check_balance(db)
    health = db.get_corpus_health()

    print("\n" + "=" * 60)
    print("CORPUS TIMELINE BALANCE REPORT")
    print("=" * 60)
    print(f"  Total papers: {report['total_papers']}")
    print(f"  Seed papers:  {health['seed_papers']}")
    print(f"  With text:    {health['with_full_text']}")
    print()

    for era, info in report["eras"].items():
        status = "✓" if info["status"] == "ok" else ("▼ UNDER" if info["status"] == "under" else "▲ OVER")
        print(f"  {era:25s}: {info['count']:5d} papers ({info['pct']:5.1f}%)  "
              f"[target: {info['target_min']}-{info['target_max']}%]  {status}")

    print()
    if report["gaps"]:
        print("  GAPS TO FILL:")
        for gap in report["gaps"]:
            print(f"    {gap['era']}: need ~{gap['needed']} more papers")
    else:
        print("  ✓ All eras within target range")

    print()
    print("  LAYER DISTRIBUTION:")
    for layer, count in health.get("layers", {}).items():
        print(f"    {layer:15s}: {count}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check and fix timeline balance")
    parser.add_argument("--fill-gaps", action="store_true", help="Automatically fill era gaps")
    parser.add_argument("--report-only", action="store_true", help="Only print the balance report")
    parser.add_argument("--force-fill", action="store_true", help="Bypass strict filters to force gap filling")
    args = parser.parse_args()

    db = get_db()
    db.run_migrations()

    print_balance_report(db)

    if args.fill_gaps and not args.report_only:
        fill_all_gaps(db, force_fill=args.force_fill)
        print_balance_report(db)

    db.close()


if __name__ == "__main__":
    main()
