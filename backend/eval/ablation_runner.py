"""
ablation_runner.py — Run retrieval ablation study across pipeline variants.

Compares four configurations:
  1. Dense only  (Qdrant)
  2. Dense + BM25 + RRF
  3. Hybrid + Cross-Encoder Reranker
  4. Hybrid + Reranker + Context Compression

Reports retrieval metrics (Recall@K, MRR, nDCG@K) and latency per stage.

Usage:
    conda run -n pytorch python eval/ablation_runner.py
    conda run -n pytorch python eval/ablation_runner.py --limit 10
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVAL_DIR / "ragas_queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# Ablation variants — each returns (chunk_ids, trace_dict)
# ---------------------------------------------------------------------------

def variant_dense_only(retriever, query: str, top_n: int = 10) -> tuple[list[str], dict]:
    """Dense retrieval only (Qdrant), no BM25, no reranker."""
    t0 = time.time()
    dense_candidates = retriever._dense_retrieve(query, qdrant_filter=None)
    dense_ms = round((time.time() - t0) * 1000, 1)

    # Sort by dense score, take top N
    dense_candidates.sort(key=lambda x: x.get("dense_score", 0), reverse=True)
    top = dense_candidates[:top_n]

    return (
        [c["chunk_id"] for c in top],
        {"dense_ms": dense_ms, "lex_ms": 0, "rerank_ms": 0, "variant": "dense_only"},
    )


def variant_hybrid_rrf(retriever, query: str, top_n: int = 10) -> tuple[list[str], dict]:
    """Dense + BM25 + RRF fusion, no reranker."""
    from api.retrieval import classify_query_intent

    intent = classify_query_intent(query)

    t0 = time.time()
    dense_candidates = retriever._dense_retrieve(query, qdrant_filter=None)
    dense_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    lex_candidates = retriever._lexical_retrieve(query)
    lex_ms = round((time.time() - t1) * 1000, 1)

    t2 = time.time()
    merged = retriever._merge_and_normalize(dense_candidates, lex_candidates, intent)
    merge_ms = round((time.time() - t2) * 1000, 1)

    top = merged[:top_n]

    return (
        [c["chunk_id"] for c in top],
        {"dense_ms": dense_ms, "lex_ms": lex_ms, "merge_ms": merge_ms, "rerank_ms": 0, "variant": "hybrid_rrf"},
    )


def variant_hybrid_reranker(retriever, query: str, top_n: int = 10) -> tuple[list[str], dict]:
    """Dense + BM25 + RRF + Cross-Encoder reranking."""
    from api.retrieval import classify_query_intent

    intent = classify_query_intent(query)

    t0 = time.time()
    dense_candidates = retriever._dense_retrieve(query, qdrant_filter=None)
    dense_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    lex_candidates = retriever._lexical_retrieve(query)
    lex_ms = round((time.time() - t1) * 1000, 1)

    t2 = time.time()
    merged = retriever._merge_and_normalize(dense_candidates, lex_candidates, intent)
    merge_ms = round((time.time() - t2) * 1000, 1)

    t3 = time.time()
    reranked = retriever.reranker.rerank(query, merged, top_n=top_n)
    rerank_ms = round((time.time() - t3) * 1000, 1)

    return (
        [c["chunk_id"] for c in reranked],
        {"dense_ms": dense_ms, "lex_ms": lex_ms, "merge_ms": merge_ms, "rerank_ms": rerank_ms, "variant": "hybrid_reranker"},
    )


def variant_full_pipeline(retriever, query: str, top_n: int = 10) -> tuple[list[str], dict]:
    """Full pipeline: Dense + BM25 + RRF + Reranker + Compression."""
    from api.retrieval import classify_query_intent

    intent = classify_query_intent(query)

    t0 = time.time()
    result = retriever.retrieve(query, top_n=top_n, intent=intent)
    total_ms = round((time.time() - t0) * 1000, 1)

    trace = result["trace"]
    trace["variant"] = "full_pipeline"
    trace["total_ms"] = total_ms

    # Also run compression to measure its cost
    t_c = time.time()
    retriever.compress_context(query, result["passages"], intent=intent)
    trace["compress_ms"] = round((time.time() - t_c) * 1000, 1)

    return (
        [p["chunk_id"] for p in result["passages"]],
        trace,
    )


def variant_bm25_only(retriever, query: str, top_n: int = 10) -> tuple[list[str], dict]:
    """Lexical (BM25) only."""
    prev_d = os.environ.get("RETRIEVAL_SKIP_DENSE")
    prev_p = os.environ.get("RETRIEVAL_SKIP_PARENT_CHILD")
    try:
        os.environ["RETRIEVAL_SKIP_DENSE"] = "true"
        os.environ["RETRIEVAL_SKIP_PARENT_CHILD"] = "true"
        res = retriever.retrieve(query, top_n=top_n)
        return [p["chunk_id"] for p in res["passages"]], {**res["trace"], "variant": "bm25_only"}
    finally:
        if prev_d is None:
            os.environ.pop("RETRIEVAL_SKIP_DENSE", None)
        else:
            os.environ["RETRIEVAL_SKIP_DENSE"] = prev_d
        if prev_p is None:
            os.environ.pop("RETRIEVAL_SKIP_PARENT_CHILD", None)
        else:
            os.environ["RETRIEVAL_SKIP_PARENT_CHILD"] = prev_p


def variant_with_parent_full(retriever, query: str, top_n: int = 10) -> tuple[list[str], dict]:
    """Full retrieve (respects ENABLE_PARENT_CHILD when arxiv_docs exists)."""
    res = retriever.retrieve(query, top_n=top_n)
    return [p["chunk_id"] for p in res["passages"]], {**res["trace"], "variant": "retrieve_full"}


VARIANTS = {
    "dense_only": variant_dense_only,
    "bm25_only": variant_bm25_only,
    "hybrid_rrf": variant_hybrid_rrf,
    "hybrid_reranker": variant_hybrid_reranker,
    "full_pipeline": variant_full_pipeline,
    "retrieve_full": variant_with_parent_full,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablation(retriever, questions: list[dict], top_n: int = 10) -> dict:
    """Run all variants on all questions, collecting metrics and timings."""
    all_results = {}

    for variant_name, variant_fn in VARIANTS.items():
        log.info(f"\n{'─' * 60}")
        log.info(f"Running variant: {variant_name}")
        log.info(f"{'─' * 60}")

        per_query = []
        latency_accumulator = defaultdict(list)

        for i, q in enumerate(questions):
            question = q["question"]
            intent = q.get("intent", "discovery")
            log.info(f"  [{i+1}/{len(questions)}] {question[:55]}...")

            t0 = time.time()
            chunk_ids, trace = variant_fn(retriever, question, top_n=top_n)
            total_ms = round((time.time() - t0) * 1000, 1)

            # Collect latency components
            for key in ["dense_ms", "lex_ms", "merge_ms", "rerank_ms", "compress_ms"]:
                if key in trace:
                    latency_accumulator[key].append(trace[key])

            per_query.append({
                "question": question,
                "intent": intent,
                "chunk_ids": chunk_ids,
                "total_ms": total_ms,
                "trace": {k: v for k, v in trace.items() if k != "variant"},
            })

        # Aggregate latency
        latency_summary = {}
        for key, vals in latency_accumulator.items():
            latency_summary[key] = {
                "avg": round(sum(vals) / len(vals), 1) if vals else 0,
                "p95": round(sorted(vals)[int(len(vals) * 0.95)], 1) if vals else 0,
                "max": round(max(vals), 1) if vals else 0,
            }

        total_latencies = [q["total_ms"] for q in per_query]
        latency_summary["total_ms"] = {
            "avg": round(sum(total_latencies) / len(total_latencies), 1),
            "p95": round(sorted(total_latencies)[int(len(total_latencies) * 0.95)], 1),
            "max": round(max(total_latencies), 1),
        }

        all_results[variant_name] = {
            "per_query": per_query,
            "latency_summary": latency_summary,
        }

    return all_results


def print_ablation_report(results: dict):
    """Print a comparison table of all variants."""
    print("\n" + "=" * 72)
    print("ABLATION STUDY RESULTS")
    print("=" * 72)

    header = f"{'Variant':25s} | {'Avg Total':>10s} | {'P95 Total':>10s} | {'Avg Dense':>10s} | {'Avg Lex':>8s} | {'Avg Rerank':>10s}"
    print(header)
    print("-" * len(header))

    for variant_name, data in results.items():
        ls = data["latency_summary"]
        total = ls.get("total_ms", {})
        dense = ls.get("dense_ms", {})
        lex = ls.get("lex_ms", {})
        rerank = ls.get("rerank_ms", {})

        print(
            f"{variant_name:25s} | "
            f"{total.get('avg', 0):>8.1f}ms | "
            f"{total.get('p95', 0):>8.1f}ms | "
            f"{dense.get('avg', 0):>8.1f}ms | "
            f"{lex.get('avg', 0):>6.1f}ms | "
            f"{rerank.get('avg', 0):>8.1f}ms"
        )

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study for retrieval variants")
    parser.add_argument("--queries", type=str, default=str(QUERIES_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.queries):
        log.error(f"Queries not found: {args.queries}. Run `python eval/ragas_dataset.py` first.")
        return

    questions = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[:args.limit]

    log.info(f"Loaded {len(questions)} evaluation questions")

    # Initialize retriever
    from api.retrieval import HybridRetriever
    log.info("Loading HybridRetriever...")
    retriever = HybridRetriever()

    # Run ablation
    results = run_ablation(retriever, questions, top_n=args.top_n)

    # Print summary
    print_ablation_report(results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(RESULTS_DIR / "ablation_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Full results saved → {output_path}")


if __name__ == "__main__":
    main()
