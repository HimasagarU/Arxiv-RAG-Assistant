"""
evaluate.py — Retrieval evaluation metrics: Recall@K, MRR, nDCG@K, Precision@K.

Usage:
    conda run -n pytorch python rerank/evaluate.py --queries tests/queries.jsonl
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import psutil

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant docs found in top-K retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-K that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in set(relevant_ids))
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / rank of first relevant result (0 if none found)."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    relevant_set = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    ideal_rels = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    queries_path: str,
    retrieval_fn=None,
    k_values=None,
    *,
    intent_buckets: bool = True,
    doc_level: bool = True,
):
    """
    Run evaluation on a set of queries.

    Args:
        queries_path: Path to JSONL file with 'query' and 'relevant_chunk_ids' fields.
        retrieval_fn: Function that takes a query string and returns list of chunk IDs.
                      If None, performs a dry-run evaluation with mock data.
        k_values: List of K values to compute metrics for.

    Returns:
        Dict of aggregated metrics.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    # Load queries
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    log.info(f"Loaded {len(queries)} evaluation queries")

    from api.retrieval import classify_query_intent

    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"precision@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["mrr"] = []
    results["dense_ms"] = []
    results["lex_ms"] = []
    results["rerank_ms"] = []
    results["doc_recall@10"] = []
    per_query = []

    def _paper_from_chunk(cid: str) -> str:
        if not cid or "_" not in cid:
            return ""
        return cid.rsplit("_", 2)[0] if "_text_" in cid else cid.split("_")[0]

    for q in queries:
        query_text = q["query"]
        relevant = q.get("relevant_chunk_ids", [])
        rel_papers = list({ _paper_from_chunk(c) for c in relevant if _paper_from_chunk(c) })

        if retrieval_fn:
            start = time.time()
            retrieved, trace = retrieval_fn(query_text)
            latency_ms = (time.time() - start) * 1000
        else:
            retrieved = []
            trace = {}
            latency_ms = 0

        qr = {
            "query": query_text, 
            "latency_ms": latency_ms,
            "dense_ms": trace.get("dense_ms", 0),
            "lex_ms": trace.get("lex_ms", 0),
            "rerank_ms": trace.get("rerank_ms", 0),
        }

        results["dense_ms"].append(trace.get("dense_ms", 0))
        results["lex_ms"].append(trace.get("lex_ms", 0))
        results["rerank_ms"].append(trace.get("rerank_ms", 0))

        for k in k_values:
            r_at_k = recall_at_k(retrieved, relevant, k)
            p_at_k = precision_at_k(retrieved, relevant, k)
            n_at_k = ndcg_at_k(retrieved, relevant, k)
            results[f"recall@{k}"].append(r_at_k)
            results[f"precision@{k}"].append(p_at_k)
            results[f"ndcg@{k}"].append(n_at_k)
            qr[f"recall@{k}"] = r_at_k
            qr[f"precision@{k}"] = p_at_k
            qr[f"ndcg@{k}"] = n_at_k

        mrr = reciprocal_rank(retrieved, relevant)
        results["mrr"].append(mrr)

        if doc_level and rel_papers:
            ret_papers = [_paper_from_chunk(c) for c in retrieved[:10]]
            doc_hits = len(set(ret_papers) & set(rel_papers))
            results["doc_recall@10"].append(doc_hits / max(len(rel_papers), 1))
        qr["mrr"] = mrr
        qr["intent"] = classify_query_intent(query_text)
        per_query.append(qr)

    # Aggregate
    aggregated = {}
    for metric, values in results.items():
        if values:
            aggregated[metric] = sum(values) / len(values)
        else:
            aggregated[metric] = 0.0

    if intent_buckets:
        from collections import defaultdict

        by_intent = defaultdict(list)
        for row in per_query:
            by_intent[row.get("intent", "unknown")].append(row)
        aggregated["intent_buckets"] = {
            intent: {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / len(rows) for k in k_values}
            for intent, rows in by_intent.items()
            if rows
        }

    return aggregated, per_query


def print_results(aggregated: dict, per_query: list):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    buckets = aggregated.pop("intent_buckets", None)
    for metric, value in sorted(aggregated.items()):
        if isinstance(value, (int, float)):
            print(f"  {metric:20s}: {value:.4f}")
    if buckets:
        print("\n  Per-intent recall@10:")
        for intent, row in sorted(buckets.items()):
            v = row.get("recall@10", 0.0)
            print(f"    {intent:16s}: {v:.4f}")

    if per_query:
        latencies = [q["latency_ms"] for q in per_query if q["latency_ms"] > 0]
        if latencies:
            print(f"\n  {'avg_latency_ms':20s}: {sum(latencies)/len(latencies):.1f}")
            print(f"  {'p95_latency_ms':20s}: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}")
        
        dense = [q["dense_ms"] for q in per_query if "dense_ms" in q]
        if dense:
            print(f"  {'avg_dense_ms':20s}: {sum(dense)/len(dense):.1f}")
        
        lex = [q["lex_ms"] for q in per_query if "lex_ms" in q]
        if lex:
            print(f"  {'avg_lex_ms':20s}: {sum(lex)/len(lex):.1f}")
            
        rerank = [q["rerank_ms"] for q in per_query if "rerank_ms" in q]
        if rerank:
            print(f"  {'avg_rerank_ms':20s}: {sum(rerank)/len(rerank):.1f}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics")
    parser.add_argument("--queries", type=str, default="tests/queries.jsonl",
                        help="Path to queries JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for per-query results")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "dense_only", "bm25_only", "no_parent", "no_rerank", "no_mmr"],
        help="Ablation preset (sets RETRIEVAL_SKIP_* env vars for this process).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.queries):
        log.error(f"Queries file not found: {args.queries}")
        log.info("Create a queries.jsonl file with 'query' and 'relevant_chunk_ids' fields.")
        return

    if args.mode == "dense_only":
        os.environ["RETRIEVAL_SKIP_LEXICAL"] = "true"
        os.environ["RETRIEVAL_SKIP_PARENT_CHILD"] = "true"
    elif args.mode == "bm25_only":
        os.environ["RETRIEVAL_SKIP_DENSE"] = "true"
        os.environ["RETRIEVAL_SKIP_PARENT_CHILD"] = "true"
    elif args.mode == "no_parent":
        os.environ["RETRIEVAL_SKIP_PARENT_CHILD"] = "true"
    elif args.mode == "no_rerank":
        os.environ["RETRIEVAL_SKIP_RERANK"] = "true"
    elif args.mode == "no_mmr":
        os.environ["RETRIEVAL_SKIP_MMR"] = "true"

    from api.retrieval import HybridRetriever
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    log.info(f"Memory before retriever init: {mem_before:.1f} MB")
    
    log.info("Loading Hybrid Retriever...")
    retriever = HybridRetriever()
    
    mem_after = process.memory_info().rss / (1024 * 1024)
    log.info(f"Memory after retriever init: {mem_after:.1f} MB")
    log.info(f"Artifact Memory Footprint: {mem_after - mem_before:.1f} MB")
    
    def retrieval_fn(q):
        res = retriever.retrieve(q, top_n=20)
        return [p["chunk_id"] for p in res["passages"]], res["trace"]
        
    aggregated, per_query = evaluate_retrieval(args.queries, retrieval_fn=retrieval_fn)
    print_results(aggregated, per_query)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"aggregated": aggregated, "per_query": per_query}, f, indent=2)
        log.info(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
