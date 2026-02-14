"""
run_metrics.py — Compute all metrics from Metrics_Guide.md for the ArXiv RAG Assistant.

Covers:
    A. Retrieval quality  — Recall@k, Precision@k, MRR, nDCG@k
    B. Latency & QPS      — p50, p95, avg (ms), queries per second
    C. Index & storage    — index sizes, embedding dim, vector count
    D. Answer quality     — embedding similarity of generated answer to source text
    E. Cost / tokens      — avg tokens per query estimate

Usage:
    conda run -n pytorch python eval/run_metrics.py
    conda run -n pytorch python eval/run_metrics.py --api-url http://localhost:8000
"""

import argparse
import json
import logging
import math
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline metric functions (identical to rerank/evaluate.py, kept here to
# avoid module-level import of rerank.evaluate which manipulates sys.path
# and can trigger huggingface_hub import conflicts on Windows).
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
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)
    ideal_rels = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# A. Retrieval quality metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(retriever, queries_path: str, k_values=None):
    """
    Compute Recall@k, Precision@k, MRR, nDCG@k using labeled queries.
    Returns aggregated dict + per-query list + list of latencies.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    # Load labeled queries (JSONL format)
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    log.info(f"Loaded {len(queries)} labeled evaluation queries")

    # Accumulate per-metric
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"precision@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["mrr"] = []
    per_query = []
    latencies_ms = []

    for i, q in enumerate(queries):
        query_text = q["query"]
        relevant = q.get("relevant_chunk_ids", [])

        if not relevant:
            log.warning(f"Query {i} has no relevant_chunk_ids, skipping: {query_text}")
            continue

        # Retrieve with large top_n for metric computation at various k
        max_k = max(k_values)
        t0 = time.perf_counter()
        retrieved = retriever.retrieve_ids(query_text, top_n=max_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        qr = {"query": query_text, "latency_ms": round(latency_ms, 1)}

        for k in k_values:
            r = recall_at_k(retrieved, relevant, k)
            p = precision_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            results[f"recall@{k}"].append(r)
            results[f"precision@{k}"].append(p)
            results[f"ndcg@{k}"].append(n)
            qr[f"recall@{k}"] = r
            qr[f"precision@{k}"] = p
            qr[f"ndcg@{k}"] = n

        mrr = reciprocal_rank(retrieved, relevant)
        results["mrr"].append(mrr)
        qr["mrr"] = mrr
        per_query.append(qr)

        log.info(f"  [{i+1}/{len(queries)}] {query_text[:50]}... recall@5={qr.get('recall@5', 0):.2f} mrr={mrr:.2f}")

    # Aggregate
    aggregated = {}
    for metric, values in results.items():
        aggregated[metric] = round(sum(values) / len(values), 4) if values else 0.0

    return aggregated, per_query, latencies_ms


# ---------------------------------------------------------------------------
# B. Latency & QPS
# ---------------------------------------------------------------------------

def compute_latency_stats(latencies_ms: list) -> dict:
    """Compute p50, p95, avg latency and QPS from a list of latencies in ms."""
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    total_seconds = arr.sum() / 1000.0
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "avg_ms": round(float(np.mean(arr)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
        "num_queries": len(latencies_ms),
        "qps": round(len(latencies_ms) / total_seconds, 2) if total_seconds > 0 else 0,
    }


# ---------------------------------------------------------------------------
# C. Latency via live API (optional, if the API server is running)
# ---------------------------------------------------------------------------

def compute_api_latency(api_url: str, queries: list, top_k: int = 5) -> dict:
    """
    Measure end-to-end latency by calling the live API.
    Returns latency stats dict.
    """
    try:
        import requests
    except ImportError:
        log.warning("requests not installed, skipping API latency test")
        return {}

    times = []
    for q in queries:
        query_text = q if isinstance(q, str) else q.get("query", "")
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{api_url}/query",
                json={"query": query_text, "top_k": top_k},
                timeout=30,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                times.append(elapsed_ms)
            else:
                log.warning(f"API returned {resp.status_code} for '{query_text[:40]}...'")
        except Exception as e:
            log.warning(f"API call failed for '{query_text[:40]}...': {e}")

    return compute_latency_stats(times)


# ---------------------------------------------------------------------------
# D. Index & storage stats
# ---------------------------------------------------------------------------

def compute_index_stats() -> dict:
    """Compute index file sizes, embedding dimension, vector count."""
    stats = {}

    # Chroma DB directory size
    chroma_dir = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    if os.path.isdir(chroma_dir):
        total_bytes = 0
        for dirpath, _, filenames in os.walk(chroma_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_bytes += os.path.getsize(fp)
        stats["chroma_db_mb"] = round(total_bytes / (1024 * 1024), 2)

    # BM25 index
    bm25_path = os.path.join(PROJECT_ROOT, "data", "bm25_index.pkl")
    if os.path.exists(bm25_path):
        stats["bm25_index_mb"] = round(os.path.getsize(bm25_path) / (1024 * 1024), 2)
        # Load to get doc count
        try:
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            stats["bm25_doc_count"] = len(bm25_data.get("chunk_ids", []))
        except Exception:
            pass

    # Embeddings file
    embeddings_path = os.path.join(PROJECT_ROOT, "data", "embeddings.npy")
    if os.path.exists(embeddings_path):
        stats["embeddings_file_mb"] = round(os.path.getsize(embeddings_path) / (1024 * 1024), 2)
        try:
            emb = np.load(embeddings_path)
            stats["num_vectors"] = emb.shape[0]
            stats["embedding_dim"] = emb.shape[1]
        except Exception:
            pass

    # Chunks file
    chunks_path = os.path.join(PROJECT_ROOT, "data", "chunks.jsonl")
    if os.path.exists(chunks_path):
        stats["chunks_file_mb"] = round(os.path.getsize(chunks_path) / (1024 * 1024), 2)
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                stats["num_chunks"] = sum(1 for _ in f)
        except Exception:
            pass

    # SQLite DB
    db_path = os.path.join(PROJECT_ROOT, "data", "arxiv_papers.db")
    if os.path.exists(db_path):
        stats["sqlite_db_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
        try:
            conn = sqlite3.connect(db_path)
            stats["num_papers"] = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            conn.close()
        except Exception:
            pass

    # Chroma collection count (via chroma client)
    if os.path.isdir(chroma_dir):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            coll = client.get_collection("arxiv_chunks")
            stats["chroma_vector_count"] = coll.count()
        except Exception:
            pass

    # Embedding model info
    stats["embedding_model"] = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    stats["reranker_model"] = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    return stats


# ---------------------------------------------------------------------------
# E. Cost / tokens estimate
# ---------------------------------------------------------------------------

def compute_token_cost(log_path: str = None) -> dict:
    """
    Estimate token usage and cost from query logs.
    Reads logs/queries.jsonl for generation timings.
    """
    if log_path is None:
        log_path = os.path.join(PROJECT_ROOT, "logs", "queries.jsonl")

    if not os.path.exists(log_path):
        return {"note": "No query logs found at " + log_path}

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return {"note": "No valid log entries"}

    gen_times = []
    retrieval_times = []
    for e in entries:
        trace = e.get("trace", {})
        gen_ms = trace.get("generation_ms", 0)
        total_ms = trace.get("total_ms", 0)
        if gen_ms > 0:
            gen_times.append(gen_ms)
        if total_ms > 0:
            retrieval_times.append(total_ms)

    stats = {
        "total_logged_queries": len(entries),
        "queries_with_generation": len(gen_times),
    }

    if gen_times:
        stats["avg_generation_ms"] = round(sum(gen_times) / len(gen_times), 1)
    if retrieval_times:
        stats["avg_retrieval_ms"] = round(sum(retrieval_times) / len(retrieval_times), 1)

    # Estimate tokens: Groq llama-3.3-70b-versatile averages ~300 output tokens per response
    # Input prompt: ~500 tokens (system + sources + query)
    # Groq free tier: no per-token cost, but we estimate for documentation
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        stats["llm_model"] = "llama-3.3-70b-versatile (Groq)"
        stats["est_input_tokens_per_query"] = 500
        stats["est_output_tokens_per_query"] = 300
        stats["est_total_tokens_per_query"] = 800
        stats["cost_per_query_usd"] = "Free (Groq free tier)"
    else:
        stats["llm_model"] = "None (local fallback, no LLM)"
        stats["est_tokens_per_query"] = 0
        stats["cost_per_query_usd"] = "$0.00"

    return stats


# ---------------------------------------------------------------------------
# F. Answer quality (embedding similarity)
# ---------------------------------------------------------------------------

def compute_answer_quality(retriever, queries_path: str, max_queries: int = 10) -> dict:
    """
    Compute answer quality by measuring embedding similarity between
    the generated answer and the source passages.
    Uses the retriever's embed model for consistency.
    """
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    queries = queries[:max_queries]

    from sentence_transformers import util

    similarities = []
    for q in queries:
        query_text = q["query"]
        result = retriever.retrieve(query_text, top_n=5)
        passages = result["passages"]

        if not passages:
            continue

        # Combine passage texts as reference
        reference = " ".join([p["chunk_text"] for p in passages[:3]])

        # Build simple answer proxy (what the model would receive as context)
        query_emb = retriever.embed_model.encode(query_text, convert_to_numpy=True)
        ref_emb = retriever.embed_model.encode(reference, convert_to_numpy=True)

        sim = float(util.cos_sim(
            query_emb.reshape(1, -1),
            ref_emb.reshape(1, -1)
        )[0][0])
        similarities.append(sim)

    if similarities:
        return {
            "avg_query_source_similarity": round(sum(similarities) / len(similarities), 4),
            "min_query_source_similarity": round(min(similarities), 4),
            "max_query_source_similarity": round(max(similarities), 4),
            "num_queries_evaluated": len(similarities),
        }
    return {"note": "No passages retrieved for similarity computation"}


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_report(report: dict):
    """Print a nicely formatted metrics report."""
    print("\n" + "=" * 70)
    print("   ArXiv RAG Assistant — Full Metrics Report")
    print("=" * 70)

    sections = [
        ("Retrieval Quality", "retrieval_quality"),
        ("Latency & Throughput (Retriever)", "latency"),
        ("Latency & Throughput (Live API)", "api_latency"),
        ("Index & Storage", "index_stats"),
        ("Answer Quality", "answer_quality"),
        ("Cost / Tokens", "cost_tokens"),
    ]

    for title, key in sections:
        data = report.get(key, {})
        if not data:
            continue
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"    {k:35s}: {v:.4f}")
            else:
                print(f"    {k:35s}: {v}")

    print("\n" + "=" * 70)
    print("  Report saved to eval/metrics_report.json")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute all ArXiv RAG metrics")
    parser.add_argument("--queries", default=None,
                        help="Path to labeled queries JSONL (default: eval/test_queries_labeled.jsonl)")
    parser.add_argument("--api-url", default=None,
                        help="Live API URL for end-to-end latency test (e.g. http://localhost:8000)")
    parser.add_argument("--output", default=None,
                        help="Output JSON report path (default: eval/metrics_report.json)")
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="Skip retrieval quality metrics (faster, index-stats only)")
    args = parser.parse_args()

    queries_path = args.queries or os.path.join(PROJECT_ROOT, "eval", "test_queries_labeled.jsonl")
    output_path = args.output or os.path.join(PROJECT_ROOT, "eval", "metrics_report.json")

    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    retriever = None

    if not args.skip_retrieval:
        # Check for labeled queries
        if not os.path.exists(queries_path):
            log.error(f"Labeled queries not found: {queries_path}")
            log.info("Run 'python eval/generate_ground_truth.py' first to generate labels.")
            # Still save partial report with index stats
            log.info("Computing index & storage stats...")
            report["index_stats"] = compute_index_stats()
            log.info("Computing cost / token estimates...")
            report["cost_tokens"] = compute_token_cost()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print_report(report)
            return

        # Load retriever FIRST (before chromadb-based index stats)
        # This ensures sentence_transformers/huggingface_hub loads before
        # chromadb's importlib_metadata usage, avoiding a Windows conflict.
        log.info("Initializing HybridRetriever for evaluation...")
        from api.retrieval import HybridRetriever
        retriever = HybridRetriever()

        # A. Retrieval quality
        log.info("Computing retrieval quality metrics...")
        aggregated, per_query, latencies = compute_retrieval_metrics(
            retriever, queries_path, k_values=[1, 5, 10, 20]
        )
        report["retrieval_quality"] = aggregated

        # B. Latency from retriever calls
        log.info("Computing latency stats...")
        report["latency"] = compute_latency_stats(latencies)

        # F. Answer quality
        log.info("Computing answer quality (embedding similarity)...")
        report["answer_quality"] = compute_answer_quality(retriever, queries_path)

        # Save per-query details
        report["_per_query_details"] = per_query

    # --- Compute index stats (safe now that sentence_transformers is loaded) ---
    log.info("Computing index & storage stats...")
    report["index_stats"] = compute_index_stats()

    # --- Compute cost/token estimate ---
    log.info("Computing cost / token estimates...")
    report["cost_tokens"] = compute_token_cost()

    # C. API latency (optional)
    if args.api_url:
        log.info(f"Testing live API latency at {args.api_url}...")
        if os.path.exists(queries_path):
            with open(queries_path, "r", encoding="utf-8") as f:
                test_queries = [json.loads(l) for l in f if l.strip()]
        else:
            test_queries = [{"query": "What are transformer architectures?"}]
        report["api_latency"] = compute_api_latency(args.api_url, test_queries)

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print_report(report)
    log.info(f"Full report saved → {output_path}")


if __name__ == "__main__":
    main()
