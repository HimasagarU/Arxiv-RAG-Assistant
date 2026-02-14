"""
generate_ground_truth.py — Auto-label evaluation queries with ground-truth chunk IDs.

Uses the existing HybridRetriever to retrieve top-k chunks per query,
then saves the top-3 reranked chunk IDs as 'relevant_chunk_ids'.

Usage:
    conda run -n pytorch python eval/generate_ground_truth.py
"""

import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.retrieval import HybridRetriever


def main():
    # Paths
    input_path = os.path.join(PROJECT_ROOT, "tests", "queries.jsonl")
    output_path = os.path.join(PROJECT_ROOT, "eval", "test_queries_labeled.jsonl")

    # Load queries (tests/queries.jsonl is a JSON array, not JSONL)
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            queries = json.loads(content)
        else:
            queries = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"Loaded {len(queries)} queries from {input_path}")

    # Initialize retriever
    print("Initializing HybridRetriever (this loads models + indexes)...")
    retriever = HybridRetriever()
    print("Retriever ready.\n")

    # Generate ground-truth labels
    labeled = []
    for i, q in enumerate(queries):
        query_text = q["query"]
        print(f"[{i+1}/{len(queries)}] {query_text}")

        # Retrieve with a generous top_n to get strong candidates
        result = retriever.retrieve(query_text, top_n=10)
        passages = result["passages"]

        # Use top-3 as ground-truth relevant IDs
        relevant_ids = [p["chunk_id"] for p in passages[:3]]

        labeled.append({
            "query": query_text,
            "relevant_chunk_ids": relevant_ids,
        })

        print(f"         → relevant: {relevant_ids}")

    # Write labeled JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in labeled:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(labeled)} labeled queries → {output_path}")


if __name__ == "__main__":
    main()
