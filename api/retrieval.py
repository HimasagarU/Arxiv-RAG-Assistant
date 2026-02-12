"""
retrieval.py — Hybrid dense + BM25 retrieval pipeline with cross-encoder reranking.

This module implements the full retrieval pipeline:
1. Dense retrieval via Chroma (sentence-transformer embeddings)
2. Lexical retrieval via BM25
3. Score normalization and weighted fusion
4. Cross-encoder reranking
5. Return top-N passages with full trace
"""

import json
import logging
import os
import pickle
import re
import time
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import chromadb
from rank_bm25 import BM25Okapi

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerank.reranker import Reranker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Stopwords for BM25 tokenization (must match build_bm25.py)
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she",
    "they", "me", "us", "him", "her", "them", "my", "our", "your",
    "his", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "not", "no", "nor", "as", "if", "then", "so",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "all", "also", "am", "any", "because", "before", "between", "both",
    "each", "few", "more", "most", "other", "own", "same", "some",
    "such", "only", "into", "over", "under", "up", "down", "out",
])


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 queries (matches build_bm25.py tokenizer)."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS]


class HybridRetriever:
    """
    Hybrid retrieval: dense (Chroma) + lexical (BM25) + cross-encoder reranking.
    """

    def __init__(
        self,
        chroma_dir: str = None,
        bm25_path: str = None,
        embedding_model: str = None,
        reranker_model: str = None,
        k_dense: int = 50,
        k_lex: int = 50,
        merge_top_m: int = 20,
        final_top_n: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        self.k_dense = k_dense
        self.k_lex = k_lex
        self.merge_top_m = merge_top_m
        self.final_top_n = final_top_n
        self.alpha = alpha
        self.beta = beta

        # Resolve paths from env
        chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "data/chroma_db")
        bm25_path = bm25_path or os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")
        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        reranker_model = reranker_model or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Initializing HybridRetriever on device={device}")

        # Load embedding model
        log.info(f"Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model, device=device)

        # Load Chroma
        log.info(f"Loading Chroma collection from: {chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_collection("arxiv_chunks")
        log.info(f"Chroma collection has {self.collection.count()} documents")

        # Load BM25
        log.info(f"Loading BM25 index from: {bm25_path}")
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25: BM25Okapi = bm25_data["bm25"]
        self.bm25_chunk_ids: list[str] = bm25_data["chunk_ids"]

        # Load reranker
        self.reranker = Reranker(model_name=reranker_model, device=device)

        log.info("HybridRetriever ready.")

    def _dense_retrieve(self, query: str) -> list[dict]:
        """Dense retrieval via Chroma."""
        query_embedding = self.embed_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k_dense,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for i in range(len(results["ids"][0])):
            candidates.append({
                "chunk_id": results["ids"][0][i],
                "chunk_text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "dense_score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
                "source": "dense",
            })
        return candidates

    def _bm25_retrieve(self, query: str) -> list[dict]:
        """Lexical retrieval via BM25."""
        tokens = tokenize_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:self.k_lex]

        candidates = []
        for idx in top_indices:
            if scores[idx] > 0:
                candidates.append({
                    "chunk_id": self.bm25_chunk_ids[idx],
                    "bm25_score": float(scores[idx]),
                    "source": "bm25",
                })
        return candidates

    def _merge_and_normalize(
        self, dense_candidates: list[dict], bm25_candidates: list[dict]
    ) -> list[dict]:
        """Merge dense and BM25 candidates, normalize scores, compute fusion score."""
        # Build lookup by chunk_id
        merged = {}

        # Add dense candidates
        for c in dense_candidates:
            merged[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "chunk_text": c.get("chunk_text", ""),
                "metadata": c.get("metadata", {}),
                "dense_score_raw": c["dense_score"],
                "bm25_score_raw": 0.0,
                "sources": ["dense"],
            }

        # Add/merge BM25 candidates
        for c in bm25_candidates:
            cid = c["chunk_id"]
            if cid in merged:
                merged[cid]["bm25_score_raw"] = c["bm25_score"]
                merged[cid]["sources"].append("bm25")
            else:
                # Need to fetch chunk text from Chroma for reranking
                merged[cid] = {
                    "chunk_id": cid,
                    "chunk_text": "",
                    "metadata": {},
                    "dense_score_raw": 0.0,
                    "bm25_score_raw": c["bm25_score"],
                    "sources": ["bm25"],
                }

        candidates = list(merged.values())

        # Fetch missing chunk texts from Chroma
        missing_ids = [c["chunk_id"] for c in candidates if not c["chunk_text"]]
        if missing_ids:
            try:
                fetched = self.collection.get(
                    ids=missing_ids, include=["documents", "metadatas"]
                )
                for i, cid in enumerate(fetched["ids"]):
                    for c in candidates:
                        if c["chunk_id"] == cid:
                            c["chunk_text"] = fetched["documents"][i]
                            c["metadata"] = fetched["metadatas"][i]
                            break
            except Exception as e:
                log.warning(f"Failed to fetch missing chunks: {e}")

        # Normalize scores (min-max per source)
        dense_scores = [c["dense_score_raw"] for c in candidates if c["dense_score_raw"] > 0]
        bm25_scores = [c["bm25_score_raw"] for c in candidates if c["bm25_score_raw"] > 0]

        dense_min = min(dense_scores) if dense_scores else 0
        dense_max = max(dense_scores) if dense_scores else 1
        bm25_min = min(bm25_scores) if bm25_scores else 0
        bm25_max = max(bm25_scores) if bm25_scores else 1

        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        for c in candidates:
            c["dense_score_norm"] = (c["dense_score_raw"] - dense_min) / dense_range if c["dense_score_raw"] > 0 else 0.0
            c["bm25_score_norm"] = (c["bm25_score_raw"] - bm25_min) / bm25_range if c["bm25_score_raw"] > 0 else 0.0
            c["fusion_score"] = (self.alpha * c["dense_score_norm"] +
                                 self.beta * c["bm25_score_norm"])

        # Sort by fusion score and take top M for reranking
        candidates.sort(key=lambda x: x["fusion_score"], reverse=True)
        return candidates[:self.merge_top_m]

    def retrieve(self, query: str, top_n: int = None) -> dict:
        """
        Full hybrid retrieval pipeline.

        Returns dict with:
            - passages: list of top-N passages with metadata
            - trace: retrieval trace with IDs and timings
        """
        top_n = top_n or self.final_top_n
        trace = {}
        t0 = time.time()

        # Step 1: Dense retrieval
        t1 = time.time()
        dense_candidates = self._dense_retrieve(query)
        trace["dense_ms"] = round((time.time() - t1) * 1000, 1)
        trace["dense_ids"] = [c["chunk_id"] for c in dense_candidates[:10]]

        # Step 2: BM25 retrieval
        t2 = time.time()
        bm25_candidates = self._bm25_retrieve(query)
        trace["bm25_ms"] = round((time.time() - t2) * 1000, 1)
        trace["bm25_ids"] = [c["chunk_id"] for c in bm25_candidates[:10]]

        # Step 3: Merge and normalize
        t3 = time.time()
        merged = self._merge_and_normalize(dense_candidates, bm25_candidates)
        trace["merge_ms"] = round((time.time() - t3) * 1000, 1)
        trace["merged_count"] = len(merged)

        # Step 4: Cross-encoder reranking
        t4 = time.time()
        reranked = self.reranker.rerank(query, merged, top_n=top_n)
        trace["rerank_ms"] = round((time.time() - t4) * 1000, 1)
        trace["reranked_ids"] = [c["chunk_id"] for c in reranked]

        # Build clean output
        passages = []
        for p in reranked:
            passages.append({
                "chunk_id": p["chunk_id"],
                "paper_id": p.get("metadata", {}).get("paper_id", ""),
                "title": p.get("metadata", {}).get("title", ""),
                "authors": p.get("metadata", {}).get("authors", ""),
                "categories": p.get("metadata", {}).get("categories", ""),
                "chunk_text": p["chunk_text"],
                "rerank_score": p.get("rerank_score", 0.0),
                "fusion_score": p.get("fusion_score", 0.0),
                "sources": p.get("sources", []),
            })

        trace["total_ms"] = round((time.time() - t0) * 1000, 1)

        return {
            "passages": passages,
            "trace": trace,
        }

    def retrieve_ids(self, query: str, top_n: int = None) -> list[str]:
        """Convenience method that returns only chunk IDs (for evaluation)."""
        result = self.retrieve(query, top_n=top_n)
        return [p["chunk_id"] for p in result["passages"]]
