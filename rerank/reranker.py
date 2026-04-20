"""
reranker.py — Cross-encoder reranking for candidate passages.

Uses FlashRank (ONNX-based, no PyTorch needed) for ultra-lightweight
cross-encoder reranking. Default model: ms-marco-MiniLM-L-12-v2 (~30MB).
"""

import logging
from typing import Optional

from flashrank import Ranker, RerankRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using FlashRank (ONNX, no PyTorch)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2",
                 device: Optional[str] = None, batch_size: int = 32):
        self.batch_size = batch_size
        log.info(f"Loading FlashRank reranker model: {model_name}")
        # FlashRank auto-downloads the ONNX model on first use (~30MB)
        self.ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
        log.info("FlashRank reranker model loaded.")

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_n: int = 5,
        text_key: str = "chunk_text",
    ) -> list[dict]:
        """
        Rerank passages by cross-encoder score via FlashRank.

        Args:
            query: Search query string.
            passages: List of dicts, each with at least `text_key` field.
            top_n: Number of top results to return.
            text_key: Key in passage dict containing the text.

        Returns:
            Top-N passages sorted by reranker score, with 'rerank_score' added.
        """
        if not passages:
            return []

        # FlashRank expects passages as list of dicts with "id" and "text" keys
        flashrank_passages = []
        for i, p in enumerate(passages):
            flashrank_passages.append({
                "id": i,
                "text": p.get(text_key, ""),
            })

        rerank_request = RerankRequest(query=query, passages=flashrank_passages)
        results = self.ranker.rerank(rerank_request)

        # Map scores back to original passages
        score_map = {r["id"]: r["score"] for r in results}
        for i, p in enumerate(passages):
            p["rerank_score"] = float(score_map.get(i, 0.0))

        reranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
