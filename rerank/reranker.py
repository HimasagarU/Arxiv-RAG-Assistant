"""
reranker.py — Cross-encoder reranking for candidate passages.

Loads a cross-encoder model (default: ms-marco-MiniLM-L-6-v2) on GPU
and provides a rerank function for (query, passage) pairs.
"""

import logging
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for passage re-scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: Optional[str] = None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        log.info(f"Loading reranker model: {model_name} on {self.device}")
        self.model = CrossEncoder(model_name, device=self.device)
        log.info("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_n: int = 5,
        text_key: str = "chunk_text",
    ) -> list[dict]:
        """
        Rerank passages by cross-encoder score.

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

        # Build (query, passage) pairs
        pairs = [(query, p[text_key]) for p in passages]

        # Score all pairs
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                    show_progress_bar=False)

        # Attach scores and sort
        for i, p in enumerate(passages):
            p["rerank_score"] = float(scores[i])

        reranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
