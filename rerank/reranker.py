"""reranker.py — Cross-encoder reranking for candidate passages.

Uses sentence-transformers CrossEncoder (PyTorch).
"""

import logging
import os
from typing import Optional

from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _normalize_hf_model_name(model_name: str) -> str:
    """
    Ensure the model name is properly formatted for HuggingFace.
    If no namespace is provided, add 'cross-encoder/'.
    BGE reranker models use 'BAAI/' namespace.
    """
    if "/" not in model_name:
        if "bge-reranker" in model_name.lower():
            return f"BAAI/{model_name}"
        return f"cross-encoder/{model_name}"
    return model_name


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.batch_size = batch_size
        self.model_name = _normalize_hf_model_name(model_name)
        self.device = device
        self.ranker = None

        enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        if not enable_reranker:
            log.info("Reranker is explicitly disabled via ENABLE_RERANKER=false")
            return

        log.info(f"Loading CrossEncoder reranker model: {self.model_name}")
        try:
            self.ranker = CrossEncoder(self.model_name, device=self.device)
            log.info("CrossEncoder loaded. Running warmup...")
            self.ranker.predict([("warmup", "warmup")])
            log.info("Reranker ready.")
        except Exception as e:
            log.error(f"Failed to load reranker model {self.model_name}: {e}")
            log.warning("System will fall back to fusion scores (no reranking).")

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_n: int = 5,
        text_key: str = "chunk_text",
        rerank_text_mode: str = "default",
    ) -> list[dict]:
        """
        Rerank passages by cross-encoder score.

        Args:
            query: Search query string.
            passages: List of dicts, each with at least `text_key` field.
            top_n: Number of top results to return.
            text_key: Key in passage dict containing the text.
            rerank_text_mode: "default" uses chunk_text only.
                              "combined" uses "title — chunk_text" for richer signal
                              (better for explanatory queries).

        Returns:
            Top-N passages sorted by reranker score, with 'rerank_score' added.
        """
        if not passages:
            return []
            
        if self.ranker is None:
            # Fallback if reranker is disabled or failed to load
            for p in passages[:top_n]:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        # Pass more candidates through the reranker than requested (limit drastically if on slow CPU)
        multiplier = 3 if self.device != "cpu" else 1
        rerank_pool_size = min(len(passages), top_n * multiplier)
        rerank_pool = passages[:rerank_pool_size]

        # Build [query, text] pairs for cross-encoder
        pairs = []
        for p in rerank_pool:
            if rerank_text_mode == "combined":
                title = p.get("metadata", {}).get("title", "") or p.get("title", "")
                text = p.get(text_key, "")
                combined = f"{title} — {text}" if title else text
                pairs.append([query, combined])
            else:
                pairs.append([query, p.get(text_key, "")])

        scores = self.ranker.predict(pairs, batch_size=self.batch_size)

        # Map scores back to the rerank pool
        for i, p in enumerate(rerank_pool):
            p["rerank_score"] = float(scores[i])

        # Sort pool by reranker score and return top_n
        reranked = sorted(rerank_pool, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
