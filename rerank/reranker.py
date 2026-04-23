"""
reranker.py — Cross-encoder reranking for candidate passages.

Uses sentence-transformers CrossEncoder.
"""

import logging
import os
from typing import Optional

# We use sentence_transformers instead of flashrank
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
    If no namespace (like 'cross-encoder/') is provided, add it.
    """
    if "/" not in model_name:
        return f"cross-encoder/{model_name}"
    return model_name


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        lazy_load: bool = True,
        enabled: bool = True,
    ):
        self.batch_size = batch_size
        self.enabled = enabled
        self.lazy_load = lazy_load
        self.model_name = _normalize_hf_model_name(model_name)
        self.device = device
        self.ranker: Optional[CrossEncoder] = None

        if not self.enabled:
            log.info("CrossEncoder reranker disabled via configuration.")
            return

        if self.lazy_load:
            log.info(
                f"CrossEncoder reranker enabled (lazy load): {self.model_name}"
            )
            return

        self._ensure_ranker_loaded()

    def _ensure_ranker_loaded(self):
        """Load the CrossEncoder model once, on-demand."""
        if not self.enabled or self.ranker is not None:
            return

        log.info(f"Loading CrossEncoder reranker model: {self.model_name}")
        self.ranker = CrossEncoder(self.model_name, device=self.device)
        log.info("CrossEncoder reranker model loaded.")

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

        if not self.enabled:
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        try:
            self._ensure_ranker_loaded()
        except Exception as e:
            log.warning(f"Failed to load CrossEncoder model ({self.model_name}): {e}")
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        if self.ranker is None:
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        # Pass more candidates through the reranker than requested
        rerank_pool_size = min(len(passages), top_n * 3)
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

        try:
            scores = self.ranker.predict(pairs, batch_size=self.batch_size)
        except Exception as e:
            log.warning(f"CrossEncoder rerank predict failed; using fusion order fallback: {e}")
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        # Map scores back to the rerank pool
        for i, p in enumerate(rerank_pool):
            p["rerank_score"] = float(scores[i])

        # Sort pool by reranker score and return top_n
        reranked = sorted(rerank_pool, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
