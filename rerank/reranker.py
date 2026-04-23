"""
reranker.py — Cross-encoder reranking for candidate passages.

Uses FlashRank (ONNX-based, no PyTorch needed) for ultra-lightweight
cross-encoder reranking. Default model: ms-marco-MiniLM-L-6-v2.
"""

import logging
import os
from typing import Optional

from flashrank import Ranker, RerankRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _normalize_flashrank_model_name(model_name: str) -> str:
    """
    FlashRank expects short model names (e.g. ms-marco-MiniLM-L-6-v2).
    Accept HuggingFace-style names and normalize to the final segment.
    """
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


class Reranker:
    """Cross-encoder reranker using FlashRank (ONNX, no PyTorch)."""

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
        self.model_name = _normalize_flashrank_model_name(model_name)
        self.cache_dir = os.getenv("FLASHRANK_CACHE_DIR", "/tmp/flashrank_cache")
        self.ranker: Optional[Ranker] = None

        if not self.enabled:
            log.info("FlashRank reranker disabled via configuration.")
            return

        if self.lazy_load:
            log.info(
                f"FlashRank reranker enabled (lazy load): {self.model_name}"
            )
            return

        self._ensure_ranker_loaded()

    def _ensure_ranker_loaded(self):
        """Load the FlashRank model once, on-demand."""
        if not self.enabled or self.ranker is not None:
            return

        log.info(f"Loading FlashRank reranker model: {self.model_name}")
        self.ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        log.info("FlashRank reranker model loaded.")

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_n: int = 5,
        text_key: str = "chunk_text",
        rerank_text_mode: str = "default",
    ) -> list[dict]:
        """
        Rerank passages by cross-encoder score via FlashRank.

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
            log.warning(f"Failed to load FlashRank model ({self.model_name}): {e}")
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        if self.ranker is None:
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        # Pass more candidates through the reranker than requested,
        # so it has room to discover truly relevant passages
        rerank_pool_size = min(len(passages), top_n * 3)
        rerank_pool = passages[:rerank_pool_size]

        # Build text for reranking
        flashrank_passages = []
        for i, p in enumerate(rerank_pool):
            if rerank_text_mode == "combined":
                title = p.get("metadata", {}).get("title", "") or p.get("title", "")
                text = p.get(text_key, "")
                combined = f"{title} — {text}" if title else text
                flashrank_passages.append({"id": i, "text": combined})
            else:
                flashrank_passages.append({"id": i, "text": p.get(text_key, "")})

        rerank_request = RerankRequest(query=query, passages=flashrank_passages)
        try:
            results = self.ranker.rerank(rerank_request)
        except Exception as e:
            log.warning(f"FlashRank rerank failed; using fusion order fallback: {e}")
            for p in passages:
                p["rerank_score"] = p.get("fusion_score", 0.0)
            return passages[:top_n]

        # Map scores back to the rerank pool
        score_map = {r["id"]: r["score"] for r in results}
        for i, p in enumerate(rerank_pool):
            p["rerank_score"] = float(score_map.get(i, 0.0))

        # Sort pool by reranker score and return top_n
        reranked = sorted(rerank_pool, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
