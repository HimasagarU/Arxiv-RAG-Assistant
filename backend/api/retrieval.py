"""
retrieval.py — Text-only hybrid retrieval with Qdrant Cloud + PostgreSQL FTS.

Pipeline:
1. Classify query intent (explanatory, comparative, technical, sota, discovery)
2. Dense retrieval from the text collection
3. PostgreSQL full-text retrieval over text chunks
4. RRF fusion
5. Cross-encoder reranking
6. Context compression
"""

import logging
import os
import re
import time
from collections import Counter
from typing import Optional
from uuid import uuid5, NAMESPACE_URL

import numpy as np
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
import joblib
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, SearchParams

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

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_EXPLANATORY = "explanatory"
INTENT_SOTA = "sota"
INTENT_COMPARATIVE = "comparative"
INTENT_TECHNICAL = "technical"
INTENT_DISCOVERY = "discovery"

_INTENT_RULES = [
    (re.compile(r"\b(what\s+is|what\s+are|how\s+does|how\s+do|explain|define|describe|overview\s+of|introduction\s+to|basics\s+of|concept\s+of|meaning\s+of|tell\s+me\s+about)\b", re.I), INTENT_EXPLANATORY),
    (re.compile(r"\b(compare|vs\.?|versus|difference\s+between|compared\s+to|similarities|pros\s+and\s+cons|advantages\s+over|trade.?offs?)\b", re.I), INTENT_COMPARATIVE),
    (re.compile(r"\b(latest|newest|recent|state.of.the.art|sota|cutting.edge|current\s+trends?|advances?\s+in|progress\s+in|2024|2025|2026)\b", re.I), INTENT_SOTA),
    (re.compile(r"\b(derive|proof|prove|formal\s+definition|theorem|lemma|mathematical|equation\s+for|algorithm\s+for|pseudocode|formula)\b", re.I), INTENT_TECHNICAL),
]


def classify_query_intent(query: str) -> str:
    q = query.strip()
    for pattern, intent in _INTENT_RULES:
        if pattern.search(q):
            return intent
    if q.endswith('?') and len(q.split()) <= 8:
        return INTENT_EXPLANATORY
    return INTENT_DISCOVERY


# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------

COLLECTION_TEXT = "arxiv_text"
ALL_COLLECTIONS = [COLLECTION_TEXT]

# Max chunks per paper in final output (diversity control)
MAX_CHUNKS_PER_PAPER = int(os.getenv("MAX_CHUNKS_PER_PAPER", "2"))

# Qdrant search ef (higher = better recall at query time)
QDRANT_SEARCH_EF = 200

# Intent-aware RRF fusion weights: (dense_weight, lexical_weight)
INTENT_RRF_WEIGHTS = {
    INTENT_EXPLANATORY: (0.7, 0.3),   # concept queries favor dense
    INTENT_COMPARATIVE: (0.6, 0.4),
    INTENT_TECHNICAL:   (0.5, 0.5),   # balanced for equations/formulas
    INTENT_SOTA:        (0.7, 0.3),
    INTENT_DISCOVERY:   (0.4, 0.6),   # keyword-heavy queries favor FTS
}

def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a deterministic UUID (must match build_qdrant.py)."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Multi-collection hybrid retriever with Qdrant + modality routing."""

    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embedding_model: str = None,
        reranker_model: str = None,
        k_dense: int = 50,
        k_lex: int = 50,
        merge_top_m: int = 30,
        final_top_n: int = 5,
        rrf_k: int = 60,
    ):
        self.k_dense = k_dense
        self.k_lex = k_lex
        self.merge_top_m = merge_top_m
        self.final_top_n = final_top_n
        self.rrf_k = int(os.getenv("RRF_K", str(rrf_k)))

        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        reranker_model = reranker_model or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        log.info("Initializing HybridRetriever (Qdrant text-only)")

        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is required for cloud deployment.")

        # Embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading embedding model: {embedding_model} (device={device})")
        self.embed_model = SentenceTransformer(embedding_model, device=device)

        # Qdrant Cloud client
        log.info(f"Connecting to Qdrant Cloud: {qdrant_url}")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Discover available collection
        self.collections = {}
        try:
            all_cols = self.qdrant_client.get_collections().collections
            available = {c.name for c in all_cols}
            if COLLECTION_TEXT in available:
                info = self.qdrant_client.get_collection(COLLECTION_TEXT)
                self.collections[COLLECTION_TEXT] = info.points_count
                log.info(f"  {COLLECTION_TEXT}: {info.points_count} points")
            else:
                log.warning(f"  {COLLECTION_TEXT}: not found (will skip)")
        except Exception as e:
            log.error(f"Failed to list Qdrant collections: {e}")

        # Reranker
        self.reranker = Reranker(model_name=reranker_model)

        # Artifacts
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        self.bm25 = None
        self.chunks_meta = []
        self.chunks_text = {}
        self.papers_meta = {}
        
        try:
            with open(data_dir / "papers_meta.json", "r", encoding="utf-8") as f:
                self.papers_meta = json.load(f)
            log.info(f"Loaded {len(self.papers_meta)} papers from papers_meta.json")
        except Exception as e:
            log.warning(f"Could not load papers_meta.json: {e}")
            
        try:
            with open(data_dir / "chunks_meta.jsonl", "r", encoding="utf-8") as f:
                self.chunks_meta = [json.loads(line) for line in f]
            log.info(f"Loaded {len(self.chunks_meta)} chunk metadata entries.")
        except Exception as e:
            log.warning(f"Could not load chunks_meta.jsonl: {e}")
            
        try:
            self.bm25 = joblib.load(data_dir / "bm25_v1.pkl")
            log.info("BM25 index loaded successfully.")
        except Exception as e:
            log.warning(f"Could not load bm25_v1.pkl: {e}")
            
        try:
            with open(data_dir / "chunks_text.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.chunks_text[entry["chunk_id"]] = entry.get("text", "")
            log.info(f"Loaded texts for {len(self.chunks_text)} chunks.")
        except Exception as e:
            log.warning(f"Could not load chunks_text.jsonl: {e}")

        log.info("HybridRetriever ready.")

    def add_paper(self, paper_meta: dict, chunks: list[dict]):
        """Dynamically add a new paper to the in-memory retriever structures."""
        paper_id = paper_meta["paper_id"]
        
        # 1. Update papers meta
        self.papers_meta[paper_id] = {
            "title": paper_meta.get("title", ""),
            "authors": paper_meta.get("authors", ""),
            "published": paper_meta.get("published", ""),
            "categories": paper_meta.get("categories", "")
        }
        
        # 2. Update chunks meta and text
        new_tokens_list = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            self.chunks_meta.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title": chunk.get("title", ""),
                "authors": chunk.get("authors", ""),
                "categories": chunk.get("categories", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
                "section_hint": chunk.get("section_hint", "other"),
                "layer": chunk.get("layer", "core"),
                "token_count": chunk.get("token_count", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "chunk_source": chunk.get("chunk_source", "full_text")
            })
            self.chunks_text[chunk_id] = chunk["chunk_text"]
            
            # Tokenize for BM25
            if self.bm25 and hasattr(self.bm25, 'tokenizer'):
                tokens = self.bm25.tokenizer(chunk["chunk_text"])
                new_tokens_list.append(tokens)
        
        # 3. Dynamically update BM25 index (approximate)
        if self.bm25 and new_tokens_list:
            for tokens in new_tokens_list:
                freqs = {}
                for t in tokens:
                    freqs[t] = freqs.get(t, 0) + 1
                self.bm25.doc_freqs.append(freqs)
                self.bm25.doc_len.append(len(tokens))
                self.bm25.corpus_size += 1
                
                # Assign average IDF to completely unseen terms
                for t in freqs:
                    if t not in self.bm25.idf:
                        self.bm25.idf[t] = getattr(self.bm25, 'average_idf', 0.0)
            
            self.bm25.avgdl = sum(self.bm25.doc_len) / max(1, self.bm25.corpus_size)

        # 4. Persist to local artifact files to survive restarts
        try:
            data_dir = Path(os.getenv("DATA_DIR", "data"))
            
            # Re-write papers_meta.json
            with open(data_dir / "papers_meta.json", "w", encoding="utf-8") as f:
                json.dump(self.papers_meta, f, indent=2)
                
            # Append to chunks_meta.jsonl
            with open(data_dir / "chunks_meta.jsonl", "a", encoding="utf-8") as f:
                for chunk in chunks:
                    chunk_id = chunk["chunk_id"]
                    meta_entry = {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "title": chunk.get("title", ""),
                        "authors": chunk.get("authors", ""),
                        "categories": chunk.get("categories", ""),
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "section_hint": chunk.get("section_hint", "other"),
                        "layer": chunk.get("layer", "core"),
                        "token_count": chunk.get("token_count", 0),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "total_chunks": chunk.get("total_chunks", 1),
                        "chunk_source": chunk.get("chunk_source", "full_text")
                    }
                    f.write(json.dumps(meta_entry) + "\n")
                    
            # Append to chunks_text.jsonl
            with open(data_dir / "chunks_text.jsonl", "a", encoding="utf-8") as f:
                for chunk in chunks:
                    text_entry = {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["chunk_text"]
                    }
                    f.write(json.dumps(text_entry) + "\n")
                    
            log.info(f"Successfully persisted metadata for {paper_id} to disk.")
        except Exception as e:
            log.error(f"Failed to persist newly added paper to local files: {e}")

    # ------------------------------------------------------------------
    # Dense retrieval (Qdrant multi-collection)
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> list[float]:
        """Encode query for BGE model (no prefix needed)."""
        emb = self.embed_model.encode([query], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        return emb[0].tolist()

    def _build_qdrant_filter(self, category: Optional[str] = None,
                             author: Optional[str] = None,
                             paper_id: Optional[str] = None) -> Optional[Filter]:
        """Build a Qdrant filter from user-supplied metadata filters.
        
        When paper_id is set, adds a strict MatchValue filter to scope
        retrieval to a single document (used by chat-with-document).
        """
        conditions = []
        if paper_id:
            conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))
        if category:
            conditions.append(FieldCondition(key="categories", match=MatchText(text=category.strip())))
        if author:
            conditions.append(FieldCondition(key="authors", match=MatchText(text=author.strip())))

        if not conditions:
            return None
        return Filter(must=conditions)

    def _dense_retrieve_collection(
        self, collection_name: str, query_embedding: list[float],
        n_results: int = 50, qdrant_filter: Optional[Filter] = None
    ) -> list[dict]:
        if collection_name not in self.collections:
            return []

        try:
            res = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=n_results,
                search_params=SearchParams(hnsw_ef=QDRANT_SEARCH_EF),
                with_payload=True,
            )
            results = res.points
        except Exception as e:
            log.warning(f"Qdrant search failed on {collection_name}: {e}")
            if qdrant_filter:
                try:
                    res = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_embedding,
                        limit=n_results,
                        search_params=SearchParams(hnsw_ef=QDRANT_SEARCH_EF),
                        with_payload=True,
                    )
                    results = res.points
                except Exception:
                    return []
            else:
                return []

        candidates = []
        for point in results:
            payload = point.payload or {}
            candidates.append({
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "chunk_text": payload.get("chunk_text", ""),
                "metadata": {
                    "paper_id": payload.get("paper_id", ""),
                    "title": payload.get("title", ""),
                    "authors": payload.get("authors", ""),
                    "categories": payload.get("categories", ""),
                    "chunk_type": payload.get("chunk_type", "text"),
                    "modality": payload.get("modality", "text"),
                    "section_hint": payload.get("section_hint", "other"),
                    "layer": payload.get("layer", "core"),
                    "token_count": payload.get("token_count", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 1),
                    "chunk_source": payload.get("chunk_source", "full_text"),
                },
                "dense_score": point.score,
                "source": "dense",
                "collection": collection_name,
            })
        return candidates

    def _dense_retrieve(self, query: str, qdrant_filter: Optional[Filter] = None) -> list[dict]:
        query_emb = self._encode_query(query)
        return self._dense_retrieve_collection(
            COLLECTION_TEXT, query_emb, self.k_dense, qdrant_filter
        )

    # ------------------------------------------------------------------
    # Lexical retrieval (PostgreSQL FTS)
    # ------------------------------------------------------------------

    def _lexical_retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        author: Optional[str] = None,
        start_year: Optional[int] = None,
    ) -> list[dict]:
        if not self.bm25 or not self.chunks_meta:
            return []
            
        tokens = re.sub(r'[^\w\s]', '', query.lower()).split()
        if not tokens:
            return []
            
        scores = self.bm25.get_scores(tokens)
        
        # Fast top-K selection
        k = min(self.k_lex * 5, len(scores))
        if k == 0:
            return []
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        candidates = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.chunks_meta[idx]
            paper_id = meta.get("paper_id", "")
            
            if category and category.lower() not in meta.get("categories", "").lower():
                continue
            if author and author.lower() not in meta.get("authors", "").lower():
                continue
            if start_year:
                paper_info = self.papers_meta.get(paper_id, {})
                published = paper_info.get("published")
                if not published or int(published.split("-")[0]) < start_year:
                    continue
                    
            chunk_id = meta["chunk_id"]
            candidates.append({
                "chunk_id": chunk_id,
                "chunk_text": self.chunks_text.get(chunk_id, ""),
                "lex_score": float(scores[idx]),
                "chunk_type": meta.get("chunk_type", "text"),
                "metadata": {
                    "paper_id": paper_id,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "categories": meta.get("categories", ""),
                    "section_hint": meta.get("section_hint", "other"),
                    "layer": meta.get("layer", "core"),
                    "chunk_source": meta.get("chunk_source", "full_text"),
                },
                "source": "lexical",
            })
            if len(candidates) >= self.k_lex:
                break
                
        return candidates

    # ------------------------------------------------------------------
    # Merge + RRF fusion with modality boost
    # ------------------------------------------------------------------

    def _merge_and_normalize(
        self, dense_candidates: list[dict], lex_candidates: list[dict],
        intent: str = INTENT_DISCOVERY
    ) -> list[dict]:
        merged = {}
        dense_rank = {c["chunk_id"]: r for r, c in enumerate(dense_candidates, 1)}
        lex_rank = {c["chunk_id"]: r for r, c in enumerate(lex_candidates, 1)}

        for c in dense_candidates:
            merged[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "chunk_text": c.get("chunk_text", ""),
                "metadata": c.get("metadata", {}),
                "dense_score_raw": c["dense_score"],
                "lex_score_raw": 0.0,
                "sources": ["dense"],
            }

        for c in lex_candidates:
            cid = c["chunk_id"]
            if cid in merged:
                merged[cid]["lex_score_raw"] = c["lex_score"]
                if "lexical" not in merged[cid]["sources"]:
                    merged[cid]["sources"].append("lexical")
                if not merged[cid].get("chunk_text"):
                    merged[cid]["chunk_text"] = c.get("chunk_text", "")
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "chunk_text": c.get("chunk_text", ""),
                    "metadata": c.get("metadata", {}),
                    "dense_score_raw": 0.0,
                    "lex_score_raw": c["lex_score"],
                    "sources": ["lexical"],
                    "chunk_type": c.get("chunk_type", "text"),
                }

        if not merged:
            return []

        candidates = list(merged.values())

        for c in candidates:
            dr = dense_rank.get(c["chunk_id"])
            lr = lex_rank.get(c["chunk_id"])
            c["dense_rrf"] = 1.0 / (self.rrf_k + dr) if dr else 0.0
            c["lex_rrf"] = 1.0 / (self.rrf_k + lr) if lr else 0.0

            # Intent-aware weighted RRF fusion
            w_dense, w_lex = INTENT_RRF_WEIGHTS.get(intent, (0.5, 0.5))
            c["fusion_score"] = c["dense_rrf"] * w_dense + c["lex_rrf"] * w_lex

        candidates.sort(key=lambda x: x["fusion_score"], reverse=True)
        return candidates[:self.merge_top_m]

    # ------------------------------------------------------------------
    # Layer-aware recency boost
    # ------------------------------------------------------------------

    def _apply_recency_boost(self, candidates: list[dict], boost_weight: float = 0.2) -> list[dict]:
        for c in candidates:
            meta = c.get("metadata", {})
            layer = meta.get("layer", "core")
            if layer == "latest":
                c["fusion_score"] = c.get("fusion_score", 0) + boost_weight * 0.005
        return candidates

    def _filter_candidates_by_year(self, candidates: list[dict], start_year: Optional[int]) -> list[dict]:
        if not start_year:
            return candidates

        filtered = []
        for candidate in candidates:
            paper_id = candidate.get("metadata", {}).get("paper_id", "")
            if not paper_id:
                continue
            paper_info = self.papers_meta.get(paper_id)
            if paper_info:
                published = paper_info.get("published")
                if published and int(published.split("-")[0]) >= start_year:
                    filtered.append(candidate)
            else:
                filtered.append(candidate)
        return filtered

    # ------------------------------------------------------------------
    # Semantic pruning
    # ------------------------------------------------------------------

    def _semantic_pruning(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        high_signal = {"we propose", "method", "pipeline", "architecture", "algorithm",
                       "training", "approach", "framework", "results", "evaluation"}
        pruned = []
        for c in candidates:
            text_lower = c.get("chunk_text", "").lower()
            meta = c.get("metadata", {})
            section = meta.get("section_hint", "other")
            score = 0
            if section in ("method", "abstract", "introduction", "background"):
                score += 2
            if any(kw in text_lower for kw in high_signal):
                score += 3
            if score >= 2 or len(pruned) < 3:
                pruned.append(c)
        return pruned

    # ------------------------------------------------------------------
    # Paper-level diversity: max N chunks per paper
    # ------------------------------------------------------------------

    def _enforce_paper_diversity(self, candidates: list[dict],
                                  max_per_paper: int = MAX_CHUNKS_PER_PAPER) -> list[dict]:
        """Deduplicate by paper_id, keeping at most max_per_paper chunks per paper."""
        paper_counts = {}
        diverse = []
        for c in candidates:
            pid = c.get("metadata", {}).get("paper_id", c.get("paper_id", ""))
            if not pid:
                diverse.append(c)
                continue
            paper_counts[pid] = paper_counts.get(pid, 0) + 1
            if paper_counts[pid] <= max_per_paper:
                diverse.append(c)
        return diverse

    # ------------------------------------------------------------------
    # Layer-aware balancing
    # ------------------------------------------------------------------

    def _ensure_layer_coverage(self, candidates: list[dict],
                                merged_pool: list[dict],
                                min_foundational: int = 1) -> list[dict]:
        """Ensure at least one foundational/prerequisite chunk in results for broad queries.
        
        If no foundational chunk is present in top results, inject the best-scoring one
        from the merged pool (replacing the weakest result).
        """
        foundational_layers = {"prerequisite", "foundation"}
        has_foundational = any(
            c.get("metadata", {}).get("layer", "") in foundational_layers
            for c in candidates
        )

        if has_foundational or not candidates:
            return candidates

        # Find best foundational candidate from the larger pool
        for c in merged_pool:
            if c.get("metadata", {}).get("layer", "") in foundational_layers:
                if c["chunk_id"] not in {x["chunk_id"] for x in candidates}:
                    # Replace the weakest result
                    candidates[-1] = c
                    return candidates

        return candidates

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def extract_analytics(self, candidates: list[dict]) -> dict:
        author_counter, category_counter, layer_counter = Counter(), Counter(), Counter()
        for c in candidates:
            meta = c.get("metadata", {})
            for a in (meta.get("authors", "") or "").split(","):
                a = a.strip()
                if a:
                    author_counter[a] += 1
            for cat in (meta.get("categories", "") or "").split(","):
                cat = cat.strip()
                if cat:
                    category_counter[cat] += 1
            layer_counter[meta.get("layer", "core")] += 1

        return {
            "top_authors": [{"name": a, "count": c} for a, c in author_counter.most_common(5)],
            "top_categories": [{"name": cat, "count": c} for cat, c in category_counter.most_common(5)],
            "layer_distribution": dict(layer_counter),
            "total_unique_papers": len({c.get("metadata", {}).get("paper_id", "") for c in candidates}),
        }

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    def compress_context(self, query: str, passages: list[dict],
                         max_sentences: int = 25, intent: str = INTENT_DISCOVERY) -> str:
        if intent == INTENT_EXPLANATORY:
            parts = []
            for i, p in enumerate(passages, 1):
                title = p.get("title", p.get("metadata", {}).get("title", ""))
                text = p.get("chunk_text", "")
                if text:
                    header = f"[Source {i}: {title}]" if title else f"[Source {i}]"
                    parts.append(f"{header}\n{text}")
            return "\n\n".join(parts)

        return " ".join(p.get("chunk_text", "") for p in passages if p.get("chunk_text"))

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_n: int = None,
                 category: Optional[str] = None, author: Optional[str] = None,
                 start_year: Optional[int] = None, intent: Optional[str] = None,
                 paper_id: Optional[str] = None) -> dict:
        """Main retrieval pipeline.
        
        Args:
            paper_id: If set, restricts retrieval to chunks from this paper only
                      (used by document-scoped chat).
        """
        top_n = top_n or self.final_top_n
        intent = intent or classify_query_intent(query)
        is_paper_scoped = paper_id is not None
        trace = {"intent": intent, "paper_scoped": is_paper_scoped}
        t0 = time.time()

        is_explanatory = (intent == INTENT_EXPLANATORY)
        qdrant_filter = self._build_qdrant_filter(
            category=category, author=author, paper_id=paper_id
        )

        # Dense retrieval (multi-collection via Qdrant)
        t1 = time.time()
        dense_candidates = self._dense_retrieve(query, qdrant_filter)
        trace["dense_ms"] = round((time.time() - t1) * 1000, 1)
        trace["dense_count"] = len(dense_candidates)

        # Lexical retrieval — filter by paper_id if scoped
        t2 = time.time()
        lex_candidates = self._lexical_retrieve(
            query,
            category=category,
            author=author,
            start_year=start_year,
        )
        # Post-filter BM25 results by paper_id for document-scoped chat
        if paper_id and lex_candidates:
            lex_candidates = [
                c for c in lex_candidates
                if c.get("metadata", {}).get("paper_id") == paper_id
            ]
        trace["lex_ms"] = round((time.time() - t2) * 1000, 1)
        trace["lex_count"] = len(lex_candidates)

        # Merge + RRF + modality boost
        orig_merge = self.merge_top_m
        if is_explanatory:
            self.merge_top_m = 40  # Wider pool for explanatory queries
        t3 = time.time()
        merged = self._merge_and_normalize(dense_candidates, lex_candidates, intent)
        trace["merge_ms"] = round((time.time() - t3) * 1000, 1)
        trace["merged_count"] = len(merged)
        self.merge_top_m = orig_merge

        # Skip corpus-wide balancing for paper-scoped retrieval
        if not is_paper_scoped:
            # Recency / layer boost (skip for explanatory)
            if not is_explanatory:
                merged = self._apply_recency_boost(merged)
                merged.sort(key=lambda x: x["fusion_score"], reverse=True)

            merged = self._filter_candidates_by_year(merged, start_year)

            # Semantic pruning for explanatory
            if is_explanatory:
                merged = self._semantic_pruning(query, merged)

            # Paper-level diversity: deduplicate BEFORE reranking
            merged = self._enforce_paper_diversity(merged, max_per_paper=MAX_CHUNKS_PER_PAPER)

        trace["diverse_count"] = len(merged)

        # Rerank
        t4 = time.time()
        rerank_mode = "combined" if is_explanatory else "default"
        reranked = self.reranker.rerank(query, merged, top_n=top_n, rerank_text_mode=rerank_mode)
        trace["rerank_ms"] = round((time.time() - t4) * 1000, 1)

        # Layer-aware balancing for broad/explanatory queries
        if is_explanatory:
            reranked = self._ensure_layer_coverage(reranked, merged)

        analytics = self.extract_analytics(merged)

        # Build output
        passages = []
        for p in reranked:
            meta = p.get("metadata", {})
            passages.append({
                "chunk_id": p["chunk_id"],
                "paper_id": meta.get("paper_id", ""),
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "categories": meta.get("categories", ""),
                "chunk_text": p["chunk_text"],
                "chunk_type": meta.get("chunk_type", "text"),
                "modality": meta.get("modality", "text"),
                "section_hint": meta.get("section_hint", "other"),
                "layer": meta.get("layer", "core"),
                "rerank_score": p.get("rerank_score", 0.0),
                "fusion_score": p.get("fusion_score", 0.0),
                "sources": p.get("sources", []),
            })

        trace["total_ms"] = round((time.time() - t0) * 1000, 1)
        trace["filters"] = {"category": category, "author": author, "start_year": start_year}
        trace["unique_papers"] = len({p["paper_id"] for p in passages if p["paper_id"]})

        return {"passages": passages, "trace": trace, "analytics": analytics}

    def retrieve_ids(self, query: str, top_n: int = None) -> list[str]:
        result = self.retrieve(query, top_n=top_n)
        return [p["chunk_id"] for p in result["passages"]]

    # ------------------------------------------------------------------
    # Similar papers
    # ------------------------------------------------------------------

    def find_similar_papers(self, paper_id: str, top_n: int = 5) -> list[dict]:
        if COLLECTION_TEXT not in self.collections:
            return []
        try:
            log.info(f"Searching for papers similar to: {paper_id}")
            # Scroll to get all points for this paper
            scroll_result = self.qdrant_client.scroll(
                collection_name=COLLECTION_TEXT,
                scroll_filter=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                with_vectors=True,
                with_payload=True,
                limit=100,
            )
            points = scroll_result[0]
            
            # Fallback to MatchText if MatchValue returns nothing
            if not points:
                log.info(f"MatchValue for paper_id {paper_id} returned no points. Retrying with MatchText...")
                scroll_result = self.qdrant_client.scroll(
                    collection_name=COLLECTION_TEXT,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="paper_id", match=MatchText(text=paper_id))]
                    ),
                    with_vectors=True,
                    with_payload=True,
                    limit=100,
                )
                points = scroll_result[0]

            if not points:
                log.warning(f"No points found for paper_id {paper_id} in Qdrant.")
                return []

            log.info(f"Found {len(points)} points for paper_id {paper_id}. Computing mean embedding...")

            # Compute mean embedding
            embeddings = [p.vector for p in points if p.vector is not None]
            if not embeddings:
                log.warning(f"Points found for paper_id {paper_id} but none had vectors.")
                return []
                
            embeddings = np.array(embeddings)
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)

            # Search for similar
            log.info(f"Querying Qdrant for similar points (excluding {paper_id})...")
            res = self.qdrant_client.query_points(
                collection_name=COLLECTION_TEXT,
                query=mean_emb.tolist(),
                query_filter=Filter(
                    must_not=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                limit=100,
                with_payload=True,
            )
            results = res.points
            log.info(f"Qdrant returned {len(results)} candidate points.")

            seen = {paper_id}
            papers = []
            for point in results:
                payload = point.payload or {}
                pid = payload.get("paper_id", "")
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append({
                        "paper_id": pid,
                        "title": payload.get("title", ""),
                        "authors": payload.get("authors", ""),
                        "categories": payload.get("categories", ""),
                        "layer": payload.get("layer", ""),
                        "similarity_score": round(point.score, 4),
                        "chunk_text": (payload.get("chunk_text", "") or "")[:300],
                    })
                    if len(papers) >= top_n:
                        break
            
            log.info(f"Found {len(papers)} unique similar papers.")
            return papers
        except Exception as e:
            log.error(f"Similar papers search failed: {e}")
            return []
