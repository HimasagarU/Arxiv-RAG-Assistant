"""
retrieval.py — Hybrid retrieval: Qdrant chunk + optional parent–child (arxiv_docs),
BM25 lexical merge, RRF, rerank, MMR, and context assembly.

Pipeline:
1. Classify query intent
2. Query decomposition / embedding + optional LLM expansion
3. Dense retrieval (arxiv_text) + optional parent–child expansion (arxiv_docs → chunks)
4. Lexical retrieval (BM25 artifacts + Qdrant payload recovery)
5. Intent-aware RRF fusion + optional citation-graph boost
6. Cross-encoder reranking, section/recency boosts, MMR, diversity
7. Context compression (caller)
"""

import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import joblib
import json
import numpy as np
import torch
from dotenv import load_dotenv
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, SearchParams, MatchAny
from sentence_transformers import SentenceTransformer

from utils.ids import chunk_id_to_uuid, paper_id_to_uuid
from utils.metadata_normalize import normalize_published
from utils.runtime import resolve_embedding_model
from utils.section_labels import normalize_section_label

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.feature_flags import env_bool, env_tri
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
INTENT_EVIDENCE = "evidence"
GENERATION_CONTEXT_TOP_N = int(os.getenv("GENERATION_CONTEXT_TOP_N", "10"))

_INTENT_RULES = [
    (re.compile(r"\b(what\s+is|what\s+are|how\s+does|how\s+do|explain|define|describe|overview\s+of|introduction\s+to|basics\s+of|concept\s+of|meaning\s+of|tell\s+me\s+about)\b", re.I), INTENT_EXPLANATORY),
    (re.compile(r"\b(summarize|summarise|summary|main\s+contributions?|key\s+contributions?|novel\s+contributions?|takeaways?|what\s+is\s+this\s+(paper|document)\s+about|overview\s+of\s+this\s+(paper|document)|summarize\s+this\s+(paper|document))\b", re.I), INTENT_EXPLANATORY),
    (re.compile(r"\b(compare|vs\.?|versus|difference\s+between|compared\s+to|similarities|pros\s+and\s+cons|advantages\s+over|trade.?offs?)\b", re.I), INTENT_COMPARATIVE),
    (re.compile(r"\b(latest|newest|recent|state.of.the.art|sota|cutting.edge|current\s+trends?|advances?\s+in|progress\s+in|2024|2025|2026)\b", re.I), INTENT_SOTA),
    (re.compile(r"\b(derive|proof|prove|formal\s+definition|theorem|lemma|mathematical|equation\s+for|algorithm\s+for|pseudocode|formula)\b", re.I), INTENT_TECHNICAL),
    (re.compile(r"\b(evidence|empirical|ablation|results\s+show|causal|demonstrate|support|proof|indicates)\b", re.I), INTENT_EVIDENCE),
]


def classify_query_intent(query: str) -> str:
    q = query.strip()
    for pattern, intent in _INTENT_RULES:
        if pattern.search(q):
            return intent
    if q.endswith('?') and len(q.split()) <= 8:
        return INTENT_EXPLANATORY
    return INTENT_DISCOVERY


def query_expansion_gate(query: str, intent: Optional[str] = None) -> dict:
    """When True, skip aggressive expansion to avoid drifting exact / technical lookups."""
    q = (query or "").strip()
    gate = {
        "restrict_decompose": False,
        "restrict_embedding_expansion": False,
        "restrict_llm_expansion": False,
    }
    
    if intent in (INTENT_TECHNICAL, INTENT_COMPARATIVE, INTENT_EVIDENCE):
        gate["restrict_embedding_expansion"] = True
        gate["restrict_llm_expansion"] = True

    if not q:
        return gate
    ql = q.lower()
    if re.search(r"\b(?:arxiv:)?\d{4}\.\d{4,5}\b", ql):
        gate["restrict_embedding_expansion"] = True
        gate["restrict_llm_expansion"] = True
        gate["restrict_decompose"] = True
    if q.count('"') >= 2:
        gate["restrict_llm_expansion"] = True
    if len(re.findall(r"\$[^$]{2,}\$", q)) >= 2 or "\\begin{" in q or "_{}" in q:
        gate["restrict_embedding_expansion"] = True
        gate["restrict_llm_expansion"] = True
    if re.search(r"\b(?:eq\.?|equation)\s*\d+", ql):
        gate["restrict_embedding_expansion"] = True
    return gate


def retrieval_skip_dense() -> bool:
    return env_bool("RETRIEVAL_SKIP_DENSE", False)


def retrieval_skip_lexical() -> bool:
    return env_bool("RETRIEVAL_SKIP_LEXICAL", False)


def retrieval_skip_parent_child() -> bool:
    return env_bool("RETRIEVAL_SKIP_PARENT_CHILD", False)


def retrieval_skip_rerank() -> bool:
    return env_bool("RETRIEVAL_SKIP_RERANK", False)


def retrieval_skip_mmr() -> bool:
    return env_bool("RETRIEVAL_SKIP_MMR", False)


def retrieval_skip_boosts() -> bool:
    return env_bool("RETRIEVAL_SKIP_BOOSTS", False)


def normalize_chunk_metadata(meta: dict) -> dict:
    """Ensure section labels are canonical for boosts and UI."""
    if not meta:
        return meta
    out = dict(meta)
    out["section_hint"] = normalize_section_label(out.get("section_hint", "other"))
    return out


def is_document_summary_query(query: str) -> bool:
    return bool(re.search(
        r"\b(summarize|summarise|summary|main\s+contributions?|key\s+contributions?|novel\s+contributions?|takeaways?|abstract|introduction|conclusion|what\s+is\s+this\s+(paper|document)\s+about|overview\s+of\s+this\s+(paper|document)|summarize\s+this\s+(paper|document))\b",
        query,
        re.I,
    ))


def decompose_query(query: str, intent: str = INTENT_DISCOVERY, paper_scoped: bool = False) -> list[str]:
    """Generate a small set of retrieval-oriented subqueries to improve recall."""
    q = " ".join((query or "").strip().split())
    if not q:
        return []

    variants = [q]

    if intent == INTENT_EXPLANATORY:
        variants.extend([
            f"{q} definition",
            f"{q} mechanism",
            f"{q} intuition",
        ])

    if intent == INTENT_EVIDENCE:
        variants.extend([
            f"{q} ablation evidence",
            f"{q} empirical evidence",
            f"{q} causal evidence",
            f"{q} results",
        ])

    if intent == INTENT_TECHNICAL:
        variants.extend([
            f"{q} equation",
            f"{q} algorithm",
            f"{q} implementation",
            f"{q} formal definition",
        ])

    if is_document_summary_query(q):
        variants.extend([
            f"{q} abstract",
            f"{q} introduction",
            f"{q} conclusion",
            "main contributions",
        ])

    if intent == INTENT_COMPARATIVE and " vs " in q.lower():
        left, right = re.split(r"\bvs\.?\b|\bversus\b", q, maxsplit=1, flags=re.I)
        for part in (left.strip(), right.strip()):
            if part:
                variants.append(part)

    if paper_scoped:
        variants.append(f"{q} in this paper")

    deduped = []
    seen = set()
    for variant in variants:
        normalized = variant.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(variant)
        if len(deduped) >= 5:
            break
    return deduped


# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------

COLLECTION_TEXT = "arxiv_text"
COLLECTION_DOCS = "arxiv_docs"
ALL_COLLECTIONS = [COLLECTION_TEXT, COLLECTION_DOCS]

# Parent–child retrieval (arxiv_docs → arxiv_text)
PARENT_TOP_DOCS = int(os.getenv("PARENT_TOP_DOCS", "20"))
PARENT_CHUNKS_PER_DOC = int(os.getenv("PARENT_CHUNKS_PER_DOC", "10"))

# Max chunks per paper in final output (diversity control)
MAX_CHUNKS_PER_PAPER = int(os.getenv("MAX_CHUNKS_PER_PAPER", "4"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))

# Qdrant search ef (higher = better recall at query time)
QDRANT_SEARCH_EF = 200

# Intent-aware RRF fusion weights: (dense_weight, lexical_weight)
INTENT_RRF_WEIGHTS = {
    INTENT_EXPLANATORY: (0.7, 0.3),   # concept queries favor dense
    INTENT_COMPARATIVE: (0.6, 0.4),
    INTENT_TECHNICAL:   (0.5, 0.5),   # balanced for equations/formulas
    INTENT_EVIDENCE:    (0.55, 0.45), # evidence balanced
    INTENT_SOTA:        (0.7, 0.3),
    INTENT_DISCOVERY:   (0.4, 0.6),   # keyword-heavy queries favor FTS
}

# Intent-tuned retrieval pool sizes, MMR, and fusion merge breadth
INTENT_RETRIEVAL_PARAMS = {
    INTENT_EXPLANATORY: {
        "k_dense": 80,
        "k_lex": 60,
        "merge_top_m": 80,
        "rerank_top": 50,
        "mmr_lambda": 0.65,
        "mmr_enabled": True,
        "recency_calendar": False,
    },
    INTENT_COMPARATIVE: {
        "k_dense": 70,
        "k_lex": 70,
        "merge_top_m": 70,
        "rerank_top": 50,
        "mmr_lambda": 0.7,
        "mmr_enabled": True,
        "recency_calendar": False,
    },
    INTENT_TECHNICAL: {
        "k_dense": 80,
        "k_lex": 80,
        "merge_top_m": 60,
        "rerank_top": 60,
        "mmr_lambda": 0.8,
        "mmr_enabled": True,
        "recency_calendar": False,
    },
    INTENT_EVIDENCE: {
        "k_dense": 50,
        "k_lex": 50,
        "merge_top_m": 50,
        "rerank_top": 50,
        "mmr_lambda": 0.8,
        "mmr_enabled": True,
        "recency_calendar": False,
    },
    INTENT_SOTA: {
        "k_dense": 80,
        "k_lex": 60,
        "merge_top_m": 90,
        "rerank_top": 50,
        "mmr_lambda": 0.6,
        "mmr_enabled": True,
        "recency_calendar": True,
        "recency_boost": 1.25,
    },
    INTENT_DISCOVERY: {
        "k_dense": 60,
        "k_lex": 80,
        "merge_top_m": 150,
        "rerank_top": 50,
        "mmr_lambda": 0.5,
        "mmr_enabled": True,
        "recency_calendar": False,
    },
}

SECTION_BASE_WEIGHT = {
    "abstract": 1.15,
    "introduction": 1.12,
    "related_work": 0.94,
    "background": 0.9,
    "method": 1.18,
    "experiments": 1.12,
    "results": 1.12,
    "discussion": 1.05,
    "conclusion": 1.1,
    "appendix": 0.82,
    "other": 1.0,
}

INTENT_SECTION_PREF = {
    INTENT_EXPLANATORY: {"abstract": 1.35, "introduction": 1.28, "background": 1.12},
    INTENT_TECHNICAL: {"method": 1.32, "experiments": 1.18, "results": 1.12, "related_work": 0.5, "appendix": 0.5, "abstract": 0.5},
    INTENT_EVIDENCE: {"results": 1.35, "experiments": 1.30, "discussion": 1.15, "method": 1.05, "related_work": 0.5, "appendix": 0.5, "abstract": 0.5},
    INTENT_SOTA: {"results": 1.22, "related_work": 1.12, "conclusion": 1.1},
    INTENT_COMPARATIVE: {"method": 1.12, "results": 1.12, "discussion": 1.1},
}


def _bm25_tokenize(text: str) -> list[str]:
    """Fallback tokenizer for BM25 artifact refreshes.

    The persisted BM25 pickle does not always expose a callable tokenizer,
    so this mirrors the offline builder's simple normalization.
    """
    return re.sub(r"[^\w\s]", "", text.lower()).split()

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
        self.k_dense = int(os.getenv("K_DENSE", str(k_dense)))
        self.k_lex = int(os.getenv("K_LEX", str(k_lex)))
        self.merge_top_m = int(os.getenv("MERGE_TOP_M", str(merge_top_m)))
        self.final_top_n = int(os.getenv("FINAL_TOP_N", str(final_top_n)))
        self.rrf_k = int(os.getenv("RRF_K", str(rrf_k)))
        self.context_top_n = int(os.getenv("GENERATION_CONTEXT_TOP_N", str(GENERATION_CONTEXT_TOP_N)))

        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        embedding_model = resolve_embedding_model(embedding_model)
        reranker_model = reranker_model or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

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

        # Discover available collections
        self.collections = {}
        self._citation_adj_cache: Optional[dict[str, set[str]]] = None
        try:
            all_cols = self.qdrant_client.get_collections().collections
            available = {c.name for c in all_cols}
            for name in (COLLECTION_TEXT, COLLECTION_DOCS):
                if name in available:
                    info = self.qdrant_client.get_collection(name)
                    self.collections[name] = info.points_count
                    log.info("  %s: %s points", name, info.points_count)
                else:
                    log.warning("  %s: not found (will skip)", name)
        except Exception as e:
            log.error("Failed to list Qdrant collections: %s", e)

        # Reranker
        self.reranker = Reranker(model_name=reranker_model)

        # Artifacts
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        self.bm25 = None
        self.bm25_dirty = False
        self.chunks_meta = []
        self.chunks_text = {}
        self.chunks_contextual_text = {}
        self.papers_meta = {}
        
        try:
            with open(data_dir / "papers_meta.json", "r", encoding="utf-8") as f:
                raw_pm = json.load(f)
                self.papers_meta = {}
                for pid, rec in raw_pm.items():
                    if not isinstance(rec, dict):
                        continue
                    r2 = dict(rec)
                    pub = normalize_published(r2.get("published"))
                    if pub:
                        r2["published"] = pub
                    self.papers_meta[pid] = r2
            log.info(f"Loaded {len(self.papers_meta)} papers from papers_meta.json")
        except Exception as e:
            log.warning(f"Could not load papers_meta.json: {e}")
            
        try:
            with open(data_dir / "chunks_meta.jsonl", "r", encoding="utf-8") as f:
                self.chunks_meta = []
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if isinstance(o, dict) and "section_hint" in o:
                        o["section_hint"] = normalize_section_label(o.get("section_hint", "other"))
                    self.chunks_meta.append(o)
            log.info(f"Loaded {len(self.chunks_meta)} chunk metadata entries.")
        except Exception as e:
            log.warning(f"Could not load chunks_meta.jsonl: {e}")
            
        try:
            self.bm25 = joblib.load(data_dir / "bm25_v1.pkl")
            log.info("BM25 index loaded successfully.")
        except Exception as e:
            log.warning(f"Could not load bm25_v1.pkl: {e}")

        if os.getenv("LEXICAL_LOAD_CHUNKS_TEXT", "").lower() in ("1", "true", "yes"):
            try:
                path = data_dir / "chunks_text.jsonl"
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        o = json.loads(line)
                        cid = o.get("chunk_id")
                        if cid:
                            self.chunks_text[cid] = o.get("text", "")
                            self.chunks_contextual_text[cid] = o.get("contextual_text", o.get("text", ""))
                log.info("Loaded %s chunk text fallbacks from chunks_text.jsonl", len(self.chunks_text))
            except Exception as e:
                log.warning("Could not load chunks_text.jsonl: %s", e)
            
        log.info("HybridRetriever ready (Memory-Optimized: Text will be fetched from Qdrant).")

    def add_paper(self, paper_meta: dict, chunks: list[dict]):
        """Dynamically add a new paper to the in-memory retriever structures."""
        paper_id = paper_meta["paper_id"]
        
        # 1. Update papers meta
        self.papers_meta[paper_id] = {
            "title": paper_meta.get("title", ""),
            "authors": paper_meta.get("authors", ""),
            "published": normalize_published(paper_meta.get("published")) or "",
            "categories": paper_meta.get("categories", "")
        }
        
        # 2. Update chunks meta and text
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            self.chunks_meta.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title": chunk.get("title", ""),
                "authors": chunk.get("authors", ""),
                "categories": chunk.get("categories", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
                "section_hint": normalize_section_label(chunk.get("section_hint", "other")),
                "layer": chunk.get("layer", "core"),
                "token_count": chunk.get("token_count", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "chunk_source": chunk.get("chunk_source", "full_text")
            })
            
        # 3. BM25 incremental updates are not supported.
        if self.bm25:
            self.bm25_dirty = True
            log.warning("BM25 artifacts are now stale; rebuild BM25 to index new papers.")

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
                        "section_hint": normalize_section_label(chunk.get("section_hint", "other")),
                        "layer": chunk.get("layer", "core"),
                        "token_count": chunk.get("token_count", 0),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "total_chunks": chunk.get("total_chunks", 1),
                        "chunk_source": chunk.get("chunk_source", "full_text")
                    }
                    f.write(json.dumps(meta_entry) + "\n")
                    
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

    def _combine_candidate_lists(self, candidate_lists: list[list[dict]], score_key: str) -> list[dict]:
        """Merge repeated results from multiple query variants, keeping the best score."""
        merged = {}
        for candidates in candidate_lists:
            for candidate in candidates:
                cid = candidate["chunk_id"]
                existing = merged.get(cid)
                if existing is None or candidate.get(score_key, 0.0) > existing.get(score_key, 0.0):
                    merged[cid] = candidate
        return sorted(merged.values(), key=lambda x: x.get(score_key, 0.0), reverse=True)

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
                "retrieval_text": payload.get("contextual_text", payload.get("chunk_text", "")),
                "metadata": normalize_chunk_metadata({
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
                }),
                "dense_score": point.score,
                "source": "dense",
                "collection": collection_name,
            })
        return candidates

    def _dense_retrieve(
        self,
        query: str,
        qdrant_filter: Optional[Filter] = None,
        query_variants: Optional[list[str]] = None,
        auxiliary_dense_strings: Optional[list[str]] = None,
    ) -> list[dict]:
        lists: list[list[dict]] = []
        for extra in auxiliary_dense_strings or []:
            s = (extra or "").strip()
            if not s:
                continue
            lists.append(
                self._dense_retrieve_collection(
                    COLLECTION_TEXT, self._encode_query(s), self.k_dense, qdrant_filter
                )
            )
        variants = query_variants or [query]
        for variant in variants:
            query_emb = self._encode_query(variant)
            lists.append(
                self._dense_retrieve_collection(COLLECTION_TEXT, query_emb, self.k_dense, qdrant_filter)
            )
        merged = self._combine_candidate_lists(lists, "dense_score")
        return merged[: max(self.k_dense * 3, 1)]

    def _embedding_query_expansion(self, query: str, max_variants: int = 4) -> list[str]:
        """Lightweight semantic expansion using nearby chunk texts (no extra LLM)."""
        if COLLECTION_TEXT not in self.collections:
            return []
        try:
            emb = self._encode_query(query)
            res = self.qdrant_client.query_points(
                collection_name=COLLECTION_TEXT,
                query=emb,
                limit=8,
                with_payload=True,
            )
        except Exception as exc:
            log.debug("Embedding query expansion skipped: %s", exc)
            return []

        phrases: set[str] = set()
        phrase_re = re.compile(r"\b[A-Z][a-z]+(?:\s+[a-z][a-z]+){0,3}\b")
        for pt in res.points or []:
            text = (pt.payload or {}).get("chunk_text", "") or ""
            for m in phrase_re.findall(text[:1200]):
                if len(m) > 5 and m.lower() not in query.lower():
                    phrases.add(m.strip())
                if len(phrases) >= 12:
                    break
            if len(phrases) >= 12:
                break

        out: list[str] = []
        for p in list(phrases)[:max_variants]:
            out.append(f"{query} {p}")
        return out

    def _fetch_chunk_payloads_from_qdrant(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch Qdrant payloads for lexical candidates; fill missing via scroll."""
        out: dict[str, dict] = {}
        if not chunk_ids or COLLECTION_TEXT not in self.collections:
            return out

        batch_size = 48
        for start in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[start : start + batch_size]
            uuids = [chunk_id_to_uuid(cid) for cid in batch_ids]
            try:
                points = self.qdrant_client.retrieve(
                    collection_name=COLLECTION_TEXT,
                    ids=uuids,
                    with_payload=True,
                )
            except Exception as exc:
                log.warning("Qdrant retrieve batch failed: %s", exc)
                points = []

            for p in points or []:
                pl = p.payload or {}
                cid = pl.get("chunk_id")
                if cid:
                    out[cid] = pl

        missing = [cid for cid in chunk_ids if cid not in out]
        if not missing:
            return out

        try:
            flt = Filter(
                must=[FieldCondition(key="chunk_id", match=MatchAny(any=missing[:96]))]
            )
            scroll_res, _ = self.qdrant_client.scroll(
                collection_name=COLLECTION_TEXT,
                scroll_filter=flt,
                with_payload=True,
                limit=min(256, max(len(missing) * 2, 32)),
            )
            for p in scroll_res or []:
                pl = p.payload or {}
                cid = pl.get("chunk_id")
                if cid and cid not in out:
                    out[cid] = pl
        except Exception as exc:
            log.debug("Qdrant scroll fallback for chunk payloads failed: %s", exc)

        return out

    def _parent_child_enabled(self) -> bool:
        """Parent–child path uses arxiv_docs; ENABLE_PARENT_CHILD unset=auto (on if collection exists)."""
        if retrieval_skip_parent_child():
            return False
        mode = env_tri("ENABLE_PARENT_CHILD")
        has_docs = COLLECTION_DOCS in self.collections
        if mode == "off":
            return False
        if mode == "on":
            if not has_docs:
                log.warning("ENABLE_PARENT_CHILD=true but collection %s missing", COLLECTION_DOCS)
            return has_docs
        return has_docs

    def _chunk_candidate_from_point(self, point, dense_score: float, source_tag: str) -> dict:
        payload = point.payload or {}
        return {
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "chunk_text": payload.get("chunk_text", ""),
            "retrieval_text": payload.get("contextual_text", payload.get("chunk_text", "")),
            "metadata": normalize_chunk_metadata({
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
            }),
            "dense_score": float(dense_score),
            "source": source_tag,
            "collection": COLLECTION_TEXT,
        }

    def _parent_child_chunk_candidates(
        self,
        query_embedding: list[float],
        qdrant_filter: Optional[Filter],
        paper_scope: Optional[str],
    ) -> tuple[list[dict], dict]:
        """Retrieve top papers in arxiv_docs, then pull their best chunks from arxiv_text."""
        trace: dict = {"active": False, "top_docs": [], "chunks_added": 0}
        if not self._parent_child_enabled() or paper_scope:
            return [], trace

        trace["active"] = True
        try:
            doc_res = self.qdrant_client.query_points(
                collection_name=COLLECTION_DOCS,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=PARENT_TOP_DOCS,
                search_params=SearchParams(hnsw_ef=QDRANT_SEARCH_EF),
                with_payload=True,
            )
        except Exception as exc:
            log.warning("Parent doc retrieval failed: %s", exc)
            return [], trace

        docs = doc_res.points or []
        if not docs:
            return [], trace

        out: list[dict] = []
        for d in docs:
            pl = d.payload or {}
            pid = pl.get("paper_id", "")
            if not pid:
                continue
            trace["top_docs"].append({"paper_id": pid, "doc_score": round(float(d.score or 0.0), 5)})

            must = [FieldCondition(key="paper_id", match=MatchValue(value=pid))]
            if qdrant_filter and qdrant_filter.must:
                must.extend(qdrant_filter.must)
            paper_flt = Filter(must=must)

            try:
                ch_res = self.qdrant_client.query_points(
                    collection_name=COLLECTION_TEXT,
                    query=query_embedding,
                    query_filter=paper_flt,
                    limit=PARENT_CHUNKS_PER_DOC,
                    search_params=SearchParams(hnsw_ef=QDRANT_SEARCH_EF),
                    with_payload=True,
                )
            except Exception as exc:
                log.debug("Parent chunk expand failed for %s: %s", pid, exc)
                continue

            doc_score = float(d.score or 0.0)
            for pt in ch_res.points or []:
                ch = float(pt.score or 0.0)
                combined = max(1e-6, doc_score) * max(1e-6, ch)
                out.append(
                    self._chunk_candidate_from_point(
                        pt,
                        dense_score=combined,
                        source_tag="dense_parent",
                    )
                )

        trace["chunks_added"] = len(out)
        return out, trace

    def _load_citation_adjacency(self) -> dict[str, set[str]]:
        if not env_bool("ENABLE_CITATION_BOOST", False):
            return {}
        if self._citation_adj_cache is not None:
            return self._citation_adj_cache
        self._citation_adj_cache = {}
        try:
            from db.database import get_db

            db = get_db()
            db.run_migrations()
            with db.conn.cursor() as cur:
                cur.execute(
                    "SELECT source_paper_id, target_paper_id FROM citation_edges LIMIT 500000"
                )
                for row in cur.fetchall():
                    a, b = row["source_paper_id"], row["target_paper_id"]
                    self._citation_adj_cache.setdefault(a, set()).add(b)
                    self._citation_adj_cache.setdefault(b, set()).add(a)
            log.info("Loaded citation adjacency: %s nodes", len(self._citation_adj_cache))
        except Exception as exc:
            log.warning("Could not load citation_edges: %s", exc)
            self._citation_adj_cache = {}
        return self._citation_adj_cache

    def _apply_citation_graph_boost(self, candidates: list[dict], trace: dict, intent: str = INTENT_DISCOVERY) -> list[dict]:
        if not env_bool("ENABLE_CITATION_BOOST", False) or not candidates:
            trace["citation_boost"] = {"enabled": False}
            return candidates
        if intent in (INTENT_TECHNICAL, INTENT_EVIDENCE, INTENT_EXPLANATORY):
            trace["citation_boost"] = {"enabled": False, "reason": "bypassed_for_intent"}
            return candidates

        adj = self._load_citation_adjacency()
        if not adj:
            trace["citation_boost"] = {"enabled": True, "applied": False, "reason": "empty_graph"}
            return candidates

        top = sorted(candidates, key=lambda x: x.get("fusion_score", 0.0), reverse=True)[:25]
        seeds: set[str] = set()
        for c in top:
            pid = c.get("metadata", {}).get("paper_id", "")
            if pid:
                seeds.add(pid)
        neighbors: set[str] = set()
        for s in seeds:
            neighbors |= adj.get(s, set())

        boosted = 0
        for c in candidates:
            pid = c.get("metadata", {}).get("paper_id", "")
            if pid and pid in neighbors and pid not in seeds:
                c["fusion_score"] = float(c.get("fusion_score", 0.0)) + 0.004
                boosted += 1
        candidates.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        trace["citation_boost"] = {
            "enabled": True,
            "boosted_candidates": boosted,
            "seed_papers": len(seeds),
        }
        return candidates

    # ------------------------------------------------------------------
    # Lexical retrieval (PostgreSQL FTS)
    # ------------------------------------------------------------------

    def _lexical_retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        author: Optional[str] = None,
        start_year: Optional[int] = None,
        query_variants: Optional[list[str]] = None,
    ) -> list[dict]:
        if not self.bm25 or not self.chunks_meta:
            return []

        variants = query_variants or [query]
        score_accumulator = np.zeros(len(self.chunks_meta), dtype=float)
        for variant in variants:
            tokens = re.sub(r'[^\w\s]', '', variant.lower()).split()
            if not tokens:
                continue
            score_accumulator = np.maximum(score_accumulator, self.bm25.get_scores(tokens))

        k = min(self.k_lex * 5, len(score_accumulator))
        if k == 0:
            return []
        top_indices = np.argpartition(score_accumulator, -k)[-k:]
        top_indices = top_indices[np.argsort(score_accumulator[top_indices])[::-1]]

        candidates = []
        for idx in top_indices:
            if score_accumulator[idx] <= 0:
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
                "chunk_text": "", # Will be fetched from Qdrant below
                "retrieval_text": "",
                "lex_score": float(score_accumulator[idx]),
                "chunk_type": meta.get("chunk_type", "text"),
                "metadata": normalize_chunk_metadata({
                    "paper_id": paper_id,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "categories": meta.get("categories", ""),
                    "section_hint": meta.get("section_hint", "other"),
                    "layer": meta.get("layer", "core"),
                    "chunk_source": meta.get("chunk_source", "full_text"),
                }),
                "source": "lexical",
            })
            if len(candidates) >= self.k_lex * 3:
                break

        if candidates:
            payload_map = self._fetch_chunk_payloads_from_qdrant([c["chunk_id"] for c in candidates])
            for c in candidates:
                payload = payload_map.get(c["chunk_id"])
                if payload:
                    c["chunk_text"] = payload.get("chunk_text", "")
                    c["retrieval_text"] = payload.get("contextual_text", c["chunk_text"])
                elif c["chunk_id"] in self.chunks_text:
                    c["chunk_text"] = self.chunks_text.get(c["chunk_id"], "")
                    c["retrieval_text"] = self.chunks_contextual_text.get(
                        c["chunk_id"], c["chunk_text"]
                    )

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
                "retrieval_text": c.get("retrieval_text", c.get("chunk_text", "")),
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
                if not merged[cid].get("retrieval_text"):
                    merged[cid]["retrieval_text"] = c.get("retrieval_text", c.get("chunk_text", ""))
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "chunk_text": c.get("chunk_text", ""),
                    "retrieval_text": c.get("retrieval_text", c.get("chunk_text", "")),
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

    def _boost_document_summary_sections(self, query: str, candidates: list[dict]) -> list[dict]:
        """Prefer abstract/introduction/conclusion chunks for paper-scoped summary queries."""
        if not candidates or not is_document_summary_query(query):
            return candidates

        boosted = []
        for candidate in candidates:
            meta = candidate.get("metadata", {})
            section = (meta.get("section_hint", "other") or "other").lower()
            boost = 0.0
            if section == "abstract":
                boost = 0.030
            elif section == "introduction":
                boost = 0.025
            elif section == "conclusion":
                boost = 0.020
            elif section in {"background", "method", "results"}:
                boost = 0.010

            candidate["fusion_score"] = candidate.get("fusion_score", 0.0) + boost
            boosted.append(candidate)

        boosted.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        return boosted

    def _diversify_candidates(self, candidates: list[dict], target_n: int, paper_scoped: bool = False) -> list[dict]:
        """Greedy diversity filter so final context covers multiple sections/evidence types."""
        if len(candidates) <= target_n:
            return candidates

        selected = []
        seen_sections = set()
        paper_counts = {}

        for candidate in candidates:
            meta = candidate.get("metadata", {})
            section = meta.get("section_hint", "other")
            paper_id = meta.get("paper_id", "")
            current_count = paper_counts.get(paper_id, 0)
            paper_limit = 6 if paper_scoped else MAX_CHUNKS_PER_PAPER

            if current_count >= paper_limit:
                continue

            if section not in seen_sections or len(selected) < min(target_n, 6):
                selected.append(candidate)
                seen_sections.add(section)
                if paper_id:
                    paper_counts[paper_id] = current_count + 1
            if len(selected) >= target_n:
                return selected

        for candidate in candidates:
            if candidate["chunk_id"] in {x["chunk_id"] for x in selected}:
                continue
            meta = candidate.get("metadata", {})
            paper_id = meta.get("paper_id", "")
            current_count = paper_counts.get(paper_id, 0)
            paper_limit = 6 if paper_scoped else MAX_CHUNKS_PER_PAPER
            if current_count >= paper_limit:
                continue
            selected.append(candidate)
            if paper_id:
                paper_counts[paper_id] = current_count + 1
            if len(selected) >= target_n:
                break
        return selected

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

    def _candidate_vectors_for_mmr(self, candidates: list[dict]) -> list[tuple[dict, np.ndarray]]:
        """Return (candidate, normalized vector) pairs for chunks that have Qdrant vectors."""
        if COLLECTION_TEXT not in self.collections or not candidates:
            return []
        out: list[tuple[dict, np.ndarray]] = []
        batch_size = 32
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            uuids = [chunk_id_to_uuid(c["chunk_id"]) for c in batch]
            try:
                pts = self.qdrant_client.retrieve(
                    collection_name=COLLECTION_TEXT,
                    ids=uuids,
                    with_vectors=True,
                    with_payload=True,
                )
            except Exception:
                pts = []
            pmap = {p.payload.get("chunk_id"): p for p in pts or [] if p.payload}
            for c in batch:
                pt = pmap.get(c["chunk_id"])
                if pt and pt.vector is not None:
                    v = np.asarray(pt.vector, dtype=np.float32)
                    n = float(np.linalg.norm(v)) + 1e-9
                    out.append((c, v / n))
        return out

    def _apply_mmr(
        self,
        candidates: list[dict],
        *,
        lambda_param: float,
        top_k: int,
    ) -> list[dict]:
        """Maximal marginal relevance on top reranked candidates using Qdrant vectors."""
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        ordered = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True,
        )
        pairs = self._candidate_vectors_for_mmr(ordered[: min(len(ordered), top_k * 4)])
        if len(pairs) < 2:
            return ordered[:top_k]

        pairs.sort(key=lambda x: x[0].get("rerank_score", 0.0), reverse=True)
        cands = [p[0] for p in pairs]
        emb = np.stack([p[1] for p in pairs], axis=0)
        rel = np.array([float(c.get("rerank_score", 0.5)) for c in cands], dtype=np.float32)
        if rel.max() > rel.min():
            rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        else:
            rel = np.ones_like(rel)

        selected = [0]
        selected_set = {0}
        while len(selected) < top_k:
            best_j = -1
            best_mmr = -1e9
            for j in range(len(cands)):
                if j in selected_set:
                    continue
                sims = emb[j] @ emb[selected].T
                max_sim = float(np.max(sims))
                mmr = float(lambda_param * rel[j] - (1.0 - lambda_param) * max_sim)
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_j = j
            if best_j < 0:
                break
            selected.append(best_j)
            selected_set.add(best_j)

        picked = [cands[j] for j in selected[:top_k]]
        if len(picked) < top_k:
            for c in ordered:
                if c["chunk_id"] in {x["chunk_id"] for x in picked}:
                    continue
                picked.append(c)
                if len(picked) >= top_k:
                    break
        return picked[:top_k]

    def _apply_section_rerank_boost(self, candidates: list[dict], intent: str) -> list[dict]:
        pref = INTENT_SECTION_PREF.get(intent, {})
        for c in candidates:
            section = (c.get("metadata", {}).get("section_hint", "other") or "other").lower()
            base = SECTION_BASE_WEIGHT.get(section, 1.0)
            extra = pref.get(section, 1.0)
            boost = base * extra
            c["rerank_score"] = float(c.get("rerank_score", 0.0)) * boost
            c["section_boost"] = boost
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates

    def _apply_recency_calendar_boost(
        self,
        candidates: list[dict],
        *,
        boost: float,
        months: int = 18,
    ) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=months * 30)

        for c in candidates:
            pid = c.get("metadata", {}).get("paper_id", "")
            pub_raw = (self.papers_meta.get(pid) or {}).get("published")
            if not pub_raw:
                continue
            try:
                y = int(str(pub_raw)[:4])
                m = int(str(pub_raw)[5:7]) if len(str(pub_raw)) >= 7 else 1
                d = int(str(pub_raw)[8:10]) if len(str(pub_raw)) >= 10 else 1
                pub_dt = datetime(y, m, d)
            except Exception:
                continue
            if pub_dt >= cutoff:
                c["rerank_score"] = float(c.get("rerank_score", 0.0)) * boost
                c["recency_boosted"] = True
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    def compress_context(self, query: str, passages: list[dict],
                         max_sentences: int = 25, intent: str = INTENT_DISCOVERY) -> str:
        parts = []
        for i, p in enumerate(passages, 1):
            title = p.get("title", p.get("metadata", {}).get("title", ""))
            text = p.get("chunk_text", "")
            section = p.get("section_hint", p.get("metadata", {}).get("section_hint", "other"))
            if text:
                header_bits = [f"Source {i}"]
                if title:
                    header_bits.append(title)
                if section:
                    header_bits.append(f"section={section}")
                header = " | ".join(header_bits)
                parts.append(f"[{header}]\n{text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_n: int = None,
        category: Optional[str] = None,
        author: Optional[str] = None,
        start_year: Optional[int] = None,
        intent: Optional[str] = None,
        paper_id: Optional[str] = None,
        dense_auxiliary_text: Optional[str] = None,
    ) -> dict:
        """Main retrieval pipeline.

        Args:
            paper_id: If set, restricts retrieval to chunks from this paper only
                      (used by document-scoped chat).
            dense_auxiliary_text: Optional HyDE-style passage; dense retrieval also encodes this string.
        """
        top_n = top_n or self.final_top_n
        intent = intent or classify_query_intent(query)
        is_paper_scoped = paper_id is not None
        trace = {"intent": intent, "paper_scoped": is_paper_scoped}
        t0 = time.time()

        params = INTENT_RETRIEVAL_PARAMS.get(intent, INTENT_RETRIEVAL_PARAMS[INTENT_DISCOVERY])
        trace["retrieval_params"] = {k: v for k, v in params.items() if k != "mmr_lambda"}

        saved_dense, saved_lex, saved_merge = self.k_dense, self.k_lex, self.merge_top_m
        self.k_dense = int(params["k_dense"])
        self.k_lex = int(params["k_lex"])
        self.merge_top_m = max(int(params["merge_top_m"]), saved_merge)

        try:
            gate = query_expansion_gate(query, intent)
            trace["expansion_gate"] = gate
            trace["ablations"] = {
                "skip_dense": retrieval_skip_dense(),
                "skip_lexical": retrieval_skip_lexical(),
                "skip_parent_child": retrieval_skip_parent_child(),
                "skip_rerank": retrieval_skip_rerank(),
                "skip_mmr": retrieval_skip_mmr(),
                "skip_boosts": retrieval_skip_boosts(),
            }

            if gate["restrict_decompose"]:
                qs = " ".join((query or "").strip().split())
                query_variants = [qs] if qs else []
            else:
                query_variants = decompose_query(query, intent=intent, paper_scoped=is_paper_scoped)

            if intent in (INTENT_COMPARATIVE, INTENT_TECHNICAL, INTENT_SOTA) and not gate["restrict_embedding_expansion"]:
                extra = self._embedding_query_expansion(query)
                merged_v = []
                seen = set()
                for v in query_variants + extra:
                    key = v.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        merged_v.append(v)
                    if len(merged_v) >= 8:
                        break
                query_variants = merged_v
            trace["query_variants"] = query_variants

            if not gate["restrict_llm_expansion"]:
                try:
                    from api.app import expand_query_variants_llm

                    llm_vars = expand_query_variants_llm(query, max_variants=3)
                    if llm_vars:
                        seen = {v.lower().strip() for v in query_variants}
                        for v in llm_vars:
                            k = v.lower().strip()
                            if k not in seen:
                                seen.add(k)
                                query_variants.append(v)
                            if len(query_variants) >= 8:
                                break
                        trace["query_variants"] = query_variants
                except Exception as exc:
                    log.debug("LLM query expansion skipped: %s", exc)

            is_explanatory = intent == INTENT_EXPLANATORY
            qdrant_filter = self._build_qdrant_filter(
                category=category, author=author, paper_id=paper_id
            )

            aux = [dense_auxiliary_text] if (dense_auxiliary_text or "").strip() else None

            t1 = time.time()
            dense_candidates = []
            if not retrieval_skip_dense():
                dense_candidates = self._dense_retrieve(
                    query, qdrant_filter, query_variants=query_variants, auxiliary_dense_strings=aux
                )
            q_emb = self._encode_query(query)
            pc_chunks, pc_trace = [], {"active": False, "skipped": retrieval_skip_parent_child()}
            if not retrieval_skip_parent_child():
                pc_chunks, pc_trace = self._parent_child_chunk_candidates(
                    q_emb, qdrant_filter, paper_id
                )
            trace["parent_child"] = pc_trace
            if pc_chunks:
                if retrieval_skip_dense() and not dense_candidates:
                    dense_candidates = pc_chunks[: max(self.k_dense * 5, 200)]
                elif dense_candidates:
                    dense_candidates = self._combine_candidate_lists(
                        [dense_candidates, pc_chunks],
                        "dense_score",
                    )[: max(self.k_dense * 5, 200)]
            trace["dense_ms"] = round((time.time() - t1) * 1000, 1)
            trace["dense_count"] = len(dense_candidates)

            t2 = time.time()
            if retrieval_skip_lexical():
                lex_candidates = []
            else:
                lex_candidates = self._lexical_retrieve(
                    query,
                    category=category,
                    author=author,
                    start_year=start_year,
                    query_variants=query_variants,
                )
            if paper_id and lex_candidates:
                lex_candidates = [
                    c for c in lex_candidates
                    if c.get("metadata", {}).get("paper_id") == paper_id
                ]
            trace["lex_ms"] = round((time.time() - t2) * 1000, 1)
            trace["lex_count"] = len(lex_candidates)

            merge_floor = max(int(params["merge_top_m"]), 120)
            self.merge_top_m = max(self.merge_top_m, merge_floor)
            if is_explanatory:
                self.merge_top_m = max(self.merge_top_m, 160)

            t3 = time.time()
            merged = self._merge_and_normalize(dense_candidates, lex_candidates, intent)
            merged = self._apply_citation_graph_boost(merged, trace, intent)
            if is_paper_scoped:
                merged = self._boost_document_summary_sections(query, merged)
            trace["merge_ms"] = round((time.time() - t3) * 1000, 1)
            trace["merged_count"] = len(merged)

            if not is_paper_scoped:
                if not is_explanatory:
                    merged = self._apply_recency_boost(merged)
                    merged.sort(key=lambda x: x["fusion_score"], reverse=True)

                merged = self._filter_candidates_by_year(merged, start_year)

                if is_explanatory:
                    merged = self._semantic_pruning(query, merged)

                merged = self._enforce_paper_diversity(merged, max_per_paper=MAX_CHUNKS_PER_PAPER)

            trace["diverse_count"] = len(merged)

            rerank_mode = "combined" if is_explanatory else "default"
            if intent == INTENT_TECHNICAL:
                rerank_mode = "default"
            target_n = top_n
            rerank_cap = min(len(merged), max(target_n * 2, int(params["rerank_top"]) + 8))

            t4 = time.time()
            if retrieval_skip_rerank():
                pool = sorted(merged, key=lambda x: x.get("fusion_score", 0.0), reverse=True)[:rerank_cap]
                reranked = []
                for row in pool:
                    row = dict(row)
                    row["rerank_score"] = float(row.get("fusion_score", 0.0))
                    reranked.append(row)
                trace["rerank"] = {"skipped": True, "reason": "RETRIEVAL_SKIP_RERANK"}
            else:
                reranked = self.reranker.rerank(
                    query, merged, top_n=rerank_cap, rerank_text_mode=rerank_mode
                )
                trace["rerank"] = dict(getattr(self.reranker, "last_trace", None) or {})
            trace["rerank_ms"] = round((time.time() - t4) * 1000, 1)

            if not retrieval_skip_boosts():
                reranked = self._apply_section_rerank_boost(reranked, intent)
                if params.get("recency_calendar") and intent == INTENT_SOTA:
                    reranked = self._apply_recency_calendar_boost(
                        reranked,
                        boost=float(params.get("recency_boost", 1.2)),
                    )
            else:
                trace["boosts"] = {"skipped": True}
                for c in reranked:
                    c["rerank_score"] = float(c.get("rerank_score", c.get("fusion_score", 0.0)))

            use_mmr = (
                env_bool("ENABLE_MMR", True)
                and params.get("mmr_enabled", True)
                and not retrieval_skip_mmr()
            )
            trace["mmr"] = {"enabled": bool(use_mmr)}
            if use_mmr:
                reranked = self._apply_mmr(
                    reranked,
                    lambda_param=float(params["mmr_lambda"]),
                    top_k=target_n,
                )
            else:
                reranked = reranked[:target_n]

            reranked = self._diversify_candidates(reranked, target_n=target_n, paper_scoped=is_paper_scoped)

            if is_explanatory:
                reranked = self._ensure_layer_coverage(reranked, merged)

            # Calculate robust relevance_score (Sigmoid confidence)
            if reranked:
                # 1. Sigmoid transformation for probability-like scaling
                # BGE-Reranker-v2-m3 scores around 1.0 are good, 5.0+ are very strong
                # We use a slight shift to make 0.0 (logit) look like ~70%
                for p in reranked:
                    logit = float(p.get("rerank_score", 0.0))
                    # Confidence probability (0 to 1)
                    p["relevance_score"] = 1.0 / (1.0 + np.exp(-(logit + 1.5) / 2.0))
                
                # 2. Final relative normalization so top is always strong
                max_rel = max(p["relevance_score"] for p in reranked)
                if max_rel > 0:
                    for p in reranked:
                        p["relevance_score"] = min(1.0, p["relevance_score"] / max_rel)

            analytics = self.extract_analytics(merged)

            passages = []
            for p in reranked:
                meta = normalize_chunk_metadata(p.get("metadata", {}))
                passages.append({
                    "chunk_id": p["chunk_id"],
                    "paper_id": meta.get("paper_id", ""),
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "categories": meta.get("categories", ""),
                    "chunk_text": p["chunk_text"],
                    "contextual_text": p.get("retrieval_text", p["chunk_text"]),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "modality": meta.get("modality", "text"),
                    "section_hint": meta.get("section_hint", "other"),
                    "layer": meta.get("layer", "core"),
                    "rerank_score": p.get("rerank_score", 0.0),
                    "fusion_score": p.get("fusion_score", 0.0),
                    "relevance_score": p.get("relevance_score", 0.0),
                    "sources": p.get("sources", []),
                })

            trace["total_ms"] = round((time.time() - t0) * 1000, 1)
            trace["filters"] = {"category": category, "author": author, "start_year": start_year}
            trace["unique_papers"] = len({p["paper_id"] for p in passages if p["paper_id"]})

            return {"passages": passages, "trace": trace, "analytics": analytics}
        finally:
            self.k_dense, self.k_lex, self.merge_top_m = saved_dense, saved_lex, saved_merge

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
            log.info("Searching for papers similar to: %s", paper_id)

            query_vec: Optional[list[float]] = None
            if COLLECTION_DOCS in self.collections:
                try:
                    doc_pt = self.qdrant_client.retrieve(
                        collection_name=COLLECTION_DOCS,
                        ids=[paper_id_to_uuid(paper_id)],
                        with_vectors=True,
                        with_payload=True,
                    )
                    if doc_pt and doc_pt[0].vector is not None:
                        query_vec = list(doc_pt[0].vector)
                except Exception as exc:
                    log.debug("arxiv_docs lookup failed, falling back to chunk mean: %s", exc)

            if query_vec is None:
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

                if not points:
                    log.info("MatchValue for paper_id %s returned no points. Retrying with MatchText...", paper_id)
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
                    log.warning("No points found for paper_id %s in Qdrant.", paper_id)
                    return []

                embeddings = [p.vector for p in points if p.vector is not None]
                if not embeddings:
                    log.warning("Points found for paper_id %s but none had vectors.", paper_id)
                    return []

                arr = np.array(embeddings)
                mean_emb = np.mean(arr, axis=0)
                query_vec = (mean_emb / (np.linalg.norm(mean_emb) + 1e-9)).tolist()

            res = self.qdrant_client.query_points(
                collection_name=COLLECTION_TEXT,
                query=query_vec,
                query_filter=Filter(
                    must_not=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                limit=100,
                with_payload=True,
            )
            results = res.points or []
            log.info("Qdrant returned %s candidate points.", len(results))

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
                        "similarity_score": round(float(point.score or 0.0), 4),
                        "chunk_text": (payload.get("chunk_text", "") or "")[:300],
                    })
                    if len(papers) >= top_n:
                        break

            log.info("Found %s unique similar papers.", len(papers))
            return papers
        except Exception as e:
            log.error("Similar papers search failed: %s", e)
            return []
