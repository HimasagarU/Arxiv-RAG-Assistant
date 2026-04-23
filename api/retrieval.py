"""
retrieval.py — Hybrid dense + BM25 retrieval pipeline with cross-encoder reranking.

This module implements the full retrieval pipeline:
1. Dense retrieval via Chroma (sentence-transformer embeddings)
2. Lexical retrieval via BM25
3. Reciprocal Rank Fusion (RRF) score fusion
4. Cross-encoder reranking
5. Return top-N passages with full trace

Enhanced with:
- Advanced metadata filtering (category, author, year)
- Temporal/recency boosting
- Similar papers retrieval
- Context compression via MMR
"""

import logging
import math
import os
import pickle
import re
import sqlite3
import time
import urllib.request
import feedparser
from datetime import datetime
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding

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


# ---------------------------------------------------------------------------
# Query Intent Classification
# ---------------------------------------------------------------------------

# Intent types
INTENT_EXPLANATORY = "explanatory"   # "what is X", "how does X work"
INTENT_SOTA = "sota"                 # "latest X", "state of the art"
INTENT_COMPARATIVE = "comparative"   # "compare X and Y", "X vs Y"
INTENT_TECHNICAL = "technical"       # "derive X", "formal definition"
INTENT_DISCOVERY = "discovery"       # everything else

_INTENT_RULES = [
    # (regex_pattern, intent)
    (re.compile(r"\b(what\s+is|what\s+are|how\s+does|how\s+do|explain|define|describe|overview\s+of|introduction\s+to|basics\s+of|concept\s+of|meaning\s+of|tell\s+me\s+about)\b", re.I), INTENT_EXPLANATORY),
    (re.compile(r"\b(compare|vs\.?|versus|difference\s+between|compared\s+to|similarities|pros\s+and\s+cons|advantages\s+over|trade.?offs?)\b", re.I), INTENT_COMPARATIVE),
    (re.compile(r"\b(latest|newest|recent|state.of.the.art|sota|cutting.edge|current\s+trends?|advances?\s+in|progress\s+in|2024|2025|2026)\b", re.I), INTENT_SOTA),
    (re.compile(r"\b(derive|proof|prove|formal\s+definition|theorem|lemma|mathematical|equation\s+for|algorithm\s+for|pseudocode)\b", re.I), INTENT_TECHNICAL),
]


def classify_query_intent(query: str) -> str:
    """
    Classify query intent using keyword/regex rules.
    Returns one of: explanatory, sota, comparative, technical, discovery.
    """
    query_stripped = query.strip()

    for pattern, intent in _INTENT_RULES:
        if pattern.search(query_stripped):
            return intent

    # Heuristic: short queries ending with '?' are often explanatory
    if query_stripped.endswith('?') and len(query_stripped.split()) <= 8:
        return INTENT_EXPLANATORY

    return INTENT_DISCOVERY


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean-like environment flags with a safe default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_container_memory_limit_mb() -> Optional[int]:
    """Best-effort container memory limit detection from cgroup files."""
    cgroup_paths = [
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ]
    for path in cgroup_paths:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw or raw == "max":
                continue
            limit_bytes = int(raw)
            # Ignore unrealistic huge values often used as "no limit" sentinels.
            if limit_bytes <= 0 or limit_bytes > (1 << 60):
                continue
            return int(limit_bytes / (1024 * 1024))
        except Exception:
            continue
    return None


class HybridRetriever:
    """
    Hybrid retrieval: dense (Chroma) + lexical (BM25) + cross-encoder reranking.

    Enhanced with metadata filtering, recency boosting, similar papers, and analytics.
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
        rrf_k: int = 60,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        self.k_dense = k_dense
        self.k_lex = k_lex
        self.merge_top_m = merge_top_m
        self.final_top_n = final_top_n
        self.rrf_k = int(os.getenv("RRF_K", str(rrf_k)))

        # Keep backward compatibility with older callers passing alpha/beta.
        if alpha is not None or beta is not None:
            log.warning("alpha/beta fusion params are deprecated; using RRF fusion instead.")

        # Resolve paths from env
        chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "data/chroma_db")
        bm25_path = bm25_path or os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")
        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        reranker_model = reranker_model or os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-6-v2")
        memory_limit_mb = _get_container_memory_limit_mb()

        reranker_env_override = os.getenv("ENABLE_RERANKER")
        if reranker_env_override is not None:
            self.enable_reranker = _env_flag("ENABLE_RERANKER", True)
        else:
            min_memory_for_reranker_mb = int(os.getenv("RERANKER_MIN_MEMORY_MB", "768"))
            self.enable_reranker = not (
                memory_limit_mb is not None and memory_limit_mb <= min_memory_for_reranker_mb
            )
            if not self.enable_reranker:
                log.info(
                    f"Disabling reranker due to low memory limit ({memory_limit_mb}MB <= {min_memory_for_reranker_mb}MB)."
                )

        bm25_env_override = os.getenv("ENABLE_BM25")
        if bm25_env_override is not None:
            self.enable_bm25 = _env_flag("ENABLE_BM25", True)
        else:
            min_memory_for_bm25_mb = int(os.getenv("BM25_MIN_MEMORY_MB", "640"))
            self.enable_bm25 = not (
                memory_limit_mb is not None and memory_limit_mb <= min_memory_for_bm25_mb
            )
            if not self.enable_bm25:
                log.info(
                    f"Disabling BM25 due to low memory limit ({memory_limit_mb}MB <= {min_memory_for_bm25_mb}MB)."
                )

        self.reranker_lazy_load = _env_flag("RERANKER_LAZY_LOAD", True)
        self.db_path = os.getenv("DB_PATH", "data/arxiv_papers.db")

        log.info(f"Initializing HybridRetriever (ONNX mode, no PyTorch)")

        # Load embedding model via fastembed (ONNX-based, lightweight)
        log.info(f"Loading embedding model: {embedding_model}")
        self.embed_model = TextEmbedding(model_name=f"sentence-transformers/{embedding_model}")

        # Load Chroma
        log.info(f"Loading Chroma collection from: {chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_collection("arxiv_chunks")
        log.info(f"Chroma collection has {self.collection.count()} documents")

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_chunk_ids: list[str] = []
        if self.enable_bm25:
            log.info(f"Loading BM25 index from: {bm25_path}")
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.bm25_chunk_ids = bm25_data["chunk_ids"]
        else:
            log.info("BM25 retrieval disabled by configuration.")

        # Configure reranker (FlashRank ONNX-based).
        self.reranker = Reranker(
            model_name=reranker_model,
            lazy_load=self.reranker_lazy_load,
            enabled=self.enable_reranker,
        )

        log.info("HybridRetriever ready.")

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _build_chroma_where(self, category: Optional[str] = None,
                            author: Optional[str] = None) -> Optional[dict]:
        """Build a Chroma 'where' filter from user-supplied metadata filters."""
        conditions = []
        if category:
            conditions.append({"categories": {"$contains": category.strip()}})
        if author:
            conditions.append({"authors": {"$contains": author.strip()}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _semantic_pruning(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Prune candidates that don't look like explanatory/technical content.
        Uses section hints and high-signal 'method' keywords.
        """
        if not candidates:
            return []

        high_signal_keywords = {
            "we propose", "method", "pipeline", "architecture", "algorithm",
            "training", "approach", "framework", "optimization", "procedure",
            "step 1", "first,", "second,", "finally", "evaluation", "results"
        }

        pruned = []
        for c in candidates:
            text_lower = c.get("chunk_text", "").lower()
            metadata = c.get("metadata", {})
            section_hint = metadata.get("section_hint", "other")

            # Score the chunk's 'explanatory signal'
            signal_score = 0
            
            # Boost if it came from a high-signal section
            if section_hint in ["method", "abstract", "intro"]:
                signal_score += 2
            
            # Boost if it contains explanatory structural keywords
            if any(kw in text_lower for kw in high_signal_keywords):
                signal_score += 3
            
            # Always keep foundational ArXiv API results
            if "arxiv_api" in c.get("chunk_id", ""):
                signal_score += 5

            # If it has some signal or is in the absolute top 3, keep it
            if signal_score >= 2:
                pruned.append(c)
            elif len(pruned) < 3: # Always keep at least 3 for reranking
                pruned.append(c)

        return pruned

    def _filter_by_year(self, candidates: list[dict], start_year: Optional[int] = None) -> list[dict]:
        """Post-filter candidates by publication year using SQLite metadata."""
        if not start_year:
            return candidates
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            filtered = []
            for c in candidates:
                paper_id = c.get("metadata", {}).get("paper_id", "")
                if not paper_id:
                    # Try to extract from chunk_id
                    paper_id = c.get("chunk_id", "").rsplit("_chunk_", 1)[0]
                
                row = conn.execute(
                    "SELECT published FROM papers WHERE paper_id = ?", (paper_id,)
                ).fetchone()
                
                if row and row["published"]:
                    try:
                        pub_year = int(row["published"][:4])
                        if pub_year >= start_year:
                            filtered.append(c)
                    except (ValueError, IndexError):
                        filtered.append(c)  # Keep if we can't parse
                else:
                    filtered.append(c)  # Keep if no data
            conn.close()
            return filtered
        except Exception as e:
            log.warning(f"Year filtering failed: {e}")
            return candidates

    # ------------------------------------------------------------------
    # Recency boosting
    # ------------------------------------------------------------------

    def _apply_recency_boost(self, candidates: list[dict], boost_weight: float = 0.2) -> list[dict]:
        """
        Apply a small recency boost to fusion scores.
        Newer papers get a slight advantage. Uses exponential decay
        with a half-life of ~2 years.
        Scaled relative to RRF magnitude to avoid overwhelming rank fusion.
        """
        now = datetime.utcnow()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            for c in candidates:
                paper_id = c.get("metadata", {}).get("paper_id", "")
                if not paper_id:
                    paper_id = c.get("chunk_id", "").rsplit("_chunk_", 1)[0]
                
                row = conn.execute(
                    "SELECT published FROM papers WHERE paper_id = ?", (paper_id,)
                ).fetchone()
                
                if row and row["published"]:
                    try:
                        pub_date = datetime.fromisoformat(row["published"].replace("Z", "+00:00"))
                        age_years = (now - pub_date.replace(tzinfo=None)).days / 365.25
                        # Exponential decay: half-life of 2 years
                        recency_factor = math.exp(-0.347 * age_years)  # ln(2)/2 ≈ 0.347
                        c["recency_boost"] = (recency_factor * boost_weight) / max(float(self.rrf_k), 1.0)
                        c["fusion_score"] = c.get("fusion_score", 0) + c["recency_boost"]
                    except (ValueError, TypeError):
                        c["recency_boost"] = 0.0
                else:
                    c["recency_boost"] = 0.0
            
            conn.close()
        except Exception as e:
            log.warning(f"Recency boost failed: {e}")
        
        return candidates

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def _dense_retrieve(self, query: str, where_filter: Optional[dict] = None) -> list[dict]:
        """Dense retrieval via Chroma with optional metadata filtering."""
        query_embedding = list(self.embed_model.embed([query]))[0].tolist()

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": self.k_dense,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            log.warning(f"Chroma query with filter failed: {e}. Retrying without filter.")
            query_params.pop("where", None)
            results = self.collection.query(**query_params)

        candidates = []
        for i in range(len(results["ids"][0])):
            candidates.append({
                "chunk_id": results["ids"][0][i],
                "chunk_text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "dense_score": 1.0 - results["distances"][0][i],
                "source": "dense",
            })
        return candidates

    def _fetch_chunk_records(
        self,
        chunk_ids: list[str],
        include_documents: bool = False,
        include_metadatas: bool = True,
    ) -> dict[str, dict]:
        """Batch-fetch chunk records from Chroma and return an ID-indexed map."""
        if not chunk_ids:
            return {}

        unique_ids = list(dict.fromkeys(chunk_ids))
        include = []
        if include_documents:
            include.append("documents")
        if include_metadatas:
            include.append("metadatas")
        if not include:
            return {}

        try:
            fetched = self.collection.get(ids=unique_ids, include=include)
        except Exception as e:
            log.warning(f"Failed to fetch chunk records from Chroma: {e}")
            return {}

        fetched_ids = fetched.get("ids", []) or []
        documents = fetched.get("documents", []) if include_documents else []
        metadatas = fetched.get("metadatas", []) if include_metadatas else []

        records = {}
        for i, cid in enumerate(fetched_ids):
            record = {}
            if include_documents:
                record["document"] = documents[i] if i < len(documents) and documents[i] else ""
            if include_metadatas:
                record["metadata"] = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            records[cid] = record

        return records

    def _bm25_retrieve(self, query: str, category: Optional[str] = None,
                       author: Optional[str] = None) -> list[dict]:
        """Lexical retrieval via BM25 with optional Chroma metadata post-filtering."""
        if not self.enable_bm25 or self.bm25 is None or not self.bm25_chunk_ids:
            return []

        tokens = tokenize_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:self.k_lex * 2]  # Fetch extra for filtering

        ranked_chunk_ids = [
            self.bm25_chunk_ids[idx]
            for idx in top_indices
            if scores[idx] > 0
        ]
        if not ranked_chunk_ids:
            return []

        needs_filtering = bool(category or author)
        metadata_by_id = {}
        if needs_filtering:
            records = self._fetch_chunk_records(
                ranked_chunk_ids,
                include_documents=False,
                include_metadatas=True,
            )
            metadata_by_id = {cid: rec.get("metadata", {}) for cid, rec in records.items()}

        normalized_category = category.strip().lower() if category else None
        normalized_author = author.strip().lower() if author else None

        candidates = []
        for idx in top_indices:
            if scores[idx] <= 0 or len(candidates) >= self.k_lex:
                continue

            chunk_id = self.bm25_chunk_ids[idx]
            meta = metadata_by_id.get(chunk_id, {}) if needs_filtering else {}

            # Post-filter by category/author
            if normalized_category and normalized_category not in meta.get("categories", "").lower():
                continue
            if normalized_author and normalized_author not in meta.get("authors", "").lower():
                continue

            candidate = {
                "chunk_id": chunk_id,
                "bm25_score": float(scores[idx]),
                "source": "bm25",
            }
            if meta:
                candidate["metadata"] = meta

            candidates.append(candidate)

        return candidates

    def _merge_and_normalize(
        self, dense_candidates: list[dict], bm25_candidates: list[dict]
    ) -> list[dict]:
        """Merge dense and BM25 candidates and compute RRF fusion score."""
        merged = {}
        dense_rank_by_id = {
            c["chunk_id"]: rank for rank, c in enumerate(dense_candidates, start=1)
        }
        bm25_rank_by_id = {
            c["chunk_id"]: rank for rank, c in enumerate(bm25_candidates, start=1)
        }

        for c in dense_candidates:
            merged[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "chunk_text": c.get("chunk_text", ""),
                "metadata": c.get("metadata", {}),
                "dense_score_raw": c["dense_score"],
                "bm25_score_raw": 0.0,
                "sources": ["dense"],
            }

        for c in bm25_candidates:
            cid = c["chunk_id"]
            bm25_meta = c.get("metadata", {})
            if cid in merged:
                merged[cid]["bm25_score_raw"] = c["bm25_score"]
                if "bm25" not in merged[cid]["sources"]:
                    merged[cid]["sources"].append("bm25")
                if not merged[cid]["metadata"] and bm25_meta:
                    merged[cid]["metadata"] = bm25_meta
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "chunk_text": "",
                    "metadata": bm25_meta,
                    "dense_score_raw": 0.0,
                    "bm25_score_raw": c["bm25_score"],
                    "sources": ["bm25"],
                }

        if not merged:
            return []

        candidates = list(merged.values())

        # Fetch only the fields that are still missing after merge.
        missing_doc_ids = [cid for cid, c in merged.items() if not c["chunk_text"]]
        missing_meta_ids = [cid for cid, c in merged.items() if not c["metadata"]]
        ids_to_fetch = list(dict.fromkeys(missing_doc_ids + missing_meta_ids))

        if ids_to_fetch:
            records = self._fetch_chunk_records(
                ids_to_fetch,
                include_documents=bool(missing_doc_ids),
                include_metadatas=bool(missing_meta_ids),
            )

            for cid, record in records.items():
                candidate = merged.get(cid)
                if not candidate:
                    continue
                if not candidate["chunk_text"]:
                    candidate["chunk_text"] = record.get("document", "")
                if not candidate["metadata"]:
                    candidate["metadata"] = record.get("metadata", {})

        candidates = list(merged.values())

        for c in candidates:
            dense_rank = dense_rank_by_id.get(c["chunk_id"])
            bm25_rank = bm25_rank_by_id.get(c["chunk_id"])

            c["dense_rank"] = dense_rank
            c["bm25_rank"] = bm25_rank
            c["dense_rrf"] = 1.0 / (self.rrf_k + dense_rank) if dense_rank else 0.0
            c["bm25_rrf"] = 1.0 / (self.rrf_k + bm25_rank) if bm25_rank else 0.0
            c["fusion_score"] = c["dense_rrf"] + c["bm25_rrf"]

        candidates.sort(key=lambda x: x["fusion_score"], reverse=True)
        return candidates[:self.merge_top_m]

    # ------------------------------------------------------------------
    # Analytics extraction
    # ------------------------------------------------------------------

    def extract_analytics(self, candidates: list[dict]) -> dict:
        """
        Extract analytics from the top retrieved candidates.
        Returns top authors and most common categories.
        """
        from collections import Counter
        
        author_counter = Counter()
        category_counter = Counter()
        
        for c in candidates:
            meta = c.get("metadata", {})
            
            # Count authors
            authors_str = meta.get("authors", "")
            if authors_str:
                for author in authors_str.split(","):
                    author = author.strip()
                    if author:
                        author_counter[author] += 1
            
            # Count categories
            cats_str = meta.get("categories", "")
            if cats_str:
                for cat in cats_str.split(","):
                    cat = cat.strip()
                    if cat:
                        category_counter[cat] += 1
        
        return {
            "top_authors": [{"name": a, "count": c} for a, c in author_counter.most_common(5)],
            "top_categories": [{"name": cat, "count": c} for cat, c in category_counter.most_common(5)],
            "total_unique_papers": len(set(
                c.get("metadata", {}).get("paper_id", "") for c in candidates
            )),
        }

    # ------------------------------------------------------------------
    # Context compression (MMR-based sentence selection)
    # ------------------------------------------------------------------

    def compress_context(self, query: str, passages: list[dict],
                         max_sentences: int = 25,
                         intent: str = INTENT_DISCOVERY) -> str:
        """
        Compress retrieved passages into a compact context.

        For explanatory queries: skip MMR sentence-shredding and use full
        passages (preserves paragraph coherence for step-by-step explanations).

        For other intents: use MMR-based sentence selection to maximize
        diversity and relevance.
        """
        # For explanatory queries, preserve full passage structure
        if intent == INTENT_EXPLANATORY:
            context_parts = []
            for i, p in enumerate(passages, 1):
                title = p.get("title", p.get("metadata", {}).get("title", ""))
                text = p.get("chunk_text", "")
                if text:
                    header = f"[Source {i}: {title}]" if title else f"[Source {i}]"
                    context_parts.append(f"{header}\n{text}")
            return "\n\n".join(context_parts)

        # For non-explanatory queries, use MMR sentence selection
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)

            all_sentences = []
            for p in passages:
                text = p.get("chunk_text", "")
                if text:
                    sents = nltk.sent_tokenize(text)
                    all_sentences.extend(sents)

            if not all_sentences or len(all_sentences) <= max_sentences:
                return " ".join(all_sentences)

            # Encode query and sentences
            query_emb = list(self.embed_model.embed([query]))[0]
            sent_embs = np.array(list(self.embed_model.embed(all_sentences)))

            # MMR selection
            selected_indices = []
            remaining = list(range(len(all_sentences)))
            lambda_param = 0.7

            for _ in range(min(max_sentences, len(all_sentences))):
                best_score = -float('inf')
                best_idx = -1

                for idx in remaining:
                    relevance = float(np.dot(sent_embs[idx], query_emb))
                    if selected_indices:
                        selected_embs = sent_embs[selected_indices]
                        max_sim = float(np.max(np.dot(selected_embs, sent_embs[idx])))
                    else:
                        max_sim = 0.0
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    remaining.remove(best_idx)

            selected_indices.sort()
            return " ".join(all_sentences[i] for i in selected_indices)

        except Exception as e:
            log.warning(f"Context compression failed: {e}. Using raw passages.")
            return " ".join(p.get("chunk_text", "") for p in passages)

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_n: int = None,
                 category: Optional[str] = None,
                 author: Optional[str] = None,
                 start_year: Optional[int] = None,
                 intent: Optional[str] = None) -> dict:
        """
        Full hybrid retrieval pipeline with optional metadata filtering,
        intent-aware recency control, and smart ArXiv API fallback.

        Returns dict with:
            - passages: list of top-N passages with metadata
            - trace: retrieval trace with IDs and timings
            - analytics: top authors and categories from results
        """
        top_n = top_n or self.final_top_n
        intent = intent or classify_query_intent(query)
        trace = {"intent": intent}
        t0 = time.time()

        is_explanatory = (intent == INTENT_EXPLANATORY)

        # Build Chroma filter
        where_filter = self._build_chroma_where(category=category, author=author)

        # Step 1: Dense retrieval
        t1 = time.time()
        dense_candidates = self._dense_retrieve(query, where_filter=where_filter)
        trace["dense_ms"] = round((time.time() - t1) * 1000, 1)
        trace["dense_ids"] = [c["chunk_id"] for c in dense_candidates[:10]]

        # Step 2: BM25 retrieval
        t2 = time.time()
        bm25_candidates = self._bm25_retrieve(query, category=category, author=author)
        trace["bm25_ms"] = round((time.time() - t2) * 1000, 1)
        trace["bm25_ids"] = [c["chunk_id"] for c in bm25_candidates[:10]]
        trace["bm25_enabled"] = self.enable_bm25

        # Step 3: Merge and RRF fusion
        # For explanatory queries, keep a larger candidate pool for reranking
        orig_merge_top = self.merge_top_m
        if is_explanatory:
            self.merge_top_m = 30

        t3 = time.time()
        merged = self._merge_and_normalize(dense_candidates, bm25_candidates)
        trace["merge_ms"] = round((time.time() - t3) * 1000, 1)
        trace["merged_count"] = len(merged)
        trace["fusion_method"] = "rrf"
        trace["rrf_k"] = self.rrf_k

        # Restore original merge_top_m
        self.merge_top_m = orig_merge_top

        # Step 3.5: Apply recency boost — DISABLED for explanatory queries
        # For explanatory queries, canonical/foundational papers matter more
        if not is_explanatory:
            merged = self._apply_recency_boost(merged)
            merged.sort(key=lambda x: x["fusion_score"], reverse=True)
            trace["recency_boost"] = True
        else:
            trace["recency_boost"] = False

        # Step 3.6: Filter by year (post-filter)
        if start_year:
            merged = self._filter_by_year(merged, start_year=start_year)

        # Step 3.6.5: Semantic Pruning — NEW
        # For explanatory queries, prune chunks that don't look like explanations/methods
        if is_explanatory:
            merged = self._semantic_pruning(query, merged)

        # Step 3.7: ArXiv API Fallback — intent-aware
        best_score = merged[0].get("fusion_score", 0) if merged else 0
        trace["local_relevance"] = round(float(best_score), 4)

        if best_score < 0.05 or (is_explanatory and best_score < 0.08):
            trace["arxiv_api_fallback"] = True
            foundational = self._fetch_foundational_papers(query, top_n=3, intent=intent)
            if foundational:
                merged.extend(foundational)
                merged.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        else:
            trace["arxiv_api_fallback"] = False

        # Step 4: Cross-encoder reranking — intent-aware text mode
        t4 = time.time()
        rerank_mode = "combined" if is_explanatory else "default"
        try:
            reranked = self.reranker.rerank(
                query, merged, top_n=top_n, rerank_text_mode=rerank_mode
            )
            trace["rerank_fallback"] = False
            trace["rerank_text_mode"] = rerank_mode
        except Exception as e:
            log.warning(f"Reranking failed; using fusion ranking fallback: {e}")
            reranked = merged[:top_n]
            trace["rerank_fallback"] = True
        trace["rerank_ms"] = round((time.time() - t4) * 1000, 1)
        trace["reranked_ids"] = [c["chunk_id"] for c in reranked]

        # Extract analytics from the broader merged set
        analytics = self.extract_analytics(merged)

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

        # Apply filters info to trace
        trace["filters"] = {
            "category": category,
            "author": author,
            "start_year": start_year,
            "recency_boost": trace.get("recency_boost", True),
        }

        return {
            "passages": passages,
            "trace": trace,
            "analytics": analytics,
        }

    def _fetch_foundational_papers(self, query: str, top_n: int = 3,
                                    intent: str = INTENT_DISCOVERY) -> list[dict]:
        """
        Fetch foundational paper abstracts directly from the ArXiv API.
        Intent-aware: for explanatory queries, search for surveys/tutorials.
        """
        try:
            log.info(f"ArXiv API fallback (intent={intent}): '{query}'")

            # For explanatory queries, append survey/tutorial terms
            search_query = query
            if intent == INTENT_EXPLANATORY:
                search_query = f"{query} survey OR tutorial OR overview"

            encoded_query = urllib.parse.quote(search_query)
            url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{encoded_query}&start=0"
                f"&max_results={top_n}&sortBy=relevance&sortOrder=descending"
            )

            with urllib.request.urlopen(url, timeout=8) as resp:
                feed = feedparser.parse(resp.read())

            foundational = []
            for entry in feed.entries:
                paper_id = entry.id.split('/abs/')[-1].split('v')[0]
                title = entry.title.replace('\n', ' ').strip()
                abstract = entry.summary.replace('\n', ' ').strip()
                foundational.append({
                    "chunk_id": f"arxiv_api_{paper_id}",
                    "chunk_text": f"{title}. {abstract}",
                    "metadata": {
                        "paper_id": paper_id,
                        "title": title,
                        "authors": ", ".join(a.name for a in entry.authors),
                        "categories": ", ".join(t.term for t in entry.tags),
                    },
                    "fusion_score": 0.08,
                    "rerank_score": 0.08,
                    "sources": ["arxiv_api"],
                })
            return foundational
        except Exception as e:
            log.warning(f"ArXiv API fallback failed: {e}")
            return []

    def retrieve_ids(self, query: str, top_n: int = None) -> list[str]:
        """Convenience method that returns only chunk IDs (for evaluation)."""
        result = self.retrieve(query, top_n=top_n)
        return [p["chunk_id"] for p in result["passages"]]

    # ------------------------------------------------------------------
    # Similar papers
    # ------------------------------------------------------------------

    def find_similar_papers(self, paper_id: str, top_n: int = 5) -> list[dict]:
        """
        Find papers similar to a given paper by averaging its chunk embeddings
        and querying Chroma for nearest neighbors.
        """
        try:
            # Get all chunks for this paper
            results = self.collection.get(
                where={"paper_id": paper_id},
                include=["documents", "metadatas", "embeddings"],
            )

            if not results["ids"]:
                return []

            # Compute mean embedding
            embeddings = np.array(results["embeddings"])
            mean_embedding = np.mean(embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            # Query Chroma for nearest neighbors
            similar = self.collection.query(
                query_embeddings=[mean_embedding.tolist()],
                n_results=top_n * 5,  # Fetch extra to deduplicate papers
                include=["documents", "metadatas", "distances"],
            )

            # Deduplicate by paper_id, skip the source paper
            seen_papers = {paper_id}
            similar_papers = []

            for i in range(len(similar["ids"][0])):
                meta = similar["metadatas"][0][i]
                pid = meta.get("paper_id", "")
                if pid and pid not in seen_papers:
                    seen_papers.add(pid)
                    similar_papers.append({
                        "paper_id": pid,
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", ""),
                        "categories": meta.get("categories", ""),
                        "similarity_score": round(1.0 - similar["distances"][0][i], 4),
                        "chunk_text": similar["documents"][0][i][:300],
                    })
                    if len(similar_papers) >= top_n:
                        break

            return similar_papers

        except Exception as e:
            log.error(f"Similar papers search failed: {e}")
            return []
