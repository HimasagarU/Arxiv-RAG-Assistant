"""
test_eval.py — Tests for retrieval pipeline and API.

Usage:
    conda run -n pytorch pytest tests/test_eval.py -v
"""

import json
import os
import sys

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Unit tests for evaluation metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test retrieval evaluation metrics."""

    def test_recall_at_k(self):
        from rerank.evaluate import recall_at_k

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["b", "d"]

        assert recall_at_k(retrieved, relevant, 5) == 1.0
        assert recall_at_k(retrieved, relevant, 2) == 0.5
        assert recall_at_k(retrieved, relevant, 1) == 0.0
        assert recall_at_k([], relevant, 5) == 0.0

    def test_precision_at_k(self):
        from rerank.evaluate import precision_at_k

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["b", "d"]

        assert precision_at_k(retrieved, relevant, 5) == 0.4
        assert precision_at_k(retrieved, relevant, 2) == 0.5
        assert precision_at_k(retrieved, relevant, 1) == 0.0

    def test_reciprocal_rank(self):
        from rerank.evaluate import reciprocal_rank

        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
        assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
        assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1.0 / 3)
        assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0

    def test_ndcg_at_k(self):
        from rerank.evaluate import ndcg_at_k

        # Perfect ranking
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0
        # No relevant docs
        assert ndcg_at_k(["x", "y"], ["a", "b"], 2) == 0.0


# ---------------------------------------------------------------------------
# Unit tests for chunking
# ---------------------------------------------------------------------------

class TestChunking:
    """Test chunking logic."""

    def test_short_text_single_chunk(self):
        from ingest.chunking import chunk_text, get_tokenizer
        tokenizer = get_tokenizer()
        chunks = chunk_text("Hello world, this is a test.", tokenizer, chunk_size=300)
        assert len(chunks) == 1
        assert chunks[0]["token_count"] < 300

    def test_overlap_chunks(self):
        from ingest.chunking import chunk_text, get_tokenizer
        tokenizer = get_tokenizer()
        # Create a long text
        long_text = " ".join(["word"] * 500)
        chunks = chunk_text(long_text, tokenizer, chunk_size=100, overlap_frac=0.2)
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size tokens
        for c in chunks:
            assert c["token_count"] <= 100


# ---------------------------------------------------------------------------
# Unit tests for ArXiv ingestion helpers
# ---------------------------------------------------------------------------

class TestIngestion:
    """Test ingestion helpers."""

    def test_extract_arxiv_id(self):
        from ingest.ingest_arxiv import extract_arxiv_id
        assert extract_arxiv_id("http://arxiv.org/abs/2301.12345v1") == "2301.12345"
        assert extract_arxiv_id("http://arxiv.org/abs/2305.01234v2") == "2305.01234"

    def test_clean_text(self):
        from ingest.ingest_arxiv import clean_text
        assert clean_text("  hello   world  \n test  ") == "hello world test"


# ---------------------------------------------------------------------------
# Integration test (requires built indexes)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists("data/chroma_db") or not os.path.exists("data/bm25_index.pkl"),
    reason="Indexes not built yet",
)
class TestIntegration:
    """Integration tests (require built indexes)."""

    def test_retriever_does_not_preload_chunk_metadata(self):
        from api.retrieval import HybridRetriever

        retriever = HybridRetriever()
        assert not hasattr(retriever, "chunk_metadata")

    def test_retrieval_pipeline(self):
        from api.retrieval import HybridRetriever
        retriever = HybridRetriever()
        result = retriever.retrieve("transformer attention mechanism", top_n=3)
        assert "passages" in result
        assert "trace" in result
        assert len(result["passages"]) <= 3

    def test_dense_only_mode_when_bm25_disabled(self, monkeypatch):
        from api.retrieval import HybridRetriever

        monkeypatch.setenv("ENABLE_BM25", "false")
        retriever = HybridRetriever()
        result = retriever.retrieve("transformer attention mechanism", top_n=3)

        assert retriever.enable_bm25 is False
        assert result["trace"].get("bm25_enabled") is False
        assert result["trace"].get("bm25_ids") == []
        assert "passages" in result
        assert len(result["passages"]) <= 3

    def test_bm25_filtering_uses_chroma_metadata(self, monkeypatch):
        from api.retrieval import HybridRetriever

        monkeypatch.setenv("ENABLE_BM25", "true")
        retriever = HybridRetriever()
        query = "learning"

        base_candidates = retriever._bm25_retrieve(query)
        if not base_candidates:
            pytest.skip("No BM25 candidates available for metadata filter test.")

        sample_id = base_candidates[0]["chunk_id"]
        fetched = retriever.collection.get(ids=[sample_id], include=["metadatas"])
        sample_metadatas = fetched.get("metadatas", [])
        if not sample_metadatas or not sample_metadatas[0]:
            pytest.skip("Sample candidate has no metadata in Chroma.")

        sample_meta = sample_metadatas[0]
        sample_category = next(
            (cat.strip() for cat in sample_meta.get("categories", "").split(",") if cat.strip()),
            None,
        )
        sample_author = next(
            (name.strip() for name in sample_meta.get("authors", "").split(",") if name.strip()),
            None,
        )

        if sample_category:
            category_candidates = retriever._bm25_retrieve(query, category=sample_category)
            assert sample_id in {c["chunk_id"] for c in category_candidates}
            assert all(
                sample_category.lower() in c.get("metadata", {}).get("categories", "").lower()
                for c in category_candidates
            )

        if sample_author:
            author_candidates = retriever._bm25_retrieve(query, author=sample_author)
            assert sample_id in {c["chunk_id"] for c in author_candidates}
            assert all(
                sample_author.lower() in c.get("metadata", {}).get("authors", "").lower()
                for c in author_candidates
            )

        if not sample_category and not sample_author:
            pytest.skip("Sample candidate metadata has no author/category values.")

    def test_api_health(self):
        from fastapi.testclient import TestClient
        from api.app import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
