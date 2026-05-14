import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class MockRetriever:
    collections = {"arxiv_text": 2}
    papers_meta = {
        "2401.00001": {
            "title": "Mock Paper",
            "abstract": "A mock paper for API contract tests.",
            "authors": "Ada Lovelace",
            "categories": "cs.AI",
            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
            "published": "2024-01-01",
            "layer": "core",
            "is_seed": False,
        }
    }
    chunks_meta = [{"chunk_id": "2401.00001_text_0"}]

    def retrieve(self, query, top_n=5, category=None, author=None, start_year=None, intent=None):
        return {
            "passages": [
                {
                    "chunk_id": "2401.00001_text_0",
                    "paper_id": "2401.00001",
                    "title": "Mock Paper",
                    "authors": "Ada Lovelace",
                    "categories": "cs.AI",
                    "chunk_text": "Mock context about transformer circuits.",
                    "rerank_score": 0.91,
                }
            ],
            "trace": {
                "intent": intent or "discovery",
                "dense_ms": 1.0,
                "lex_ms": 1.0,
                "rerank_ms": 1.0,
                "total_ms": 3.0,
                "filters": {
                    "category": category,
                    "author": author,
                    "start_year": start_year,
                },
            },
            "analytics": {
                "top_authors": [{"name": "Ada Lovelace", "count": 1}],
                "top_categories": [{"name": "cs.AI", "count": 1}],
                "layer_distribution": {"core": 1},
                "total_unique_papers": 1,
            },
        }

    def compress_context(self, query, passages, intent="discovery"):
        return "\n".join(p["chunk_text"] for p in passages)

    def find_similar_papers(self, paper_id, top_n=5):
        return [
            {
                "paper_id": "2401.00002",
                "title": "Similar Mock Paper",
                "authors": "Grace Hopper",
                "categories": "cs.LG",
                "layer": "latest",
                "similarity_score": 0.87,
                "chunk_text": "A related mock paper.",
            }
        ][:top_n]


def _client(monkeypatch):
    import api.app as app_module

    retriever = MockRetriever()
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    app_module._state["retriever"] = retriever
    app_module.query_cache._cache.clear()
    app_module.app.dependency_overrides.clear()
    app_module.app.dependency_overrides[app_module.get_retriever] = lambda: retriever
    app_module.app.dependency_overrides[app_module.get_intent_classifier] = (
        lambda: lambda query: "explanatory" if "explain" in query.lower() else "discovery"
    )
    app_module.app.dependency_overrides[app_module.get_answer_generator] = (
        lambda: lambda prompt, intent="discovery", **kwargs: "Mock answer with citation [1]."
    )
    return TestClient(app_module.app)


def test_health_runs_without_external_dependencies(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["collections"]["arxiv_text"] == 2
    assert body["db_papers"] == 1


def test_query_contract_with_mocked_retriever_and_llm(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/query", json={"query": "Explain transformer circuits", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Mock answer with citation [1].")
    assert body["cached"] is False
    assert body["sources"][0]["paper_id"] == "2401.00001"
    assert body["retrieval_trace"]["intent"] == "explanatory"
    assert body["analytics"]["total_unique_papers"] == 1


def test_similar_papers_contract_with_mocked_retrieval(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/paper/2401.00001/similar?top_n=1")

    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"] == "2401.00001"
    assert body["similar_papers"] == [
        {
            "paper_id": "2401.00002",
            "title": "Similar Mock Paper",
            "authors": "Grace Hopper",
            "categories": "cs.LG",
            "layer": "latest",
            "similarity_score": 0.87,
            "chunk_text": "A related mock paper.",
        }
    ]


def test_generate_answer_falls_back_to_groq_when_gemini_is_rate_limited(monkeypatch):
    import api.app as app_module

    class GeminiStub:
        def generate(self, prompt, system_prompt, intent):
            raise RuntimeError("Gemini rate limit reached")

    class GroqStub:
        def generate(self, prompt, system_prompt, intent):
            return "Groq fallback answer"

    monkeypatch.setattr(app_module, "_get_gemini_client", lambda: GeminiStub())
    monkeypatch.setattr(app_module, "_get_groq_client", lambda: GroqStub())

    assert app_module.generate_answer(
        "prompt",
        surface=app_module.GenerationSurface.CHAT,
    ) == "Groq fallback answer"


def test_generate_answer_uses_groq_directly_for_public_surface(monkeypatch):
    import api.app as app_module

    class GroqStub:
        def generate(self, prompt, system_prompt, intent):
            return "Groq public answer"

    monkeypatch.setattr(app_module, "_get_groq_client", lambda: GroqStub())

    assert app_module.generate_answer(
        "prompt",
        surface=app_module.GenerationSurface.PUBLIC,
    ) == "Groq public answer"


def test_generate_answer_does_not_fallback_on_non_rate_limit(monkeypatch):
    import api.app as app_module

    class GeminiStub:
        def generate(self, prompt, system_prompt, intent):
            raise RuntimeError("Gemini backend error")

    monkeypatch.setattr(app_module, "_get_gemini_client", lambda: GeminiStub())

    with pytest.raises(RuntimeError, match="Gemini backend error"):
        app_module.generate_answer("prompt", surface=app_module.GenerationSurface.CHAT)


def test_generate_answer_stream_falls_back_to_groq_when_gemini_is_rate_limited(monkeypatch):
    import api.app as app_module

    class GeminiStub:
        def stream(self, prompt, system_prompt, intent):
            raise RuntimeError("Gemini daily request cap reached")

    class GroqStub:
        def stream(self, prompt, system_prompt, intent):
            yield "Groq "
            yield "stream"

    monkeypatch.setattr(app_module, "_get_gemini_client", lambda: GeminiStub())
    monkeypatch.setattr(app_module, "_get_groq_client", lambda: GroqStub())

    assert "".join(
        app_module._generate_answer_stream(
            "prompt",
            surface=app_module.GenerationSurface.CHAT,
        )
    ) == "Groq stream"


def test_generate_answer_stream_uses_groq_directly_for_public_surface(monkeypatch):
    import api.app as app_module

    class GroqStub:
        def stream(self, prompt, system_prompt, intent):
            yield "Groq "
            yield "public"

    monkeypatch.setattr(app_module, "_get_groq_client", lambda: GroqStub())

    assert "".join(
        app_module._generate_answer_stream(
            "prompt",
            surface=app_module.GenerationSurface.PUBLIC,
        )
    ) == "Groq public"
