"""Shared ID utilities."""

import re
from uuid import NAMESPACE_URL, uuid5


def normalize_arxiv_paper_id(raw: str) -> str:
    """Return canonical ArXiv id ``YYYY.NNNNN`` for API, DB, and Qdrant filters.

    Accepts ``arxiv:2301.12345``, ``2301.12345v2``, URLs ending in ``.pdf``, etc.
    Version suffixes are stripped so Qdrant ``paper_id`` matches corpus conventions.
    """
    s = (raw or "").strip()
    if not s:
        return s
    s = s.lower().replace("arxiv:", "").strip()
    if s.endswith(".pdf"):
        s = s[:-4].strip()
    for prefix in ("https://arxiv.org/pdf/", "http://arxiv.org/pdf/", "https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", s)
    if m:
        return m.group(1)
    return s.strip()


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a deterministic UUID (Qdrant compatible)."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def paper_id_to_uuid(paper_id: str) -> str:
    """Deterministic point id for paper-level (parent) vectors in Qdrant."""
    return str(uuid5(NAMESPACE_URL, f"arxiv_paper:{paper_id}"))
