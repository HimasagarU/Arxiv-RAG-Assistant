"""Shared ID utilities."""

from uuid import NAMESPACE_URL, uuid5


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a deterministic UUID (Qdrant compatible)."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def paper_id_to_uuid(paper_id: str) -> str:
    """Deterministic point id for paper-level (parent) vectors in Qdrant."""
    return str(uuid5(NAMESPACE_URL, f"arxiv_paper:{paper_id}"))
