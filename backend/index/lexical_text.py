"""Build BM25-oriented lexical strings (field-tagged) from chunk records."""

from __future__ import annotations

from utils.section_labels import normalize_section_label


def build_lexical_index_text(chunk: dict) -> str:
    """Compact multi-field string for lexical recall (title weighted by repetition)."""
    title = (chunk.get("title") or "").strip()
    authors = (chunk.get("authors") or "").strip()
    categories = (chunk.get("categories") or "").strip()
    sec = normalize_section_label(chunk.get("section_hint", "other"))
    body = (chunk.get("chunk_text") or "").strip()
    parts = []
    if title:
        parts.extend([f"title {title}", f"title {title}"])
    if authors:
        parts.append(f"authors {authors}")
    if categories:
        parts.append(f"categories {categories}")
    parts.append(f"section {sec}")
    if body:
        parts.append(f"body {body}")
    return "\n".join(parts)
