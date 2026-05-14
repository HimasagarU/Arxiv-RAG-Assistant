"""Canonical section labels for chunking, indexing, and retrieval boosts."""

from __future__ import annotations

import re

# Labels used across BM25 meta, Qdrant payload, and retrieval SECTION_BASE_WEIGHT
CANONICAL_SECTIONS = frozenset(
    {
        "abstract",
        "introduction",
        "related_work",
        "background",
        "method",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "appendix",
        "preface",
        "other",
    }
)

_HEADING_TOKEN_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\babstract\b", re.I), "abstract"),
    (re.compile(r"\bintroduction\b", re.I), "introduction"),
    (re.compile(r"\brelated\s+work\b", re.I), "related_work"),
    (re.compile(r"\bbackground\b", re.I), "background"),
    (re.compile(r"\bpreliminar(?:y|ies)\b", re.I), "background"),
    (re.compile(r"\bmethod(?:s|ology)?\b", re.I), "method"),
    (re.compile(r"\bexperiment(?:s)?\b", re.I), "experiments"),
    (re.compile(r"\bevaluation\b", re.I), "experiments"),
    (re.compile(r"\bresult(?:s)?\b", re.I), "results"),
    (re.compile(r"\bablation\b", re.I), "results"),
    (re.compile(r"\banalysis\b", re.I), "results"),
    (re.compile(r"\bdiscussion\b", re.I), "discussion"),
    (re.compile(r"\bconclusion(?:s)?\b", re.I), "conclusion"),
    (re.compile(r"\bappendix\b", re.I), "appendix"),
    (re.compile(r"\bachnowledg", re.I), "other"),
    (re.compile(r"\breference|bibliography|works\s+cited\b", re.I), "other"),
    (re.compile(r"\bpreface\b", re.I), "preface"),
]


def normalize_section_label(raw: str | None) -> str:
    """Map arbitrary heading or hint string to a single canonical snake_case label."""
    if not raw or not str(raw).strip():
        return "other"
    s = str(raw).strip().lower()
    s = re.sub(r"^\d+(?:\.\d+)*\s*\.?\s*", "", s)
    s = re.sub(r"^[ivxlcdm]+\s*\.?\s*", "", s, flags=re.I)
    s = s.replace("_", " ").strip()

    if "abstract" in s:
        return "abstract"
    if "appendix" in s or "supplementary" in s:
        return "appendix"
    if "preface" in s:
        return "preface"

    for pattern, label in _HEADING_TOKEN_MAP:
        if pattern.search(s):
            return label

    if any(x in s for x in ("method", "approach", "architecture")):
        return "method"
    if "experiment" in s or "setup" in s:
        return "experiments"
    if "result" in s or "finding" in s:
        return "results"
    if "related" in s and "work" in s:
        return "related_work"
    if "conclusion" in s:
        return "conclusion"
    if "introduction" in s:
        return "introduction"
    if "discussion" in s:
        return "discussion"

    return "other"
