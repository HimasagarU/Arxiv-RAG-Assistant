"""Versioned artifact and cache contract for rebuild safety."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Bump when chunks_meta / BM25 / Qdrant payload contract changes materially.
ARTIFACT_SCHEMA_VERSION = 2

# Bump when retrieval fusion, rerank, or expansion logic changes (invalidates query cache).
RETRIEVAL_PIPELINE_VERSION = "2026.05.r2"


def retrieval_cache_token() -> str:
    """Short token for Redis keys (schema + pipeline)."""
    raw = f"v{ARTIFACT_SCHEMA_VERSION}|{RETRIEVAL_PIPELINE_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def write_artifact_manifest(
    data_dir: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write reproducibility manifest next to other artifacts."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "retrieval_pipeline_version": RETRIEVAL_PIPELINE_VERSION,
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
    }
    if extra:
        payload.update(extra)
    path = data_dir / "artifact_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
