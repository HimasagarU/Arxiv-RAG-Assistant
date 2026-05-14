"""Runtime configuration helpers."""

import logging
import os

_log = logging.getLogger(__name__)


def get_generation_context_top_n(default: int = 10) -> int:
    """GENERATION_CONTEXT_TOP_N — shared default for retrieval and generation."""
    raw = os.getenv("GENERATION_CONTEXT_TOP_N")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _log.warning("Invalid GENERATION_CONTEXT_TOP_N=%s; using default %s", raw, default)
        return default


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def is_low_memory_mode() -> bool:
    """Return True when the environment indicates a low-memory deployment."""
    if _env_truthy("LOW_MEMORY_MODE"):
        return True

    raw_mem = os.getenv("MEMORY_GB", "").strip()
    if raw_mem:
        try:
            return float(raw_mem) <= 1.0
        except ValueError:
            return False

    return False


def resolve_embedding_model(
    explicit_model: str | None = None,
    default_large: str = "BAAI/bge-large-en-v1.5",
    default_small: str = "BAAI/bge-small-en-v1.5",
) -> str:
    """Pick the embedding model with a low-memory default and env override."""
    if explicit_model:
        return explicit_model

    env_model = os.getenv("EMBEDDING_MODEL", "").strip()
    if env_model:
        return env_model

    return default_small if is_low_memory_mode() else default_large
