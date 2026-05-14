"""
Centralized runtime feature flags (env-driven).

Flag history
------------
HyDE (ENABLE_HYDE), LLM query expansion (ENABLE_QUERY_EXPANSION_LLM), and LLM context
compression (ENABLE_CONTEXT_COMPRESSION) were originally *off by default* to keep Groq API
costs low (~$0.15–0.20 per 1 M tokens) and latency tight. Each adds ~1–3 s per query.

As of the May 2026 production update these three flags default to *true* in .env.example
because the quality improvement outweighs the marginal latency/cost.  Operators can still
disable them in their .env file for strict cost-control deployments.

All flags are read at call-time from environment variables, so they can be toggled without
restarting the server (values are not cached at import time).
"""

from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def env_tri(name: str) -> str | None:
    """Return True / False / None (unset → treat as 'auto' by caller)."""
    raw = (os.getenv(name, "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "on"
    if raw in ("0", "false", "no", "off"):
        return "off"
    return None


def get_mmr_lambda() -> float:
    try:
        return float(os.getenv("MMR_LAMBDA", "0.5"))
    except ValueError:
        return 0.5
