"""
cache.py — Redis caching layer for the ArXiv RAG Assistant.

Features:
    1. Query Response Cache   — hash(query + paper_id) → cached LLM answer (1h TTL)
    2. Session Chat Cache     — conversation messages in Redis list (24h TTL)
    3. Graceful degradation   — all operations silently skip if Redis is unavailable

Design for 30MB Redis Cloud free tier:
    - allkeys-lru eviction policy (set in Redis Cloud dashboard)
    - Each cached response ≈ 2-5KB → ~6,000-10,000 entries fit comfortably
    - Session cache ≈ 1KB per conversation → negligible overhead
    - TTLs ensure natural turnover
"""

import hashlib
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection (lazy singleton)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_available = False


def _get_redis():
    """Lazy-initialize Redis connection."""
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        log.info("REDIS_URL not set — caching disabled.")
        _redis_available = False
        _redis_client = "disabled"  # sentinel to avoid retrying
        return None

    try:
        import redis
        _redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=False,
        )
        # Test connection
        _redis_client.ping()
        _redis_available = True
        log.info("Redis connected successfully.")
        return _redis_client
    except Exception as e:
        log.warning(f"Redis connection failed: {e}. Caching disabled.")
        _redis_available = False
        _redis_client = "disabled"
        return None


# ---------------------------------------------------------------------------
# Query Response Cache (1h TTL)
# ---------------------------------------------------------------------------

QUERY_CACHE_TTL = 3600  # 1 hour


def _make_query_cache_key(query: str, paper_id: Optional[str] = None) -> str:
    """Generate deterministic cache key from query + optional paper scope."""
    raw = f"{query.strip().lower()}|{paper_id or 'corpus'}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"qcache:{h}"


def get_cached_response(query: str, paper_id: Optional[str] = None) -> Optional[dict]:
    """Check Redis for a cached LLM response. Returns None on miss."""
    r = _get_redis()
    if r is None:
        return None

    key = _make_query_cache_key(query, paper_id)
    try:
        data = r.get(key)
        if data:
            log.debug(f"Cache HIT: {key}")
            return json.loads(data)
    except Exception as e:
        log.warning(f"Redis GET error: {e}")
    return None


def set_cached_response(
    query: str,
    response: dict,
    paper_id: Optional[str] = None,
    ttl: int = QUERY_CACHE_TTL,
) -> None:
    """Store an LLM response in Redis cache."""
    r = _get_redis()
    if r is None:
        return

    key = _make_query_cache_key(query, paper_id)
    try:
        # Only cache essential fields to save memory
        cache_data = {
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "retrieval_trace": response.get("retrieval_trace", {}),
        }
        r.setex(key, ttl, json.dumps(cache_data, default=str))
        log.debug(f"Cache SET: {key} (TTL={ttl}s)")
    except Exception as e:
        log.warning(f"Redis SET error: {e}")


# ---------------------------------------------------------------------------
# Session Chat Cache (24h TTL) — for fast reload on page refresh
# ---------------------------------------------------------------------------

SESSION_CACHE_TTL = 86400  # 24 hours


def _session_key(conversation_id: str) -> str:
    return f"chat:{conversation_id}:msgs"


def cache_message(conversation_id: str, role: str, content: str,
                  sources_json: Optional[str] = None) -> None:
    """Append a message to the conversation's Redis cache list."""
    r = _get_redis()
    if r is None:
        return

    key = _session_key(conversation_id)
    msg = json.dumps({
        "role": role,
        "content": content,
        "sources_json": sources_json,
    })
    try:
        r.rpush(key, msg)
        r.expire(key, SESSION_CACHE_TTL)
        # Keep list bounded (max 100 messages per conversation)
        r.ltrim(key, -100, -1)
    except Exception as e:
        log.warning(f"Redis session cache error: {e}")


def get_cached_messages(conversation_id: str) -> Optional[list[dict]]:
    """Retrieve cached conversation messages. Returns None if not cached."""
    r = _get_redis()
    if r is None:
        return None

    key = _session_key(conversation_id)
    try:
        raw_msgs = r.lrange(key, 0, -1)
        if raw_msgs:
            return [json.loads(m) for m in raw_msgs]
    except Exception as e:
        log.warning(f"Redis session read error: {e}")
    return None


def invalidate_session_cache(conversation_id: str) -> None:
    """Delete a conversation's cached messages (e.g., on conversation delete)."""
    r = _get_redis()
    if r is None:
        return

    try:
        r.delete(_session_key(conversation_id))
    except Exception as e:
        log.warning(f"Redis session invalidate error: {e}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def redis_health() -> dict:
    """Return Redis connection status for health endpoint."""
    r = _get_redis()
    if r is None:
        return {"status": "unavailable", "memory_used": "N/A"}
    try:
        info = r.info("memory")
        return {
            "status": "connected",
            "memory_used_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
            "memory_max_mb": round(info.get("maxmemory", 0) / (1024 * 1024), 2),
        }
    except Exception:
        return {"status": "error", "memory_used": "N/A"}
