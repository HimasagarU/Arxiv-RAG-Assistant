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
import redis.asyncio as aioredis

from utils.artifact_schema import retrieval_cache_token

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection (lazy pool)
# ---------------------------------------------------------------------------

_redis_pool: Optional[aioredis.ConnectionPool] = None
_redis_available: Optional[bool] = None


async def _get_redis() -> Optional[aioredis.Redis]:
    """Lazy-initialize Redis connection pool."""
    global _redis_pool, _redis_available

    if _redis_available is False:
        return None

    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        if _redis_available is None:
            log.info("REDIS_URL not set — caching disabled.")
        _redis_available = False
        return None

    if _redis_pool is None:
        max_conn = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        _redis_pool = aioredis.ConnectionPool.from_url(
            redis_url,
            max_connections=max(10, max_conn),
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=False,
        )

    client = aioredis.Redis(connection_pool=_redis_pool)
    if _redis_available is None:
        try:
            await client.ping()
            _redis_available = True
            log.info("Redis connected successfully.")
        except Exception as e:
            _redis_available = False
            log.warning("Redis connection failed: %s. Caching disabled.", e)
            return None

    return client


# ---------------------------------------------------------------------------
# Query Response Cache (1h TTL)
# ---------------------------------------------------------------------------

QUERY_CACHE_TTL = 3600  # 1 hour


async def _query_cache_epoch() -> str:
    """Redis-wide buster so ingest can invalidate cached answers without scanning keys."""
    r = await _get_redis()
    if r is None:
        return "0"
    try:
        v = await r.get("qcache:buster")
        return str(v or "0")
    except Exception:
        return "0"


def bump_query_cache_buster_sync() -> None:
    """Increment cache epoch (sync; safe from background threads)."""
    url = (os.getenv("REDIS_URL", "") or "").strip()
    if not url:
        return
    try:
        import redis as sync_redis

        client = sync_redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        client.incr("qcache:buster")
        client.close()
    except Exception as e:
        log.debug("bump_query_cache_buster_sync: %s", e)


async def bump_query_cache_buster() -> None:
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.incr("qcache:buster")
    except Exception as e:
        log.warning("Redis INCR qcache:buster failed: %s", e)


def _make_query_cache_key(query: str, paper_id: Optional[str] = None, *, epoch: str = "0") -> str:
    """Generate deterministic cache key from query + optional paper scope + pipeline version."""
    raw = f"{query.strip().lower()}|{paper_id or 'corpus'}|{retrieval_cache_token()}|{epoch}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"qcache:{h}"


async def get_cached_response(query: str, paper_id: Optional[str] = None) -> Optional[dict]:
    """Check Redis for a cached LLM response. Returns None on miss."""
    r = await _get_redis()
    if r is None:
        return None

    epoch = await _query_cache_epoch()
    key = _make_query_cache_key(query, paper_id, epoch=epoch)
    try:
        data = await r.get(key)
        if data:
            log.debug(f"Cache HIT: {key}")
            return json.loads(data)
    except Exception as e:
        log.warning(f"Redis GET error: {e}")
    return None


async def set_cached_response(
    query: str,
    response: dict,
    paper_id: Optional[str] = None,
    ttl: int = QUERY_CACHE_TTL,
) -> None:
    """Store an LLM response in Redis cache."""
    r = await _get_redis()
    if r is None:
        return

    epoch = await _query_cache_epoch()
    key = _make_query_cache_key(query, paper_id, epoch=epoch)
    try:
        # Only cache essential fields to save memory
        cache_data = {
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "retrieval_trace": response.get("retrieval_trace", {}),
        }
        await r.setex(key, ttl, json.dumps(cache_data, default=str))
        log.debug(f"Cache SET: {key} (TTL={ttl}s)")
    except Exception as e:
        log.warning(f"Redis SET error: {e}")


# ---------------------------------------------------------------------------
# Session Chat Cache (24h TTL) — for fast reload on page refresh
# ---------------------------------------------------------------------------

SESSION_CACHE_TTL = 86400  # 24 hours


def _session_key(conversation_id: str) -> str:
    return f"chat:{conversation_id}:msgs"


async def cache_message(conversation_id: str, role: str, content: str,
                        sources_json: Optional[str] = None) -> None:
    """Append a message to the conversation's Redis cache list."""
    r = await _get_redis()
    if r is None:
        return

    key = _session_key(conversation_id)
    msg = json.dumps({
        "role": role,
        "content": content,
        "sources_json": sources_json,
    })
    try:
        await r.rpush(key, msg)
        await r.expire(key, SESSION_CACHE_TTL)
        # Keep list bounded (max 100 messages per conversation)
        await r.ltrim(key, -100, -1)
    except Exception as e:
        log.warning(f"Redis session cache error: {e}")


async def get_cached_messages(conversation_id: str) -> Optional[list[dict]]:
    """Retrieve cached conversation messages. Returns None if not cached."""
    r = await _get_redis()
    if r is None:
        return None

    key = _session_key(conversation_id)
    try:
        raw_msgs = await r.lrange(key, 0, -1)
        if raw_msgs:
            return [json.loads(m) for m in raw_msgs]
    except Exception as e:
        log.warning(f"Redis session read error: {e}")
    return None


async def invalidate_session_cache(conversation_id: str) -> None:
    """Delete a conversation's cached messages (e.g., on conversation delete)."""
    r = await _get_redis()
    if r is None:
        return

    try:
        await r.delete(_session_key(conversation_id))
    except Exception as e:
        log.warning(f"Redis session invalidate error: {e}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def redis_health() -> dict:
    """Return Redis connection status for health endpoint."""
    r = await _get_redis()
    if r is None:
        return {"status": "unavailable", "memory_used": "N/A"}
    try:
        info = await r.info("memory")
        return {
            "status": "connected",
            "memory_used_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
            "memory_max_mb": round(info.get("maxmemory", 0) / (1024 * 1024), 2),
        }
    except Exception:
        return {"status": "error", "memory_used": "N/A"}
