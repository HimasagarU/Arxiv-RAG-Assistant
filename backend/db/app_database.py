"""
app_database.py — Application database connection manager (Consolidated Neon PostgreSQL).

Handles the database for:
  - User accounts & authentication (SQLAlchemy)
  - Conversation / message history (SQLAlchemy)
  - Document ingestion job tracking (SQLAlchemy)
  - Paper metadata & Citations (via the shared connection)

Now consolidated into Neon to reduce latency and complexity.
"""

import logging
import os
import ssl as _ssl
import urllib.parse as _urlparse

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the same database URL for both app logic and knowledge base
_raw_url = os.getenv("DATABASE_URL", os.getenv("APP_DATABASE_URL", ""))

if not _raw_url:
    log.error("DATABASE_URL not set. Application database will fail.")

def _make_async_url(raw: str) -> tuple[str, bool]:
    """
    Convert a standard postgres:// URL to asyncpg-compatible postgresql+asyncpg://.
    Strips psycopg2-style query params (sslmode, options, etc.) that asyncpg
    does not accept as connect() keyword arguments, and returns whether SSL
    was requested so we can pass it via connect_args instead.
    """
    if raw.startswith("postgres://"):
        raw = "postgresql+asyncpg://" + raw[len("postgres://"):]
    elif raw.startswith("postgresql://"):
        raw = "postgresql+asyncpg://" + raw[len("postgresql://"):]

    parsed = _urlparse.urlparse(raw)
    params = _urlparse.parse_qs(parsed.query, keep_blank_values=True)

    # Detect SSL intent from sslmode before stripping it
    sslmode = params.pop("sslmode", [None])[0]
    wants_ssl = sslmode in ("require", "verify-ca", "verify-full", "prefer")

    # Strip any remaining psycopg2/libpq-only params asyncpg rejects
    for key in ("options", "channel_binding"):
        params.pop(key, None)

    clean_query = _urlparse.urlencode(
        {k: v[0] for k, v in params.items()}, safe=""
    ) if params else ""
    clean_parsed = parsed._replace(query=clean_query)
    return _urlparse.urlunparse(clean_parsed), wants_ssl


_async_url, _wants_ssl = _make_async_url(_raw_url)

# Build connect_args — asyncpg uses ssl= (True / SSLContext), not sslmode=
_ssl_ctx: bool | _ssl.SSLContext = (
    _ssl.create_default_context() if _wants_ssl else False
)

# ---------------------------------------------------------------------------
# Engine & Session Factory
# ---------------------------------------------------------------------------

engine = create_async_engine(
    _async_url,
    echo=False,
    poolclass=NullPool,
    connect_args={
        "server_settings": {"application_name": "arxiv-rag-app"},
        "ssl": _ssl_ctx,
        "statement_cache_size": 0,
    },
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Dependency for FastAPI
# ---------------------------------------------------------------------------

async def get_app_db():
    """Yield an async DB session — use as a FastAPI dependency."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

async def init_app_db():
    """Create all tables on startup (idempotent)."""
    from db.app_models import Base  # Imported here to avoid a circular import.

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Application database tables created/verified on Neon.")


async def close_app_db():
    """Dispose engine on shutdown."""
    await engine.dispose()
    log.info("Application database engine disposed.")
