"""
app_database.py — Application database connection manager (Supabase PostgreSQL).

Handles the SEPARATE application database for:
  - User accounts & authentication
  - Conversation / message history
  - Document ingestion job tracking

This is intentionally isolated from the existing Neon/Qdrant
vector-search database to prevent space exhaustion.

Usage:
    from db.app_database import get_app_db, init_app_db
    await init_app_db()            # call once at startup
    async with get_app_db() as db:
        ...
"""

import logging
import os

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

_raw_url = os.getenv("APP_DATABASE_URL", "")

# Convert standard postgresql:// to asyncpg-compatible postgresql+asyncpg://
if _raw_url.startswith("postgresql://"):
    _async_url = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgres://"):
    _async_url = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
else:
    _async_url = _raw_url

# ---------------------------------------------------------------------------
# Engine & Session Factory
# ---------------------------------------------------------------------------

import ssl as _ssl

engine = create_async_engine(
    _async_url,
    echo=False,
    poolclass=NullPool,  # Supabase free tier has limited connections
    connect_args={
        "server_settings": {"application_name": "arxiv-rag-app"},
        "ssl": "require",  # Supabase requires SSL
        "statement_cache_size": 0,  # Required for Supabase transaction-mode pooler
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
    from db.app_models import Base  # noqa: avoid circular import

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Application database tables created/verified (Supabase).")


async def close_app_db():
    """Dispose engine on shutdown."""
    await engine.dispose()
    log.info("Application database engine disposed.")
