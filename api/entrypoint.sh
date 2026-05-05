#!/bin/bash
set -e

DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@postgres:5432/arxiv_rag}"

wait_for_postgres() {
    python - <<'PY'
import os
import sys
import time

import psycopg

db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/arxiv_rag")
deadline = time.time() + 60
last_error = None

while time.time() < deadline:
    try:
        conn = psycopg.connect(db_url, connect_timeout=5)
        conn.close()
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

print(f"Timed out waiting for PostgreSQL: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

run_db_migrations() {
    python - <<'PY'
import os

from db.database import get_db

db = get_db(os.environ.get("DATABASE_URL"))
db.run_migrations()
db.close()
PY
}

# --- Wait for PostgreSQL to be ready ---
echo "Waiting for PostgreSQL..."
wait_for_postgres

# --- Apply schema migrations ---
echo "Applying PostgreSQL schema migrations..."
run_db_migrations

# --- Start the FastAPI application ---
echo "Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
