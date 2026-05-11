#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Spaces defaults to 7860, Render defaults to 10000.
PORT="${PORT:-7860}"

echo "Starting FastAPI server on port ${PORT}..."
exec uvicorn api.app:app --host 0.0.0.0 --port "$PORT"
