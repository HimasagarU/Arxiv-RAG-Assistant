#!/bin/bash
set -e

# --- Download data from R2 if not already present ---
if [ ! -d "/app/data/chroma_db" ]; then
    echo "Data index not found. Downloading 20k-paper index from Cloudflare R2..."
    python api/fetch_data.py
    echo "Data download and extraction complete."
else
    echo "Data index already present, skipping download."
fi

# --- Start the FastAPI application ---
echo "Starting FastAPI server..."
exec uvicorn api.app:app --host 0.0.0.0 --port 8000
