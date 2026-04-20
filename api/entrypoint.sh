#!/bin/bash
set -e

DATA_DIR="/app/data"
CHROMA_DIR="$DATA_DIR/chroma_db"
CHROMA_SQLITE="$CHROMA_DIR/chroma.sqlite3"
BM25_INDEX="$DATA_DIR/bm25_index.pkl"
PAPERS_DB="$DATA_DIR/arxiv_papers.db"

has_chroma_shard_dir() {
    if find "$CHROMA_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

is_data_ready() {
    [ -f "$CHROMA_SQLITE" ] || return 1
    [ -f "$BM25_INDEX" ] || return 1
    [ -f "$PAPERS_DB" ] || return 1
    has_chroma_shard_dir || return 1
    return 0
}

# --- Download data from R2 if required artifacts are missing ---
if is_data_ready; then
    echo "Data index already present and complete, skipping download."
else
    echo "Required index files are missing or incomplete. Downloading 20k-paper index from Cloudflare R2..."
    python api/fetch_data.py

    if ! is_data_ready; then
        echo "Data download did not produce all required artifacts. Exiting startup."
        exit 1
    fi

    echo "Data download and extraction complete."
fi

# --- Start the FastAPI application ---
echo "Starting FastAPI server..."
exec uvicorn api.app:app --host 0.0.0.0 --port 8000
