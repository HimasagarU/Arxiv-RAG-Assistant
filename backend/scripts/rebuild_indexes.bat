@echo off
REM rebuild_indexes.bat — Rebuild chunks and Qdrant indexes for 20k
echo ============================================
echo  ArXiv RAG — Rebuilding Indexes (20k Papers)
echo ============================================

echo.
echo [1/3] Chunking all exactly 20,000 papers...
conda run -n pytorch python ingest/chunking.py --reset

echo.
echo [2/3] Building Qdrant Cloud Vector Index...
conda run -n pytorch python index/build_qdrant.py

echo.
echo [3/3] PostgreSQL full-text search is automatic; no separate index needed.

echo.
echo Done! Pipeline successfully updated to exactly 20,000 papers.
pause
