@echo off
REM rebuild_indexes.bat — Rebuild chunks and Chroma/BM25 indexes for 20k
echo ============================================
echo  ArXiv RAG — Rebuilding Indexes (20k Papers)
echo ============================================

echo.
echo [1/3] Chunking all exactly 20,000 papers...
conda run -n pytorch python ingest/chunking.py

echo.
echo [2/3] Building Chroma Vector Index...
conda run -n pytorch python index/build_chroma.py

echo.
echo [3/3] Building BM25 Index...
conda run -n pytorch python index/build_bm25.py

echo.
echo Done! Pipeline successfully updated to exactly 20,000 papers.
pause
