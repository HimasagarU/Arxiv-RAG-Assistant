@echo off
REM run_index.bat — Build Qdrant Cloud index and rely on PostgreSQL FTS
echo ============================================
echo  ArXiv RAG — Index Build Pipeline (Cloud)
echo ============================================

echo.
echo [1/2] Building Qdrant Cloud vector index...
conda run -n pytorch python index/build_qdrant.py

echo.
echo [2/2] PostgreSQL full-text search is automatic; no separate index needed.

echo.
echo Done! Cloud indexes are ready.
pause
