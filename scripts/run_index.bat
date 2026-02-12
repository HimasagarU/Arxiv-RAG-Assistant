@echo off
REM run_index.bat — Build Chroma and BM25 indexes
echo ============================================
echo  ArXiv RAG — Index Build Pipeline
echo ============================================

echo.
echo [1/2] Building Chroma vector index (GPU)...
conda run -n pytorch python index/build_chroma.py

echo.
echo [2/2] Building BM25 lexical index...
conda run -n pytorch python index/build_bm25.py

echo.
echo Done! Indexes built in data/
pause
