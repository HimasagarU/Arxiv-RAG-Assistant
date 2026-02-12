@echo off
REM run_ingest.bat — Fetch ArXiv papers and chunk them
echo ============================================
echo  ArXiv RAG — Ingestion Pipeline
echo ============================================

echo.
echo [1/2] Fetching papers from ArXiv API...
conda run -n pytorch python ingest/ingest_arxiv.py --max-papers 5000

echo.
echo [2/2] Chunking papers...
conda run -n pytorch python ingest/chunking.py

echo.
echo Done! Check data/ for arxiv_papers.db and chunks.jsonl
pause
