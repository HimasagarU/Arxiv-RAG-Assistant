@echo off
REM run_ingest.bat — Fetch ArXiv papers and chunk them
echo ============================================
echo  ArXiv RAG — Ingestion Pipeline
echo ============================================

echo.
echo [1/2] Fetching papers from ArXiv API + extracting full text from PDFs...
conda run -n pytorch python ingest/ingest_arxiv.py --max-papers 10000 --include-full-text --max-fulltext-papers 0

echo.
echo [2/2] Chunking papers (source=auto, prefer full_text)...
conda run -n pytorch python ingest/chunking.py --source auto --reset

echo.
echo Done! Check data/ for arxiv_papers.db and chunks.jsonl
pause
