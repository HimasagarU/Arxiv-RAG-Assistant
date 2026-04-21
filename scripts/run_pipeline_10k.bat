@echo off
REM run_pipeline_10k.bat — Run 10k Pipeline and Eval
echo ============================================
echo  ArXiv RAG — 10k Pipeline Automation
echo ============================================

echo.
echo [1/4] Fetching 10,000 papers + extracting full text from PDFs (this can take longer)...
conda run -n pytorch python ingest/ingest_arxiv.py --max-papers 10000 --include-full-text --max-fulltext-papers 0

echo.
echo [2/4] Chunking papers (source=auto, prefer full_text)...
conda run -n pytorch python ingest/chunking.py --source auto

echo.
echo [3/4] Building Vector and Lexical Indexes...
conda run -n pytorch python index/build_chroma.py
conda run -n pytorch python index/build_bm25.py

echo.
echo [4/4] Pipeline Complete!
echo Ready for Evaluation.
pause
