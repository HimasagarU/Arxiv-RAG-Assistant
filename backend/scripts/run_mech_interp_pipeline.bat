@echo off
REM ================================================================
REM Mechanistic Interpretability RAG - Full Pipeline
REM ================================================================
REM Usage: scripts\run_mech_interp_pipeline.bat
REM Requires: conda env "pytorch", PostgreSQL running
REM ================================================================

REM Ensure we run from the project root
cd /d "%~dp0\.."

echo ================================================================
echo  Mechanistic Interpretability RAG Pipeline
echo ================================================================

REM Step 1: Ingest seed papers
echo.
echo [Step 1/8] Ingesting seed papers...
call python ingest/ingest_arxiv.py --mode seed
if %errorlevel% neq 0 echo ERROR: Seed ingestion failed! & exit /b 1

REM Step 2: Citation expansion via Semantic Scholar
echo.
echo [Step 2/8] Expanding citations - backward + forward...
call python ingest/citation_expander.py
if %errorlevel% neq 0 echo WARNING: Citation expansion had errors, continuing...

REM Step 3: Keyword gap-filling
echo.
echo [Step 3/8] Keyword-based gap filling...
call python ingest/ingest_arxiv.py --mode keyword --max-pages 10
if %errorlevel% neq 0 echo WARNING: Keyword ingestion had errors, continuing...

REM Step 4: Timeline balance check + fill
echo.
echo [Step 4/8] Checking timeline balance...
call python ingest/timeline_balancer.py --fill-gaps
if %errorlevel% neq 0 echo WARNING: Timeline balancing had errors, continuing...

REM Step 5: Enrich with full text - PDF download + extraction
echo.
echo [Step 5/8] Enriching papers with full text...
call python ingest/ingest_arxiv.py --mode enrich --pdf-timeout 60
if %errorlevel% neq 0 echo WARNING: PDF enrichment had errors, continuing...

REM Step 6: Chunk text only
echo.
echo [Step 6/8] Chunking full text...
call python ingest/chunking.py --source auto --chunk-size 450 --reset
if %errorlevel% neq 0 echo ERROR: Chunking failed! & exit /b 1

REM Step 7: Build indexes - text Qdrant (FTS is automatic via PostgreSQL)
echo.
echo [Step 7/8] Building indexes...
call python index/build_qdrant.py
if %errorlevel% neq 0 echo ERROR: Qdrant index build failed! & exit /b 1
echo PostgreSQL full-text search is used for lexical retrieval; no separate index needed.

echo.
echo ================================================================
echo  Pipeline Complete!
echo  Run the API: uvicorn api.app:app --reload
echo ================================================================
