@echo off
setlocal enabledelayedexpansion
set PYTHON_EXE=C:\Users\himas\anaconda3\envs\pytorch\python.exe
if not defined START_OFFSET set START_OFFSET=0

REM run_pipeline_8cats_fullpdf.bat — Fresh 8-category full-PDF ingestion + indexing

echo ============================================================
echo  ArXiv RAG - 8 Categories x 2500 Full-PDF Ingestion
echo ============================================================

echo.
echo [0/4] Categories:
echo   cs.AI, cs.LG, cs.CV, cs.CL, stat.ML, cs.RO, cs.NE, eess.SP

set PER_CATEGORY=2500
set CATEGORIES=cs.AI cs.LG cs.CV cs.CL stat.ML cs.RO cs.NE eess.SP

echo.
echo [1/4] Ingesting !PER_CATEGORY! papers per category with full-text extraction...
set FIRST_CATEGORY=1
for %%C in (%CATEGORIES%) do (
  echo.
  echo [INGEST] Category %%C
  set "RESUME_ARG="
  if !FIRST_CATEGORY! EQU 1 if not "!START_OFFSET!"=="0" set "RESUME_ARG=--start-offset !START_OFFSET!"
  "%PYTHON_EXE%" -u ingest/ingest_arxiv.py --categories %%C --max-papers !PER_CATEGORY! --include-full-text --max-fulltext-papers 0 --pdf-timeout 60 !RESUME_ARG!
  if errorlevel 1 (
    echo.
    echo [ERROR] Ingestion failed for category %%C. Stop and retry this category.
    exit /b 1
  )
  set "FIRST_CATEGORY=0"
)

echo.
echo [2/4] Chunking from full_text...
"%PYTHON_EXE%" -u ingest/chunking.py --source full_text
if errorlevel 1 (
  echo [ERROR] Chunking failed.
  exit /b 1
)

echo.
echo [3/4] Building indexes...
"%PYTHON_EXE%" -u index/build_chroma.py
if errorlevel 1 (
  echo [ERROR] Chroma build failed.
  exit /b 1
)
"%PYTHON_EXE%" -u index/build_bm25.py
if errorlevel 1 (
  echo [ERROR] BM25 build failed.
  exit /b 1
)

echo.
echo [4/4] Final DB summary...
"%PYTHON_EXE%" -u -c "import sqlite3; c=sqlite3.connect('data/arxiv_papers.db'); cur=c.cursor(); cur.execute('select count(*) from papers'); t=cur.fetchone()[0]; cur.execute('select count(distinct paper_id) from papers'); d=cur.fetchone()[0]; cur.execute(\"select count(*) from papers where full_text is not null and trim(full_text)<>''\"); ft=cur.fetchone()[0]; print('total_rows', t); print('distinct_paper_ids', d); print('duplicates', t-d); print('full_text_nonempty', ft); c.close()"

echo.
echo Pipeline complete.
endlocal
