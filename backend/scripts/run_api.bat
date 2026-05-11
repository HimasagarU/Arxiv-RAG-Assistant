@echo off
REM run_api.bat — Start the FastAPI server
echo ============================================
echo  ArXiv RAG - API Server
echo ============================================
echo.
echo Starting FastAPI server on http://localhost:8000
echo Docs at http://localhost:8000/docs
echo.
pushd "%~dp0\.."
conda run -n pytorch uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
popd
