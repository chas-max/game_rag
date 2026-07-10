@echo off
echo ===================================================
echo Starting Game RAG Servers...
echo ===================================================

echo [1/2] Starting Python FastAPI Backend on port 8000...
start "Game RAG Backend (FastAPI)" cmd /c "python main.py"

echo [2/2] Starting Next.js Frontend on port 3000...
start "Game RAG Frontend (Next.js)" cmd /c "cd frontend && npm run dev"

echo.
echo Both servers have been launched in separate console windows.
echo - Backend API: http://localhost:8000
echo - Frontend UI: http://localhost:3000
echo.
pause
