@echo off
title AgroSense Launcher
color 0A

echo ===================================================
echo           AgroSense AI System Launch
echo ===================================================

:: Start Backend
echo.
echo [1/2] Launching Backend API (FastAPI)...
start "AgroSense Backend" cmd /k "uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

:: Start Frontend
echo.
echo [2/2] Launching Frontend UI (Next.js)...
cd frontend
start "AgroSense Frontend" cmd /k "npm run dev"

echo.
echo ===================================================
echo              SYSTEM ONLINE
echo ===================================================
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this launcher (servers will keep running)...
pause >nul
