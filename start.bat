@echo off
setlocal enabledelayedexpansion
title Story Teller

echo.
echo  ============================================
echo   Story Teller - Dark Fairy Tales for Adults
echo  ============================================
echo.

:: ── Find Python ─────────────────────────────────
set "PYTHON="
for %%P in (
    "C:\Users\siyan\AppData\Local\Programs\Python\Python312\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist %%P (
        set "PYTHON=%%~P"
        goto :found_python
    )
)
where py >nul 2>&1 && set "PYTHON=py" && goto :found_python
echo [ERROR] Python not found. Please install Python 3.12.
pause
exit /b 1

:found_python
echo [OK] Python: %PYTHON%

:: ── Check FFmpeg ────────────────────────────────
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] FFmpeg not in PATH. Checking winget install...
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
        echo [OK] FFmpeg found via winget
    ) else (
        echo [WARN] FFmpeg not found. Video assembly will fail.
        echo        Install with: winget install Gyan.FFmpeg
    )
) else (
    echo [OK] FFmpeg found
)

:: ── Setup Python venv ───────────────────────────
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating Python virtual environment...
    "%PYTHON%" -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)
echo [OK] Python venv active

:: ── Start Backend ───────────────────────────────
echo [START] Backend on port 8102...
start "StoryTeller Backend" cmd /k "cd /d %~dp0 && venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8102"

:: ── Wait for backend ────────────────────────────
echo Waiting for backend...
timeout /t 3 /nobreak >nul

:: ── Start Frontend ──────────────────────────────
cd /d "%~dp0\frontend"
if not exist "node_modules" (
    echo [SETUP] Installing frontend dependencies...
    call npm install
)
echo [START] Frontend on port 5191...
start "StoryTeller Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

:: ── Open browser ────────────────────────────────
timeout /t 3 /nobreak >nul
start http://localhost:5191

echo.
echo  Story Teller is running!
echo  Frontend: http://localhost:5191
echo  Backend:  http://localhost:8102
echo.
echo  Close this window to stop all services.
pause
