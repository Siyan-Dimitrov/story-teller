@echo off
setlocal enabledelayedexpansion
title Story Teller - Background Music Edition

echo.
echo  ============================================
echo   Story Teller - Background Music Edition
echo   (Jamendo API + Local Fallback)
echo  ============================================
echo.

:: ── Switch to the Background Music branch ─────
cd /d "%~dp0"
echo [GIT] Switching to feature/background-music branch...
git checkout feature/background-music 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Could not switch branch. Continuing with current branch.
)
for /f "tokens=*" %%b in ('git branch --show-current 2^>nul') do echo [OK] Branch: %%b

:: ── Jamendo API key (shared with yt_facts_video_gen) ──
set "JAMENDO_CLIENT_ID=8f198b3f"
echo [OK] Jamendo client ID configured

:: ── Kill existing processes ───────────────────
echo [CLEANUP] Stopping any existing services...

:: Kill backend on port 8102
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8102 " ^| findstr "LISTENING" 2^>nul') do (
    echo   Killing old backend (PID %%a)
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill frontend on port 5191
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5191 " ^| findstr "LISTENING" 2^>nul') do (
    echo   Killing old frontend (PID %%a)
    taskkill /F /PID %%a >nul 2>&1
)

:: Small pause to let ports free up
timeout /t 1 /nobreak >nul

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
    pip install -r requirements.txt --quiet --upgrade 2>nul
)
echo [OK] Python venv active

:: ── Replicate API token (cloud image generation) ──
if defined REPLICATE_API_TOKEN (
    echo [OK] Replicate API token set
) else (
    echo [WARN] REPLICATE_API_TOKEN not set. Cloud image generation will not work.
    echo        Get a token at: https://replicate.com/account/api-tokens
    echo        Then set it:    set REPLICATE_API_TOKEN=r8_YourTokenHere
)

:: ── Start ComfyUI ───────────────────────────────
set "COMFYUI_DIR=C:\Dev\ComfyUI"
curl -s http://127.0.0.1:8188/system_stats >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] ComfyUI already running
) else (
    if exist "%COMFYUI_DIR%\venv\Scripts\activate.bat" (
        echo [START] ComfyUI on port 8188...
        start "ComfyUI" cmd /k "cd /d %COMFYUI_DIR% && venv\Scripts\activate.bat && python main.py --listen 127.0.0.1 --port 8188"
        echo Waiting for ComfyUI...
        :wait_comfyui
        curl -s http://127.0.0.1:8188/system_stats >nul 2>&1
        if %errorlevel% neq 0 (
            timeout /t 2 /nobreak >nul
            goto :wait_comfyui
        )
        echo [OK] ComfyUI ready
    ) else (
        echo [WARN] ComfyUI not found at %COMFYUI_DIR%. Image generation will not work.
    )
)

:: ── Ensure SoX is in PATH (required by Qwen TTS) ──
where sox >nul 2>&1
if %errorlevel% neq 0 (
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\ChrisBagwell.SoX*") do (
        if exist "%%D\sox-14.4.2\sox.exe" (
            set "PATH=%%D\sox-14.4.2;%PATH%"
            echo [OK] SoX found via winget
        )
    )
    where sox >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARN] SoX not found. Voice generation will fail.
        echo        Install with: winget install sox
    )
) else (
    echo [OK] SoX found
)

:: ── Start VoiceBox backend ──────────────────────
set "VOICEBOX_DIR=C:\Dev\voice_box\voicebox-temp\voicebox"
curl -s http://localhost:17493/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] VoiceBox already running
) else (
    if exist "%VOICEBOX_DIR%\backend\venv\Scripts\activate.bat" (
        echo [START] VoiceBox on port 17493...
        start "VoiceBox Backend" cmd /k "cd /d %VOICEBOX_DIR% && backend\venv\Scripts\activate.bat && uvicorn backend.main:app --host 127.0.0.1 --port 17493"
        echo Waiting for VoiceBox...
        :wait_voicebox
        curl -s http://localhost:17493/health >nul 2>&1
        if %errorlevel% neq 0 (
            timeout /t 1 /nobreak >nul
            goto :wait_voicebox
        )
        echo [OK] VoiceBox ready
    ) else (
        echo [WARN] VoiceBox not found at %VOICEBOX_DIR%. Voice generation will not work.
    )
)

:: ── Start Backend (with --reload for dev, Jamendo key forwarded) ──
echo [START] Backend on port 8102 (with hot-reload)...
start "StoryTeller Backend (Music)" cmd /k "cd /d %~dp0 && venv\Scripts\activate.bat && set JAMENDO_CLIENT_ID=%JAMENDO_CLIENT_ID% && set REPLICATE_API_TOKEN=%REPLICATE_API_TOKEN% && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8102 --reload"

:: ── Wait for backend ────────────────────────────
echo Waiting for backend...
:wait_backend
curl -s http://127.0.0.1:8102/api/health >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto :wait_backend
)
echo [OK] Backend ready

:: ── Start Frontend ──────────────────────────────
cd /d "%~dp0\frontend"
if not exist "node_modules" (
    echo [SETUP] Installing frontend dependencies...
    call npm install
)
echo [START] Frontend on port 5191...
start "StoryTeller Frontend (Music)" cmd /k "cd /d %~dp0\frontend && npm run dev"

:: ── Open browser ────────────────────────────────
timeout /t 3 /nobreak >nul
start http://localhost:5191

echo.
echo  Story Teller (Background Music Edition) is running!
echo  Frontend:  http://localhost:5191
echo  Backend:   http://localhost:8102
echo  ComfyUI:   http://127.0.0.1:8188
echo  VoiceBox:  http://localhost:17493
echo.
echo  Features: Jamendo auto-select music, local fallback (data/music/), volume control
echo.
echo  Close this window to stop all services.
pause
