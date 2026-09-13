@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "PYTHONPATH=%PROJECT_ROOT%"
set "PYTHONUNBUFFERED=1"
set "OLLAMA_URL=http://127.0.0.1:11434"
set "OLLAMA_MODEL=qwen3-vl:4b"
set "AIRSIM_HOST=127.0.0.1"
set "AIRSIM_PORT=41451"
set "RUNTIME_DIR=%SCRIPT_DIR%runtime"
set "START_LOG=%RUNTIME_DIR%\startup.log"
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
>>"%START_LOG%" echo.
>>"%START_LOG%" echo [%date% %time%] Fly0 startup
echo [Fly0] Script directory: %SCRIPT_DIR%
echo [Fly0] Checking Ollama at %OLLAMA_URL% ...
set "OLLAMA_EXE="
for /f "delims=" %%P in ('where ollama 2^>nul') do if not defined OLLAMA_EXE set "OLLAMA_EXE=%%P"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if defined OLLAMA_EXE (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -Uri '%OLLAMA_URL%/api/tags' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
    if errorlevel 1 (
        echo [Fly0] Ollama is not running; starting ollama serve...
        start "Fly0 Ollama" /min "%OLLAMA_EXE%" serve
    )
) else (
    echo [Fly0] ollama.exe was not found; using an already-running Ollama service if available.
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0;$i -lt 30;$i++){try{$r=Invoke-RestMethod -Uri '%OLLAMA_URL%/api/tags' -TimeoutSec 2; if(@($r.models.name) -contains '%OLLAMA_MODEL%'){$ok=$true;break}}catch{}; Start-Sleep 1}; if(-not $ok){exit 1}"
if errorlevel 1 (
    echo [Fly0][ERROR] Ollama/model %OLLAMA_MODEL% not ready within 30 seconds.
    echo [Fly0] Run: ollama pull %OLLAMA_MODEL%
    pause
    exit /b 11
)
echo [Fly0] Ollama and %OLLAMA_MODEL% are ready.

echo [Fly0] Waiting for AirSim RPC at %AIRSIM_HOST%:%AIRSIM_PORT% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0;$i -lt 60;$i++){if(Test-NetConnection -ComputerName '%AIRSIM_HOST%' -Port %AIRSIM_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue){$ok=$true;break}; Start-Sleep 1}; if(-not $ok){exit 1}"
if errorlevel 1 (
    echo [Fly0][ERROR] AirSim RPC not ready within 60 seconds.
    echo [Fly0] Start UE4 with the AirSim plugin and check port %AIRSIM_PORT%.
    pause
    exit /b 12
)
echo [Fly0] AirSim RPC is ready.

if exist "I:\fly0_py38\Scripts\python.exe" (
    set "PYTHON_EXE=I:\fly0_py38\Scripts\python.exe"
) else (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
    echo [Fly0][ERROR] Python was not found.
    pause
    exit /b 10
)
echo [Fly0] Python: %PYTHON_EXE%
echo [Fly0] Starting Fly0. Press Ctrl+C to stop.
"%PYTHON_EXE%" -u main.py --config config.json --prompt sysprompt\sysprompt.txt
set "FLY0_EXIT=%ERRORLEVEL%"
echo [Fly0] Fly0 exited with code !FLY0_EXIT!.
pause
exit /b !FLY0_EXIT!
