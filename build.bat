@echo off
rem One-command Windows build of GlassTranslate.exe (wrapper around build.py).
rem   build.bat            build + smoke test
rem   build.bat --no-test  build only
rem   build.bat --test     smoke-test an existing dist\GlassTranslate.exe
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo build.bat: .venv\Scripts\python.exe not found - create the venv and install requirements.txt + requirements-build.txt first.
    exit /b 1
)
".venv\Scripts\python.exe" build.py %*
exit /b %ERRORLEVEL%
