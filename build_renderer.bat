@echo off
rem Freeze the quality-renderer sidecar into dist\renderer\ (wrapper around build_renderer.py).
rem   build_renderer.bat          build, then self-check the frozen sidecar in fake mode
rem   build_renderer.bat --test   self-check an existing dist\renderer without rebuilding
rem The driver runs under the APP venv; it invokes PyInstaller with renderer\.venv (torch lives
rem only there, and the app exe must stay torch-free).
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo build_renderer.bat: .venv\Scripts\python.exe not found - create the app venv first.
    exit /b 1
)
if not exist "renderer\.venv\Scripts\python.exe" (
    echo build_renderer.bat: renderer\.venv\Scripts\python.exe not found - run renderer\install.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" build_renderer.py %*
exit /b %ERRORLEVEL%
