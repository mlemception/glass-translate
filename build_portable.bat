@echo off
rem Assemble the two portable zips (wrapper around build_portable.py).
rem   build_portable.bat            build dist\GlassTranslate-<v>-portable-win64.zip + -models-win64.zip
rem   build_portable.bat --dry-run  verify the store and list what would be zipped, write nothing
rem   build_portable.bat --stage-only  refresh build\portable\ (licence index included) only
rem   build_portable.bat --no-research-models  leave out the research-licensed Sugoi pack
rem   build_portable.bat --single   write one dist\GlassTranslate-<v>-full-win64.zip (program + models, same layout) instead of the pair
rem Needs dist\GlassTranslate.exe (build.bat) and dist\renderer\glassrenderer.exe (build_renderer.bat).
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo build_portable.bat: .venv\Scripts\python.exe not found - create the venv and install requirements.txt + requirements-build.txt first.
    exit /b 1
)
".venv\Scripts\python.exe" build_portable.py %*
exit /b %ERRORLEVEL%
