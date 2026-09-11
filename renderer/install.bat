@echo off
rem Create / refresh the quality-renderer sidecar venv (renderer\.venv).
rem
rem This venv is deliberately separate from the app's .venv: the main app and
rem the PyInstaller exe must never import torch.  Run it from anywhere; it
rem always works on its own directory.
setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo [1/4] creating %VENV% with Python 3.13
    py -3.13 -m venv "%VENV%"
    if errorlevel 1 goto :fail
) else (
    echo [1/4] reusing %VENV%
)

echo [2/4] upgrading pip
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/4] installing torch 2.14.0+cu130 / torchvision 0.29.0+cu130
"%PY%" -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :fail

echo [4/4] installing the rest of requirements.txt
"%PY%" -m pip install -r "%HERE%requirements.txt"
if errorlevel 1 goto :fail

echo.
echo --- torch / CUDA check ---
"%PY%" -c "import torch;print('torch', torch.__version__);print('cuda available:', torch.cuda.is_available());print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if errorlevel 1 goto :fail

echo.
echo Sidecar venv ready: %VENV%
echo Models are downloaded by the app (Engines page) into models\quality.
endlocal
exit /b 0

:fail
echo.
echo install.bat FAILED (see the error above)
endlocal
exit /b 1
