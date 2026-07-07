@echo off
REM HivisionIDPhotos quick start (enables GPU)
REM Fix: cuDNN 9 not in PATH causes ORT to fallback to CPU

setlocal

cd /d "%~dp0"

set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
set "CUDA_BIN_X64=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\x64"
set "PATH=%CUDA_BIN_X64%;%CUDA_BIN%;%PATH%"

REM 直接用 venv python 绝对路径, 不依赖 activate (activate 在某些环境下不覆盖 PATH)
"%~dp0venv\Scripts\python.exe" -u app.py --port 7860 --host 127.0.0.1

endlocal