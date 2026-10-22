@echo off
title Handwritten Notes to Digital Text Converter
echo ======================================================================
echo Starting Handwritten Notes to Digital Text Converter Server...
echo ======================================================================

cd /d "%~dp0"

REM Activate conda environment and run unified FastAPI server
call "C:\Users\Tuhid Khan\miniconda3\Scripts\activate.bat" "C:\Users\Tuhid Khan\miniconda3\envs\my_env"

echo Opening browser at http://localhost:8000 ...
start "" "http://localhost:8000"

python api.py
pause
