@echo off
title Handwritten Notes to Digital Text Converter - Streamlit
echo ======================================================================
echo Starting Streamlit Handwritten Notes OCR Application...
echo ======================================================================

cd /d "%~dp0"

call "C:\Users\Tuhid Khan\miniconda3\Scripts\activate.bat" "C:\Users\Tuhid Khan\miniconda3\envs\my_env"

streamlit run app.py
pause
