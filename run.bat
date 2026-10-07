@echo off
cd /d "%~dp0"
title RAG Chatbot
".venv\Scripts\python.exe" -m streamlit run app.py
pause
