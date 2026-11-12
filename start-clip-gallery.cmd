@echo off
setlocal
"%~dp0venv\Scripts\python.exe" "%~dp0scripts\clip_gallery.py"
if errorlevel 1 pause
