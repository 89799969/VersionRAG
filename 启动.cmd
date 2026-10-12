@echo off
setlocal
chcp 65001 >nul
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project Python environment is missing. See README.md.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\start_project.py"
exit /b %errorlevel%
