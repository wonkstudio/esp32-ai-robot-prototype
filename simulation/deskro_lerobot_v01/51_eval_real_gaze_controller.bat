@echo off
cd /d "%~dp0"

".venv\Scripts\python.exe" "scripts\evaluate_real_gaze_controller.py"
pause
