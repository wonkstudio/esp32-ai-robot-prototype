@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv Python was not found.
  pause
  exit /b 1
)

echo Opening DESKRO with a short but visible neck...
".venv\Scripts\python.exe" "scripts\view_square_tv.py"
pause
