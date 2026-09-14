@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv Python was not found.
  pause
  exit /b 1
)

echo Running the neck physics test. No PPO policy is used.
".venv\Scripts\python.exe" "scripts\check_neck_stability.py"
pause
