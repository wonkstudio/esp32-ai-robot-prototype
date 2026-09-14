@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv Python was not found.
  pause
  exit /b 1
)

echo Testing realistic smooth servo motion. No PPO policy is used.
".venv\Scripts\python.exe" "scripts\check_smooth_servo.py"
pause
