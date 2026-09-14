@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv Python was not found.
  pause
  exit /b 1
)

echo DESKRO real-style gaze controller preview
echo NO RL / NO PPO
".venv\Scripts\python.exe" "scripts\check_real_gaze_controller.py"
pause
