@echo off
cd /d %~dp0
.venv\Scripts\python.exe scripts\check_lerobot_env.py
pause
