$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$versionText = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$parts = $versionText.Split('.')
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 12)) {
    throw "LeRobot 0.6.1 requires Python 3.12 or newer. Current: $versionText"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
& .\.venv\Scripts\python.exe -m pip install -e .\robot_plugin

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next: double-click 01_check_env.bat"
