# One-shot setup and launch for Windows PowerShell.
#   .\scripts\run.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment ..." -ForegroundColor Cyan
    python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
}

if (-not (Test-Path "data\derived\cadastre.json")) {
    Write-Host "Building the demonstration dataset ..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe scripts\build_demo.py
}

Write-Host "Starting Bhoomi3D ..." -ForegroundColor Green
& .\.venv\Scripts\python.exe scripts\serve.py
