Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$KeyPath = "competition_secrets\gee-service-account-key.json",
    [switch]$SkipInstall
)

function Get-PythonLauncher {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    throw "Python launcher not found. Install Python 3.10+ so 'py' or 'python' is available."
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$websiteDir = Join-Path $repoRoot "website"
$backendDir = Join-Path $websiteDir "backend"
$sourceKey = Join-Path $repoRoot $KeyPath
$targetKey = Join-Path $backendDir "gee-service-account-key.json"

if (-not (Test-Path $sourceKey)) {
    Write-Host ""
    Write-Host "Missing Earth Engine key file." -ForegroundColor Red
    Write-Host "Expected at: $sourceKey"
    Write-Host "Place the competition key there, then run this script again."
    exit 1
}

Write-Host "Copying Earth Engine key to backend runtime path..."
Copy-Item -Path $sourceKey -Destination $targetKey -Force

Push-Location $websiteDir
try {
    $launcher = Get-PythonLauncher

    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Write-Host "Creating virtual environment..."
        & $launcher -m venv .venv
    }

    $venvPython = ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment python not found at $venvPython"
    }

    if (-not $SkipInstall) {
        Write-Host "Installing backend dependencies..."
        & $venvPython -m pip install -r requirements.txt
    }

    Write-Host ""
    Write-Host "Starting website backend at http://127.0.0.1:8000" -ForegroundColor Green
    & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
