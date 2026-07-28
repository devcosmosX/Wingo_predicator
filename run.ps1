Write-Host "============================================="
Write-Host "  WinGo 30S Predictor — Starting Services"
Write-Host "============================================="

if (-not (Test-Path "wingo_model") -and -not (Test-Path "wingo_model.pkl")) {
    Write-Host "[!] No trained model found."
    Write-Host "[!] Run: python train.py"
    Write-Host "[!] Or start anyway (will use RL-only mode)"
}

Write-Host ""
Write-Host "[1] Starting Web Server on http://localhost:8000"
Write-Host "[2] Open http://localhost:8000 in browser"
Write-Host "[3] Press Ctrl+C to stop"
Write-Host ""

python server.py
