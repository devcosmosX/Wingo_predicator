#!/bin/bash
echo "============================================="
echo "  WinGo 30S Predictor — Starting Services"
echo "============================================="

# Check if model exists
if [ ! -f "wingo_model/saved.pkl" ] && [ ! -f "wingo_model.pkl" ]; then
    echo "[!] No trained model found."
    echo "[!] Run: python train.py"
    echo "[!] Or start anyway (will use RL-only mode)"
fi

echo ""
echo "[1] Starting Web Server on http://localhost:8000"
echo "[2] Open http://localhost:8000 in browser"
echo "[3] Press Ctrl+C to stop"
echo ""

python server.py
