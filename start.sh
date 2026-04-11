#!/bin/bash
set -e

export PYTHONPATH="/app:${PYTHONPATH}"
export MPLBACKEND=Agg
export GRADIO_TEMP_DIR=/tmp/gradio_cache

# HuggingFace Spaces only exposes port 7860 publicly.
# FastAPI (OpenEnv /reset /step /health) starts FIRST on port 7860.
# Gradio UI starts AFTER as a background process connecting to FastAPI.
# This ensures /reset responds instantly (evaluator requirement).

PUBLIC_PORT="${PORT:-7860}"
export ENV_BASE_URL="http://localhost:${PUBLIC_PORT}"

echo "=== TradeExecGym Starting on port ${PUBLIC_PORT} ==="

# Step 1: Start Unified Server (FastAPI + Gradio)
# Note: Gradio is mounted at '/' in server/app.py
echo "[1/1] Starting Unified TradeExecGym Server on port ${PUBLIC_PORT}..."
uvicorn server.app:app --host 0.0.0.0 --port "${PUBLIC_PORT}" --log-level warning

