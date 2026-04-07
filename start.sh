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

# Step 1: Start FastAPI (lean, no Gradio) on public port - starts in <2s
echo "[1/2] Starting FastAPI (OpenEnv server) on port ${PUBLIC_PORT}..."
uvicorn server.app:app --host 0.0.0.0 --port "${PUBLIC_PORT}" --log-level warning &
FASTAPI_PID=$!

# Step 2: Wait for FastAPI to be ready (up to 30s)
echo "[1/2] Waiting for FastAPI to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PUBLIC_PORT}/health" > /dev/null 2>&1; then
        echo "[1/2] FastAPI ready after ${i}s -- POST /reset and /step available"
        break
    fi
    sleep 1
done

# Step 3: Start Gradio UI in background (takes 10-15s due to PyTorch/model loading)
# Gradio connects to FastAPI internally via ENV_BASE_URL
if [ "${INFERENCE_MODE:-false}" != "true" ]; then
    echo "[2/2] Starting Gradio UI in background (connecting to FastAPI on ${PUBLIC_PORT})..."
    python ui/app.py --port 7861 --host 0.0.0.0 &
    GRADIO_PID=$!
    echo "[2/2] Gradio loading (may take 10-15s)... available at port 7861 when ready"
else
    echo "[2/2] INFERENCE_MODE=true -- Gradio UI skipped"
fi

# Keep FastAPI running (it's the critical process)
wait $FASTAPI_PID
