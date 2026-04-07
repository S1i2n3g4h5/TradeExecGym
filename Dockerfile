FROM python:3.11-slim

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
# Headless matplotlib — no display required in container
ENV MPLBACKEND=Agg
# Add project root to PYTHONPATH for reliable imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (pinned for reproducible builds) ───────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .

# ── Create writable runtime directories (HF Spaces runs as UID 1000) ──────────
RUN mkdir -p /app/results /app/models /tmp/gradio_cache \
    && chmod -R 777 /app/results /app/models /tmp/gradio_cache

# ── Install dependencies via uv ────────────────────────────────────────────────
# torch is large (~2GB) — install CPU-only variant to keep image lean & fast
RUN uv pip install --system --no-cache \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache -e .

# ── Startup script (use cat heredoc — echo mishandles \n in some shells) ──────
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

export PYTHONPATH="/app:${PYTHONPATH}"
export MPLBACKEND=Agg
export GRADIO_TEMP_DIR=/tmp/gradio_cache

echo "=== TradeExecGym Starting ==="

# Start backend (FastAPI + MCP) on internal port 7865
echo "Starting backend on port 7865..."
uvicorn server.app:app --host 0.0.0.0 --port 7865 --log-level warning &
BACKEND_PID=$!

# Wait for backend to be ready (up to 30s)
echo "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:7865/health > /dev/null 2>&1; then
        echo "Backend ready after ${i}s"
        break
    fi
    sleep 1
done

# Start Gradio dashboard on public port (HF Spaces injects $PORT, default 7860)
echo "Starting Gradio dashboard on port ${PORT:-7860}..."
python ui/app.py --port "${PORT:-7860}" --host 0.0.0.0

# If Gradio exits, kill backend too
kill $BACKEND_PID 2>/dev/null || true
EOF
RUN chmod +x /app/start.sh

# ── Expose public port ─────────────────────────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["/app/start.sh"]
