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

# ── Startup script ─────────────────────────────────────────────────────────────
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Expose public port ─────────────────────────────────────────────────────────
# Primary port: FastAPI (OpenEnv /reset /step /health) + Gradio UI at /ui
# HF Spaces only exposes port 7860 publicly
EXPOSE 7860

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["/app/start.sh"]
