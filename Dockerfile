FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

COPY . .

RUN mkdir -p /app/results /app/models /tmp/gradio_cache \
    && chmod -R 777 /app/results /app/models /tmp/gradio_cache

# Runtime image excludes training-only packages such as torch and stable-baselines3.
RUN uv pip install --system --no-cache -e .

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["/app/start.sh"]
