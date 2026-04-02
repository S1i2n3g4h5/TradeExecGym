FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add project root to PYTHONPATH for reliable imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install uv for high-speed dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the entire project
COPY . .

# Install the project and dependencies using uv
# --system tells uv to install into the container's global python environment
RUN uv pip install --system --no-cache -e .

# Create the startup multi-process script
RUN echo '#!/bin/bash\n\
export PYTHONPATH="/app:${PYTHONPATH}"\n\
echo "Starting TradeExecGym Backend (FastAPI)..."\n\
# Run uvicorn in the background\n\
uvicorn server.app:app --host 0.0.0.0 --port 7860 &\n\
\n\
echo "Starting TradeExecGym Dashboard (Gradio)..."\n\
# Run Gradio on the port provided by HF Spaces (default 7860)\n\
python ui/app.py --port ${PORT:-7860}\n\
' > start.sh && chmod +x start.sh

# HF Spaces expects the primary app on port 7860
EXPOSE 7860

# Start the environment
CMD ["./start.sh"]
