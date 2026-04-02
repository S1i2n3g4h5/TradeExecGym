FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv pip install --system --no-cache -e .

# Create a startup script to run both FastAPI (Backend) and Gradio (Frontend)
RUN echo '#!/bin/bash\n\
# Start FastAPI backend in the background\n\
uvrun uvicorn server.app:app --host 0.0.0.0 --port 7860 &\n\
\n\
# Start Gradio dashboard\n\
# HF Spaces provides the PORT env var for the main entrypoint\n\
python ui/app.py --port ${PORT:-7860}\n\
' > start.sh && chmod +x start.sh

# HF Spaces expects the app on port 7860 by default
EXPOSE 7860

CMD ["./start.sh"]
