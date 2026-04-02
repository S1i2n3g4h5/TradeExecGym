FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Core pip tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in layers
# Hackathon rule: no server PyTorch.
# (Note: requirements.txt can be passed or we just copy project)
COPY pyproject.toml .
RUN pip install -e .

COPY . .

# HuggingFace standard port
EXPOSE 7860

# We use uvicorn directly on the app factory
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
