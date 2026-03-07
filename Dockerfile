# ── Base Image ────────────────────────────────────────────────────────────────
# Python 3.11 slim: smaller image than full python:3.11, has all we need
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# build-essential: needed to compile some Python packages (faiss-cpu)
# git: needed by some sentence-transformers downloads
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first for Docker layer caching
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Models directory ──────────────────────────────────────────────────────────
# The 'models' directory is populated by running build.py
# In production, you'd mount this as a volume or copy pre-built models
RUN mkdir -p models

# ── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Startup command ──────────────────────────────────────────────────────────
# --host 0.0.0.0 makes the server accessible outside the container
# --port 8000 matches the EXPOSE above
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]