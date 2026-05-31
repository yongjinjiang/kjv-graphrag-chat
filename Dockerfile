# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies first so this layer caches across code edits.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (only what the API serves; ingestion + Streamlit excluded
# via .dockerignore).
COPY api.py bible_rag.py bible_graph.py ./

# Pre-built artifacts. .dockerignore intentionally allows these even though
# they're gitignored, so `railway up` / `docker build` bundle them.
COPY data/verses.parquet data/verse_embeddings.npy data/graph.json ./data/

# Railway / Render / Fly set $PORT; default to 8000 for local docker run.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
