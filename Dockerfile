# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl for fetching the embeddings tarball at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer caches across code edits.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (only what the API serves; ingestion + Streamlit excluded
# via .dockerignore).
COPY api.py bible_rag.py bible_graph.py ./

# Small artifacts ship via the upload (railway up / local docker context).
COPY data/verses.parquet data/graph.json ./data/

# The 191 MB embeddings file is too large for Railway's snapshot init limit,
# so we fetch it during the build from a GitHub Release attachment.
ARG EMBEDDINGS_URL=https://github.com/yongjinjiang/kjv-graphrag-chat/releases/download/data-v1/verse_embeddings.npy
RUN curl -fsSL "$EMBEDDINGS_URL" -o data/verse_embeddings.npy \
    && ls -lh data/

# Railway / Render / Fly set $PORT; default to 8000 for local docker run.
ENV PORT=8000
EXPOSE 8000

# --proxy-headers trusts X-Forwarded-For from Railway's edge proxy so the
# per-IP rate limiter (slowapi, keyed on the client address) sees real
# client IPs instead of bucketing every user under the proxy's IP.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
