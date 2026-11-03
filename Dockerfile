# Production image.
#
# Build:  docker build -t yt-summarizer .
# Run:    docker run -p 8000:8000 --env-file .env yt-summarizer
#
# Note on --workers 1: conversations live in an in-process store, so multiple
# worker *processes* would each see a different set of sessions. Concurrency
# comes from threads instead. Move to Redis (sessions.SessionStore) before
# scaling to multiple workers.
#
# Mount a volume for the cache, or every redeploy starts cold:
#   -v yt-cache:/app/data -e CACHE_DB=/app/data/cache.sqlite3
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so this layer is cached across source changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# Served by gunicorn, not the Werkzeug development server.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "180", "--access-logfile", "-", "app:create_app()"]
