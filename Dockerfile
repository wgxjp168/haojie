# syntax=docker/dockerfile:1.6

# ---------- base ----------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Runtime libs required by Pillow + ffmpeg for audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ---------- deps ----------
FROM base AS deps

# Compilers needed only during install; stripped from the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------- production ----------
FROM base AS production

# Copy installed packages from deps stage.
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --home /app --shell /bin/bash app

COPY --chown=app:app . /app
RUN mkdir -p /app/storage /app/logs && chown -R app:app /app/storage /app/logs

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--max-requests", "10000", \
     "--max-requests-jitter", "1000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
