# syntax=docker/dockerfile:1.7
# Human Possibility Engine API (FastAPI). Build from the repo root:
#   docker build -f infra/docker/api.Dockerfile -t hpe-api:0.1.0 .
# Runs without any API key. Synthetic research prototype; not validated for predicting human futures.
#
# Production notes (Render / Fly / Railway / any PaaS):
#   * $PORT is honoured (default 8000); uvicorn trusts X-Forwarded-* from the platform proxy so
#     per-client rate limits see the real client address.
#   * Set HPE_READ_ONLY=1 for public exposure: simulation/experiment starters answer 405 and the
#     research lab serves the committed stored real outputs baked into the image at /app/reports.
#   * Mount a persistent disk at /data and set HPE_DATA_DIR=/data to keep pseudonymous Future Self
#     demo records across deploys; without a disk, use HPE_DATA_DIR=/app/data (ephemeral).
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HPE_DATA_DIR=/app/data \
    HPE_REPORTS_DIR=/app/reports \
    HPE_LLM_PROVIDER=none \
    PORT=8000

RUN pip install --no-cache-dir "uv==0.11.32"

WORKDIR /app
# Dependency + project layer (pinned by uv.lock).
COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
COPY apps/api ./apps/api
RUN uv sync --frozen --no-dev --no-editable

# Committed real outputs served in offline/read-only mode (compose mounts ./reports read-only over this).
COPY reports ./reports

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin hpe \
    && mkdir -p /app/data /data \
    && chown -R hpe:hpe /app/data /data
USER hpe

EXPOSE 8000
# Health check follows $PORT so it stays valid on platforms that assign the port.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["/bin/sh", "-c", "/app/.venv/bin/python -c \"import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4).status == 200 else 1)\""]

# Shell form so $PORT expands; exec keeps uvicorn as PID 1 for clean signal handling.
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/uvicorn hpe_api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --forwarded-allow-ips='*'"]
