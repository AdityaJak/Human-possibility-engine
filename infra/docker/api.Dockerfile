# syntax=docker/dockerfile:1.7
# Human Possibility Engine API (FastAPI). Build from the repo root:
#   docker build -f infra/docker/api.Dockerfile -t hpe-api:0.1.0 .
# Runs without any API key. Synthetic research prototype; not validated for predicting human futures.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HPE_DATA_DIR=/app/data \
    HPE_REPORTS_DIR=/app/reports \
    HPE_LLM_PROVIDER=none

RUN pip install --no-cache-dir "uv==0.11.32"

WORKDIR /app
# Dependency + project layer (pinned by uv.lock).
COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
COPY apps/api ./apps/api
RUN uv sync --frozen --no-dev --no-editable

# Committed real outputs served in offline demo mode (compose mounts ./reports read-only over this).
COPY reports ./reports

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin hpe \
    && mkdir -p /app/data \
    && chown -R hpe:hpe /app/data
USER hpe

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["/app/.venv/bin/python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["/app/.venv/bin/uvicorn", "hpe_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
