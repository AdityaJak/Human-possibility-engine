# Infrastructure and release

Everything here runs without API keys.

## Local stack (no Docker)

```bash
make setup          # uv venv + editable install, npm ci
make demo           # API on http://127.0.0.1:8000, web app on http://127.0.0.1:5173 (Ctrl-C to stop)
make demo OFFLINE=1 # no-network mode: never runs npm install, forces HPE_LLM_PROVIDER=none
make demo-check     # headless: start API + built web app, walk the demo endpoints, shut down (CI)
make demo-check DEMO_ARGS="--api-port 8790 --web-port 5190"   # if 8000/5173 are taken
```

`scripts/demo.py` refuses to start when a port is already in use, so its checks can never talk to a
different server. `--check` uses a throw-away data directory.

## Docker

```bash
make docker-config  # validate docker-compose.yml (no daemon needed)
make docker-build   # build hpe-api:0.1.0 and hpe-web:0.1.0 (needs a Docker daemon)
docker compose up   # API :8000, web :5173
```

* `infra/docker/api.Dockerfile`: python:3.12-slim, `uv sync --frozen --no-dev --no-editable` from
  `uv.lock`, non-root uid 10001, healthcheck on `/health`, committed `reports/` baked in (compose
  mounts `./reports` read-only over it), runtime data in the `hpe-data` named volume.
* `infra/docker/web.Dockerfile`: node:22-slim build stage (`VITE_API_BASE_URL` build arg) →
  `nginxinc/nginx-unprivileged:1.27-alpine` serving the SPA on 8080 with an index.html fallback.
* Per-Dockerfile ignore files (`*.Dockerfile.dockerignore`, BuildKit) keep `.venv`, `node_modules`,
  `data/` and `.env` out of the build context.

Status (2026-11-04): `docker compose config` validates; the images have **not** been built in the
development container (no Docker daemon there). The API image's dependency step
(`uv sync --frozen --no-dev --no-editable` on Python 3.12) was verified outside Docker.

## Public hosting

See `docs/DEPLOYMENT.md`: GitHub Pages (static, automatic), Render Blueprint (`render.yaml`, API in
read-only mode, `HPE_READ_ONLY=1`), Vercel (`apps/web/vercel.json`). The API image honours `$PORT`,
trusts the platform proxy's forwarded headers, and creates `/data` for a persistent disk
(`HPE_DATA_DIR=/data`). No deployment exists until the founder connects the accounts.

## Benchmarks

`scripts/bench_scale.py` → `reports/scale/<id>.json` (see `reports/scale/README.md`).

## CI (`.github/workflows/ci.yml`)

| Job | Trigger | What |
|---|---|---|
| python | push, PR | ruff, mypy, contracts + OpenAPI freshness, fast unit tests with coverage, integration tests (API + real World), seeded smoke (replay hashes must match), 25-agent benchmark smoke artifact |
| web | push, PR | eslint, tsc, vitest, vite build |
| demo-check | after python + web | `make setup` + `make demo-check` from a clean checkout |
| security | push, PR | `scripts/scan_secrets.py`, `pip-audit` on `uv.lock`-exported pins, `npm audit --omit=dev --audit-level=high` |
| slow | nightly + manual | `pytest -m slow` (full 100-agent × 80-year lifespan, 1,000-agent run) |

Branch protection is a repository setting and is not configured from code.
