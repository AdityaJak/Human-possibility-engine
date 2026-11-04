# Human Possibility Engine — one-command developer workflow.
# All targets run without API keys.

PY ?= .venv/bin/python
UV ?= uv
NPM ?= npm
WEB := apps/web

.PHONY: help setup setup-py setup-web lint format typecheck test test-fast contracts contracts-check \
        smoke experiments demo-data api web demo clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

setup: setup-py setup-web ## Install Python + web dependencies
setup-py:
	$(UV) venv -q .venv || true
	$(UV) pip install -q --python $(PY) -e ".[dev,parquet]"
setup-web:
	cd $(WEB) && $(NPM) ci --no-audit --no-fund

lint: ## Ruff lint + format check (Python), ESLint (web)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	cd $(WEB) && $(NPM) run lint

format: ## Auto-format Python
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

typecheck: ## mypy + tsc
	$(PY) -m mypy
	cd $(WEB) && $(NPM) run typecheck

test: ## All Python tests (incl. slow) + web tests
	$(PY) -m pytest -q
	cd $(WEB) && $(NPM) test

test-fast: ## Python tests excluding slow
	$(PY) -m pytest -q -m "not slow"

contracts: ## Regenerate JSON Schema + TypeScript contracts
	$(PY) scripts/export_contracts.py
contracts-check:
	$(PY) scripts/export_contracts.py --check

smoke: ## Short seeded simulation (25 agents, 3 years) — used by CI
	$(PY) -m hpe.cli smoke

experiments: ## Run all preregistered experiments and regenerate reports
	$(PY) -m hpe.cli experiments --all

demo-data: ## Produce the stored 100-agent demo run used by offline demo mode
	$(PY) -m hpe.cli demo-run

api: ## Start the API on :8000
	$(PY) -m uvicorn hpe_api.main:app --host 127.0.0.1 --port 8000

web: ## Start the web dev server on :5173
	cd $(WEB) && $(NPM) run dev

demo: ## Full local stack: API + web (Ctrl-C to stop)
	$(PY) scripts/demo.py

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache $(WEB)/dist
