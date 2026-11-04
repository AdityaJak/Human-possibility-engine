# Human Possibility Engine — one-command developer workflow.
# Every target runs without API keys. `make help` lists targets.

PY ?= .venv/bin/python
UV ?= uv
NPM ?= npm
WEB := apps/web
WORKERS ?= 3
UV_VENV_FLAGS ?= --allow-existing

.DEFAULT_GOAL := help
.PHONY: help setup setup-py setup-web lint format typecheck test test-fast test-slow test-integration \
        contracts contracts-check openapi openapi-check smoke experiments demo-data api web demo demo-check \
        bench bench-1k bench-10k docker-config docker-build secrets-scan audit clean

help: ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

setup: setup-py setup-web ## Install Python (uv, .venv) + web (npm ci) dependencies
setup-py:
	$(UV) venv -q $(UV_VENV_FLAGS) .venv
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

test-slow: ## Only slow tests (full lifespan / 1,000-agent runs)
	$(PY) -m pytest -q -m slow

test-integration: ## API + real World integration tests
	$(PY) -m pytest -q -m integration

contracts: ## Regenerate JSON Schema + TypeScript contracts
	$(PY) scripts/export_contracts.py
contracts-check: ## Fail if generated contracts are stale
	$(PY) scripts/export_contracts.py --check

openapi: ## Regenerate docs/openapi.json
	$(PY) apps/api/export_openapi.py
openapi-check: ## Fail if docs/openapi.json is stale
	$(PY) apps/api/export_openapi.py --check

smoke: ## Seeded 25-agent x 3-year simulation, run twice, replay hashes must match (CI)
	$(PY) -m hpe.cli smoke --out data/smoke

experiments: ## Run every preregistered simulation experiment and regenerate reports (long: hours on 4 CPUs)
	$(PY) -m hpe.evaluation run $$($(PY) -c "from hpe.evaluation.spec import simulation_spec_files as f; print(' '.join(map(str, f())))") --workers $(WORKERS)

human-experiments: ## Human-data experiments (downloads Psych-101 + HeartSteps V1, or set HPE_PSYCH101_SUBSET)
	$(PY) -m hpe.evaluation.human run experiments/exp07b_causal_grounding.yaml
	$(PY) -m hpe.evaluation.human run experiments/exp07_human_heldout.yaml

demo-data: ## Produce the stored 100-agent x 80-year demo run used by offline demo mode
	$(PY) -m hpe.cli demo-run

api: ## Start the API on :8000
	PYTHONPATH=packages:apps/api $(PY) -m uvicorn hpe_api.main:app --host 127.0.0.1 --port 8000

web: ## Start the web dev server on :5173
	cd $(WEB) && $(NPM) run dev

demo: ## Full local stack: API + built web app (Ctrl-C to stop); OFFLINE=1 for no-network mode
	$(PY) scripts/demo.py $(if $(OFFLINE),--offline,) $(DEMO_ARGS)

demo-check: ## Headless full-stack smoke: start API + web, walk the demo endpoints, shut down
	$(PY) scripts/demo.py --check $(DEMO_ARGS)

bench: ## Scale benchmark (AGENTS=100 YEARS=2 POLICY=developmental) -> reports/scale/*.json
	$(PY) scripts/bench_scale.py --agents $(or $(AGENTS),100) --years $(or $(YEARS),2) --policy $(or $(POLICY),developmental)

bench-1k: ## 1,000 agents x 10 years benchmark (run on a quiet machine)
	$(PY) scripts/bench_scale.py --agents 1000 --years 10 --policy developmental --timeout 7200

bench-10k: ## 10,000 agents x 1 year benchmark with profile (run on a quiet machine)
	$(PY) scripts/bench_scale.py --agents 10000 --years 1 --policy developmental --timeout 3600 --profile

docker-config: ## Validate docker-compose.yml (no daemon needed)
	docker compose config --quiet

docker-build: ## Build API + web images (needs a Docker daemon)
	docker compose build

secrets-scan: ## Scan git-visible files for credentials
	$(PY) scripts/scan_secrets.py

audit: ## Dependency vulnerability audit (Python locked deps + npm production deps)
	mkdir -p data
	$(UV) export --frozen --no-dev --no-emit-project --format requirements-txt -o data/requirements.lock.txt
	uvx pip-audit==2.9.0 --disable-pip --require-hashes -r data/requirements.lock.txt
	cd $(WEB) && $(NPM) audit --omit=dev --audit-level=high

clean: ## Remove runtime data and caches (never touches reports/)
	rm -rf data .pytest_cache .mypy_cache .ruff_cache $(WEB)/dist
