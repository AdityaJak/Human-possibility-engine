# Human Possibility Engine

> **Synthetic research prototype; not validated for predicting human futures.**
> Mission: give every human the power to make better decisions for their future self — grounded in science, not certainty theater.

This repository is under active construction. See [`BUILD_STATUS.md`](BUILD_STATUS.md) for exactly what exists, what passed, and what does not yet work. Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). Contributor lanes: [`AGENTS.md`](AGENTS.md). Claims policy: [`docs/CLAIMS_POLICY.md`](docs/CLAIMS_POLICY.md).

## Quick start (Python foundation)

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,parquet]"
.venv/bin/python -m pytest -q
```
