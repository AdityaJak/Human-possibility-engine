# AGENTS.md — lanes, ownership, and merge rules

This repository is built by several coordinated Claude Code workers (spec Part IX). Every worker
reads this file first. The orchestrator (Chief Architect) owns this file.

## 0. Prime directive: we are not building an LLM wrapper

* Tier 1 (world) and Tier 2 (learning) run with **no API keys and no language-model calls**.
* Skill is only what `hpe.tasks.assess` measures on **held-out** symbols with learning disabled.
  Self-reported confidence, biography text, memory size or prompt length are never skill.
* Never hard-code an outcome the experiment is supposed to measure (e.g. "tutored agents get +0.1
  accuracy"). Interventions change *inputs/mechanisms* (feedback quality, episode counts, access);
  outcomes must emerge through learning and be measured.
* Never fabricate results, citations, URLs, benchmark numbers, or test outcomes. Report failures.
  See `docs/CLAIMS_POLICY.md`.

## 1. Lanes and file ownership

| Lane / branch | Owner role | Owns (write access) | Must not edit |
|---|---|---|---|
| `orchestrator` (main dev branch) | Chief Architect | `packages/hpe/contracts/`, `packages/hpe/core/`, `packages/hpe/tasks/`, `AGENTS.md`, `ARCHITECTURE.md`, `docs/INTERFACES.md`, `docs/CLAIMS_POLICY.md`, `docs/ADRS/`, `tasks/board.yaml`, `BUILD_STATUS.md` | — |
| `research/development` | Developmental Learning Researcher | `packages/hpe/development/`, `tests/development/` | contracts, other lanes |
| `sim/world` | Society / World Engineer | `packages/hpe/world/`, `tests/world/` | contracts, other lanes |
| `science/evaluation` | Experimental Science Engineer | `packages/hpe/evaluation/`, `experiments/`, `tests/evaluation/`, `reports/experiments/` | contracts, the developmental learner |
| `platform/backend` | Backend / Data Engineer | `apps/api/`, `packages/hpe/storage/`, `packages/hpe/futures/`, `tests/api/`, `tests/storage/`, `tests/futures/` | contracts, world, learners |
| `product/frontend` | Frontend / Product Engineer | `apps/web/` (except `apps/web/src/types/contracts.ts`, which is generated) | python packages |
| `infra/release` | Infrastructure / Release | `.github/`, `infra/`, `Makefile`, `docker-compose.yml`, `scripts/` (except `export_contracts.py`), `reports/scale/` | package internals |
| `science/human-validation` | Experimental Science (specialist) | `packages/hpe/evaluation/human/`, `datasets/adapters/`, `tests/evaluation/human/`, `reports/human_validation/` | learners |
| `audit/review` | Independent Critic / QA | `reports/audit/`, may open fix requests in `tasks/board.yaml` comments | production code it reviews |

Shared tests in `tests/integration/` are owned by the orchestrator.

## 2. Contracts are frozen

`hpe.contracts` (runtime dataclasses, `Policy` protocol, Pydantic schemas, API models, World/Sink
protocols, registry) is **v0.1.0 and frozen**. If you need a change, do not edit it: write the
request in your handoff (`what`, `why`, `backward compatible?`). The orchestrator decides via ADR.
Generated files (`packages/hpe/contracts/schema/*.json`, `apps/web/src/types/contracts.ts`) come from
`python scripts/export_contracts.py` only.

Plug-in points:

* Policies: implement `hpe.contracts.runtime.Policy` (subclass `BasePolicy`), pass
  `hpe.contracts.conformance.run_policy_conformance`, and call
  `hpe.contracts.registry.register("<policy_class>", builder)` at import time of your package.
* World: implement `hpe.contracts.protocols.WorldRunner` as `hpe.world.World`.
* Sinks: implement `hpe.contracts.protocols.EventSink`.
* Skill measurement: always `hpe.tasks.assess` / `hpe.tasks.transfer_probe`. Practice: `hpe.tasks.practice`.

## 3. Engineering rules for every lane

1. Python ≥3.12, `numpy`, `pydantic` v2, stdlib. Ask before adding a dependency; prefer none.
2. Determinism: all randomness from `hpe.core.rng` named streams. No `random`, no unseeded numpy,
   no wall-clock-dependent behaviour (wall time may be *reported*, never used in logic). Iterate
   dicts/sets in sorted order where order affects results.
3. Tests live in your `tests/<lane>/` folder. Fast tests (<30 s total per lane) by default; mark
   long ones `@pytest.mark.slow`.
4. Before handing off, run from the repo root and paste the real output into your handoff:
   `ruff check . && ruff format --check . && mypy && pytest -q` (frontend: `npm run lint && npm run typecheck && npm test && npm run build` in `apps/web`).
5. Commit on your lane branch with clear messages (`<lane>: <what>`). Do not push; the orchestrator
   merges in dependency order: contracts → tasks → development/evaluation-baselines → world →
   evaluation-experiments → storage/api → web → infra.
6. Handoff note (final message): files added, public API, test command + actual output, known
   gaps, and any contract-change requests. Never say "done" for code that does not run.

## 4. Environment

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,parquet]"   # per worktree
.venv/bin/python -m pytest -q
```

`pytest` adds `packages/` and `apps/api/` to `sys.path` (see `pyproject.toml`), so each git worktree
tests its own code even without reinstalling.
