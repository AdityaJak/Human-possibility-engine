# BUILD_STATUS

_Last updated: 2026-11-04 — 4 of 6 parallel lanes merged; world + human-validation lanes still running; full-budget learning-lab experiments executing._

Legend: **implemented** (code exists) · **tested** (tests pass in integrated branch) · **partial** · **not started**

| Deliverable | Status | Evidence / location |
|---|---|---|
| Contracts v0.1 (schemas, Policy protocol, World/Sink protocols, registry, JSON Schema + TS export) | implemented, tested | `packages/hpe/contracts`, `tests/contracts` |
| Deterministic RNG streams + stable hashing | implemented, tested | `packages/hpe/core`, `tests/core` |
| Novel-symbol task world, held-out splits, assessment, transfer probe, leakage checks | implemented, tested | `packages/hpe/tasks`, `tests/tasks` |
| Developmental learner (shared encoder, plasticity, replay, forgetting, curriculum, trust-weighted social learning, skepticism, 9 ablation flags) | implemented, tested | `packages/hpe/development` (lane `research/development`) |
| Baselines: static persona, memory-only, plain RL | implemented, tested (conformance) | `packages/hpe/evaluation/baselines.py` |
| Statistics, preregistered runner, analysis-from-raw-logs, learning lab, contamination audit | implemented, tested | `packages/hpe/evaluation` (lane `science/evaluation`) |
| Experiment definitions (8 YAML, 10 seeds each) | implemented | `experiments/` |
| Full-budget learning-lab results (H1, Exp1, Exp6, ablations) | **running** — no result reported yet | `reports/experiments/` |
| Society world kernel | not merged (lane running) | lane `sim/world` |
| Society experiments (Exp2–5) | implemented against protocol; not executed (needs world) | `experiments/exp0[2-5]*.yaml` |
| SQLite storage (namespaced, migrations), JSONL/SQLite sinks, checkpoints, stored-run loader | implemented, tested | `packages/hpe/storage` |
| FastAPI service (all INTERFACES §4 endpoints + spec §21 aliases) | implemented, tested with a fake world; `POST /runs` returns 503 until world merges | `apps/api/hpe_api`, `docs/openapi.json` |
| Future Self engine (templates, deterministic calculators, safety routing, no-probability validator, privacy) | implemented, tested | `packages/hpe/futures` |
| Web app (live society, agent inspector, proof of learning, twin experiment, evidence, benchmarks, Future Self 10-step flow, /demo) | implemented, unit-tested, builds; **not yet run against real API** | `apps/web` |
| CLI (`hpe info/smoke/run/demo-run`) | implemented; `smoke` falls back to learning smoke until world merges | `packages/hpe/cli.py` |
| CI (python lint/type/tests/contracts/smoke + web lint/type/test/build) | implemented; first GitHub run pending | `.github/workflows/ci.yml` |
| Docker / compose | not started | — |
| External human validation (Psych-101 MCPL/IGT, HeartSteps V1) | in progress (lane running) | `docs/DATA_SOURCES.md` |
| 1,000 / 10,000-agent benchmarks | not started | — |
| GitHub remote | **verified**: `AdityaJak/Human-possibility-engine` branch `claude/human-possibility-engine-yjujoc` matches local HEAD (public; founder-authorised, ADR-0007) | `git rev-parse origin/...` |

## Last executed commands (orchestrator, integrated branch)

```
.venv/bin/python -m pytest -q             → 207 passed
.venv/bin/ruff check . && ruff format --check .  → clean
.venv/bin/mypy                             → Success: no issues found in 63 source files
python scripts/export_contracts.py --check → contracts up to date
apps/web: npm run lint / typecheck / test / build → 0 warnings / ok / 58 passed / built
python -m hpe.cli smoke                    → learning smoke: held-out compare accuracy 0.500 -> 0.883 (world not yet merged)
```

## Pending contract v0.2 changes (one ADR, after running experiments finish)

1. Optional `OutcomeObservation.shots` (frontend: few-shot transfer curves).
2. Optional `ExperimentDefinition.protocol` / `protocol_params` (evaluation: API-created learning-lab experiments).
3. Exclude `schema_version` from `preregistration_hash` so contract bumps do not invalidate preregistrations.
4. `GET /experiments/{id}/figures/{name}` endpoint (backend).
