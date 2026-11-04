# BUILD_STATUS

_Last updated: 2026-11-04 05:05 UTC. All 8 lanes merged into `claude/human-possibility-engine-yjujoc`;
remote verified; CI green on the last completed run. Society experiments 3–5 were still executing
locally at the time of writing (not committed; nothing is implied to continue after the session ends)._

Legend: **implemented** (code exists) · **tested** (tests pass in the integrated branch) · **partial** · **not started**

## Deliverables (spec §31)

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Monorepo with README and one-command setup (`make setup`, `make demo`) | implemented, tested | `make demo-check` 14/14 (infra lane + audit, ports 8790/8798) |
| 2 | Test suite with actual output | tested | `pytest -m "not slow"` → **285 passed, 5 deselected** (95 s); web → **58 passed**; slow tier 5 tests (100×80y, 1,000×2y) passed in the world lane (176 s) |
| 3 | Synthetic demo data, sample simulations, reproducible experiment configs | implemented, tested | `reports/runs/demo_developmental_100a_80y_s42` (replay verified), `experiments/*.yaml` with preregistration hashes |
| 4 | Research dashboard + Future Self app with real navigation | implemented, tested, verified end-to-end against the real API (9 routes, 0 console errors, 0 failed requests) | `apps/web`, `docs/screenshots/` |
| 5 | OpenAPI docs, schemas, architecture, DB design | implemented | `docs/openapi.json` (33 paths), `packages/hpe/contracts/schema/`, `ARCHITECTURE.md`, `packages/hpe/storage/migrations.py` |
| 6 | Benchmark results, charts, assumptions, model cards, provenance, limitations | implemented (what was run) | `RESEARCH_RESULTS.md`, `reports/experiments/*/figures`, `docs/MODEL_CARDS.md`, `docs/DATA_SOURCES.md`, `docs/CLAIMS_AND_LIMITATIONS.md` |
| 7 | BUILD_STATUS.md | this file | — |
| 8 | NEXT_STEPS.md | implemented | `NEXT_STEPS.md` |
| 9 | No invented results | enforced | evidence ledger generated from artifacts (`python -m hpe.ledger`); independent audit `reports/audit/AUDIT.md` |

## Components

| Component | Status | Notes |
|---|---|---|
| Contracts v0.1 + JSON Schema/TS export | implemented, tested | 68 types; `--check` in CI |
| RNG streams, hashing | implemented, tested | |
| Novel-symbol task world + measurement protocol | implemented, tested | contamination audit clean |
| Developmental learner (9 ablation flags) | implemented, tested | ~55 µs/episode (lane measurement) |
| Baselines: static persona, memory-only, plain RL | implemented, tested | plain_rl tuned on seeds 1001–1003 |
| Statistics, preregistered runner, analysis-from-raw-logs, learning lab | implemented, tested | re-analysis byte-identical for all 5 committed experiments |
| Society kernel (`hpe.world`) | implemented, tested | 100×80y: 117 s; determinism + snapshot/resume bit-identical |
| Storage (SQLite namespaces, sinks, checkpoints, stored-run loader) | implemented, tested | |
| FastAPI service (all INTERFACES §4 endpoints + §21 aliases) | implemented, tested with real World | `POST /experiments/{id}/run` returns 501 (runner not wired to API) |
| Future Self engine (templates, calculators, safety, privacy) | implemented, tested; audit findings H-1/M-1/M-2 fixed and re-tested | keyword classifier is not a risk detector |
| Web app (9 routes, Future Self 10 steps, /demo) | implemented, tested, e2e verified | accessibility: light-mode contrast + scrollable-focus issues open (audit M-6) |
| CLI (`info/smoke/run/demo-run/verify-replay`) | implemented, tested | |
| CI (lint, types, contracts, openapi, unit, integration, smoke, demo-check, security, nightly slow) | implemented; **green** on run #13 (last completed), #14+ in progress at time of writing | `.github/workflows/ci.yml` |
| Docker (api/web Dockerfiles, compose) | partial | `docker compose config` validates; **no image built** (no daemon in this environment) |
| Human validation adapters (Psych-101, HeartSteps V1) | implemented, tested | licences verified; no raw data committed |
| Evidence ledger / Evidence page | implemented | 18 claims; 9 supported, 6 not supported, 1 inconclusive, 2 not tested |
| Investor assets | implemented | `docs/INVESTOR_DEMO.md`, `docs/INVESTOR_TECHNICAL_DILIGENCE.md`, screenshots, recording shot list |
| 1,000 / 10,000-agent benchmarks | **not started** | harness ready (`make bench-1k`, `make bench-10k`); needs a quiet machine |
| Society experiments 3–5 | **running / not committed** | Exp 2 committed |
| Exp 7 with all policy classes | committed | `reports/human_validation/exp07_human_heldout` (2,654 s) |

## Experiments executed (10 seeds each unless stated)

| Experiment | Verdict | Primary estimate (95% CI) |
|---|---|---|
| H1 developmental vs strongest baseline (plain RL) | **FAILED** | +0.012 [−0.008, +0.031] |
| Exp 1 instructional vs no feedback | passed | +0.391 [+0.374, +0.410] |
| Exp 6 transfer, shared vs specialists | passed | +0.090 [+0.050, +0.123]; orthogonal control −0.092 |
| Ablations: learning necessary | passed | +0.409 [+0.395, +0.426] |
| Exp 2 twin tutoring | passed; manipulation check failed as defined | +0.029 [+0.022, +0.037]; non-learning society 0.000 |
| Exp 3 social influence (H2) | **FAILED** | connected − isolated −0.004 [−0.007, −0.000]; developmental agents form 0.018 friends on average (mechanism gap) |
| Exp 7b HeartSteps causal grounding (real data) | passed (weak rule, see audit) | +12.2% [−0.5%, +27.4%] vs published +14% |
| Exp 7 Psych-101 held-out choices (real data, all 5 classes + 4 references) | **FAILED** primary (MCPL); secondary (IGT) passed | MCPL: +0.229 nats worse than instance-based [+0.171, +0.289]; IGT: best model, −0.110 [−0.140, −0.083] |

## Gates

| Gate | Verdict | Basis |
|---|---|---|
| §52 Gate 0 monorepo, contracts, ownership, local run | GO | audit |
| §52 Gate 1 determinism, persistence, replay, coverage | GO | replay verified; branch coverage 91% (audit) |
| §52 Gate 2 measurable learning, ablations, honest uncertainty | GO | preregistered results incl. a failed H1 |
| §52 Gate 3 web/API integration, demo, accessibility, secrets | conditional | e2e verified; secrets scan clean; accessibility fixes open |
| §52 Gate 4 external human evidence for the learner | **NO-GO** | primary human test (MCPL) failed; secondary (IGT) passed; causal pipeline validated on HeartSteps |
| §52 Gate 5 performance to 10,000 agents | **NO-GO** | no benchmark record |
| §42 Gate 1 `make demo` + 100-agent seeded run with replay | GO | |
| §42 Gate 2 learned parameters change with experience; held-out skill; baselines | GO | |
| §42 Gate 3 randomised simulated intervention reproduces, exports, flags model dependence | GO (Exp 2 committed after audit) | |
| §42 Gate 4 Future Self functional, consent-based, no unsupported predictions | GO after fixes (audit H-1, M-1 re-tested) | |
| §42 Gate 5 CI green, no credentials, remote verified | GO | |
| §42 Gate 6 investor can follow the script and reproduce a result | GO | `docs/INVESTOR_DEMO.md`; Exp 6 from-scratch re-run 0/40 logs differ |

## Last executed commands (integrated branch)

```
.venv/bin/python -m pytest -q -m "not slow"       → 285 passed, 5 deselected in 95.14s
.venv/bin/ruff check . / ruff format --check .     → clean (155 files)
.venv/bin/mypy                                      → Success: no issues found in 79 source files
python scripts/export_contracts.py --check          → contracts up to date
python apps/api/export_openapi.py --check           → openapi up to date
apps/web: npm run lint / typecheck / test / build   → 0 warnings / ok / 58 passed / built (react-router 7.18.5, npm audit 0)
python -m hpe.cli verify-replay reports/runs/demo_developmental_100a_80y_s42 → "passed": true
python -m hpe.evaluation run experiments/exp06_transfer.yaml --out data/rerun → 0 of 40 raw logs differ from committed
AUDIT_RAISE_RATE_LIMIT=1 python reports/audit/scripts/adversarial_api.py ... → crisis_oblique routed to crisis; 200 KB input → 422
```

## Blockers / needs the founder

* **License**: none assigned (ADR-0007). Choose one.
* **Branch protection**: repository setting; requires owner.
* **Docker images**: need a machine with a Docker daemon to build and smoke-test.
* **Scale benchmarks**: need a quiet machine (`make bench-1k`, `make bench-10k`).

## Pending contract v0.2 (one ADR, not yet applied)

Optional `OutcomeObservation.shots`; `ExperimentDefinition.protocol` + a human-dataset variant; figures
endpoint; exclude `schema_version` from preregistration hashes; `RunManifest.event_stream_hash`;
`SocialEdgeKind.ACQUAINTANCE`; `FeedbackSource` caregiver/mentor; `WorldParams.prehistory_max_episodes`.
