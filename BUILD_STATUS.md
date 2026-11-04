# BUILD_STATUS

_Last updated: 2026-11-04 05:05 UTC. All 8 lanes merged into `claude/human-possibility-engine-yjujoc`;
remote verified; CI green on the last completed run. All preregistered experiments (learning lab,
society 2–5, human data 7/7b) and the 1,000- and 10,000-agent benchmarks are executed and committed.
Nothing is implied to continue after the session ends._

Legend: **implemented** (code exists) · **tested** (tests pass in the integrated branch) · **partial** · **not started**

## Deliverables (spec §31)

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Monorepo with README and one-command setup (`make setup`, `make demo`) | implemented, tested | `make demo-check` 14/14 (infra lane + audit, ports 8790/8798) |
| 2 | Test suite with actual output | tested | `pytest -m "not slow"` → **285 passed, 5 deselected** (95 s); `pytest -m slow` at integrated HEAD → **5 passed** (148 s; 100×80y lifespan, 1,000×2y, under concurrent experiment load); web → **58 passed** |
| 3 | Synthetic demo data, sample simulations, reproducible experiment configs | implemented, tested | `reports/runs/demo_developmental_100a_80y_s42` (replay verified), `experiments/*.yaml` with preregistration hashes |
| 4 | Research dashboard + Future Self app with real navigation | implemented, tested, verified end-to-end against the real API (9 routes, 0 console errors, 0 failed requests) | `apps/web`, `docs/screenshots/` |
| 5 | OpenAPI docs, schemas, architecture, DB design | implemented | `docs/openapi.json` (33 paths), `packages/hpe/contracts/schema/`, `ARCHITECTURE.md`, `packages/hpe/storage/migrations.py` |
| 6 | Benchmark results, charts, assumptions, model cards, provenance, limitations | implemented (what was run) | `RESEARCH_RESULTS.md`, `reports/experiments/*/figures`, `docs/MODEL_CARDS.md`, `docs/DATA_SOURCES.md`, `docs/CLAIMS_AND_LIMITATIONS.md` |
| 7 | BUILD_STATUS.md | this file | — |
| 8 | NEXT_STEPS.md | implemented | `NEXT_STEPS.md` |
| 9 | No invented results | enforced | evidence ledger generated from artifacts (`python -m hpe.ledger`); independent audit `reports/audit/AUDIT.md` |

## Deployment (2026-11-04)

| Target | Status | Evidence / blocker |
|---|---|---|
| GitHub Pages static site (dashboard + in-browser Future Self) | **LIVE** — <https://adityajak.github.io/Human-possibility-engine/> | Pages enabled by the founder; deep-link 404 defect found on first deploy, fixed (`scripts/static_routes.py`) and re-verified against the live site: 9/9 routes, 0 errors; Future Self end-to-end 11.1 s with the real in-browser engine (`docs/DEPLOYMENT.md`) |
| Render (API, read-only) | config verified, **not deployed** (needs account) | `render.yaml`, `HPE_READ_ONLY=1` tested (405 on job endpoints), Dockerfile `$PORT`/healthcheck exercised outside Docker |
| Vercel (web) | config verified, **not deployed** (needs account) | `apps/web/vercel.json` |
| Local full stack | works | `make demo`, `make demo-check` 14/14 |

## Phase II — Scientific Breakthrough Mandate (2026-11-04)

| Item | Status | Evidence |
|---|---|---|
| Stage 1 diagnose: reproduction, gap analysis, thesis v2 with fixed gates, frozen partitions | done | `docs/phase2/`, `reports/phase2/stage1_reproduction.json`, partitions sha `7dc2e779…` |
| Candidate A `hybrid_exemplar` + v2 task families | built, dev-evaluated (exploratory) | X-003/X-004: MCPL −0.040 vs exemplar reference; ≤ best exemplar baseline on 3/5 synthetic families |
| Candidate-free gate tooling (composition family, Gate A controls) | built, dev-evaluated | X-007 (controls pass), X-008 (composition uninformative: parity unlearned by all) |
| Human validation v2 (9 families, test lock, equal grids, GRU/PVL/Bayes/GCM references) | built, dev pilot done | X-009: population-fitted GRU beats every per-participant learner in 8/9 families |
| Candidate A on pristine families (pilot B, X-010) | **running at time of writing** | `reports/phase2/human/p2_human_dev_pilot_hybrid/` (not committed until complete) |
| Social mechanism v2 (Candidate E) | built, dev-evaluated, bit-identical off | friendships repaired via information exchange; connected − isolated ≈ 0 |
| Causal validity (estimator sim, HTE, OPE) | done incl. two test-partition confirmations (deviations recorded) | C-001 not supported; C-002 passed; Gate E untouched |
| Affective mechanisms (Candidate D) | **not executed** (lane killed by usage-credit limit, twice) | `AFFECTIVE_MODEL_REPORT.md` |
| Independent critic | design review done (blocking findings acted on); results review **pending** (lane killed by credit limit) | `INDEPENDENT_REVIEW_V2.md`, `scripts/phase2_guard.py` (clean) |
| State-of-the-art benchmark | written; no SOTA claim | Centaur/Minitaur not runnable here; published figures non-comparable |
| Confirmatory gate evaluations | **none run** | `EXPERIMENT_REGISTRY.md` §4; draft `gateD_v2.yaml` not frozen |
| GO / PIVOT / NO-GO | pending X-010 | `BREAKTHROUGH_STATUS.md` |

Phase II compute ≈ 3 CPU-h, 0 GPU, 0 paid API calls. All lanes ran as parallel worktrees; four lane
runs were terminated by the account's API usage-credit limit (two salvaged by the orchestrator from
their worktrees, two never started).

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
| 1,000-agent benchmark | measured | 1,000 × 10 y: 165.7 s wall, 64.2 agent-years/s, 912 MB peak RSS, load avg 2.2–2.9 disclosed (`reports/scale/developmental_1000a_10y_*.json`) |
| 10,000-agent benchmark | measured | 10,000 × 1 y: 720.7 s wall / 712.6 s CPU, 5.1 GB peak RSS, 194,969 events, load avg ≈1 (quiet); 476 s of it is one-time prehistory; cProfile top-25 committed alongside (`reports/scale/developmental_10000a_1y_*.json/.profile.txt`) |
| Society experiments 2–5 | committed | all four executed at full budget (10 paired seeds each); re-analysis byte-identical for each |
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
| Exp 4 midlife education | **FAILED** | −0.003 [−0.015, +0.007]; balance and mechanism checks passed; uptake not measured (design weakness) |
| Exp 5 multigenerational (cross-fostering) | inconclusive | rearing-household corr. −0.075 [−0.177, +0.023]; no transmission detectable through either channel; underpowered at the task ceiling |
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
| §52 Gate 5 performance to 10,000 agents | GO as an engineering benchmark only | 1,000 × 10 y: 165.7 s, 912 MB; 10,000 × 1 y: 720.7 s, 5.1 GB, completed on disclosed hardware. No scientific claim at scale: the spec's "only scale when scientifically justified" condition is not met while H1/H2 stand failed |
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
