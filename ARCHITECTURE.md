# Architecture

> Synthetic research prototype; not validated for predicting human futures.

## 1. Tiers (spec §48)

```
Tier 5  Consumer Future Self      apps/web (Future Self routes) ── hpe.futures (scenario engine, evidence registry)
Tier 4  Evidence-grounded research hpe.evaluation (experiments, stats, human-data adapters) ── reports/
Tier 3  Optional language model    adapter interface only, disabled by default, narration only
Tier 2  Trainable learning         hpe.development (developmental learner) + hpe.evaluation.baselines
Tier 1  Simulation infrastructure  hpe.world (society kernel) + hpe.tasks (novel-symbol tasks) + hpe.core
                                   hpe.storage (event logs, checkpoints, SQLite) ── apps/api (FastAPI)
```

Tiers 1–2 never call a language model. Tier 3 may only produce narration text and can never be the
source of numbers, ground truth, or evidence of competence.

## 2. Monorepo layout

```
packages/hpe/contracts   frozen v0.1 schemas, Policy protocol, World/Sink protocols, registry   (orchestrator)
packages/hpe/core        named RNG streams (common random numbers), stable hashing              (orchestrator)
packages/hpe/tasks       novel-symbol task world, held-out splits, assess/practice/transfer     (orchestrator)
packages/hpe/world       event-driven society: households, lifecycle, school, jobs, social net  (sim/world)
packages/hpe/development developmental learner: shared representation, plasticity, replay       (research/development)
packages/hpe/evaluation  baselines, statistics, experiment runner, report generation           (science/evaluation)
packages/hpe/storage     SQLite (namespaced), JSONL event sinks, checkpoints, migrations        (platform/backend)
packages/hpe/futures     Future Self scenario engine, deterministic calculators, safety          (platform/backend)
apps/api/hpe_api         FastAPI service                                                        (platform/backend)
apps/web                 React + TypeScript (Vite): research dashboard + Future Self app        (product/frontend)
experiments/             preregistered experiment definitions (YAML)                            (science/evaluation)
reports/                 committed real outputs: experiments, runs, benchmarks, audits
docs/                    ADRs, protocols, claims policy, data sources, investor docs
```

Deviation from the spec's suggested `packages/<name>/` paths: Python code lives in
`packages/hpe/<name>/` so a single importable namespace (`hpe.*`) works in every git worktree
without per-worktree installation (ADR-0001).

## 3. Data flow

```
SimulationConfig ──► World ──Observation──► Policy.decide ──Decision──► World
                       │  ◄──LearningUpdate── Policy.learn ◄──Feedback──┘
                       │
                       ├─ yearly hpe.tasks.assess (held-out, no learning) ─► CapabilityState / OutcomeObservation
                       ├─ ExperienceEvent stream ─► EventSink (JSONL.gz / SQLite) ─► event_stream_hash
                       └─ snapshot() ─► checkpoint ─► load_snapshot() (bit-identical continuation)

ExperimentDefinition ─(preregistration hash)─► runner: arms × seeds ─► RunManifests + raw logs
                    ─► analysis regenerates metrics/figures FROM RAW LOGS ─► ExperimentResult + report.md

DecisionRequest (consented demo user) ─► hpe.futures ─► Scenario[] + EvidenceRecord[]   (no simulation data used)
```

## 4. Key design decisions (see docs/ADRS)

1. ADR-0001 Monorepo with `hpe.*` namespace; Python 3.12+, numpy, pydantic v2, FastAPI, SQLite.
2. ADR-0002 Determinism via named RNG streams and sorted iteration; event-stream hashing.
3. ADR-0003 Novel-symbol task world: held-out symbols, shared latent structure, orthogonal control.
4. ADR-0004 Policy classes and budget matching (static persona, memory-only, plain RL, developmental).
5. ADR-0005 Data separation namespaces; Future Self never consumes simulation outputs.
6. ADR-0006 Multi-worker orchestration with git worktrees and lane ownership.
7. ADR-0007 Repository visibility and license are founder decisions.

## 5. Approximations (documented, not hidden)

* Time step defaults to 30 days; idle infants/elders still tick (cheap). An 80-year life is
  ~973 ticks. Learning happens in a bounded number of task episodes per tick, not continuous time.
* Age gates (perceptual feature access, plasticity schedules) are **scripted assumptions** and are
  applied identically to every policy class. Ablations switch them off.
* The economy, schooling, mortality and fertility are abstract mechanisms with made-up parameters;
  they are not calibrated to any population.
