# Interfaces (contract v0.1.0 — frozen)

Source of truth: `packages/hpe/contracts/`. This document explains intent; the code wins on detail.

## 1. Runtime loop (world ⇄ policy)

```
for each tick (dt_days, default 30):
  for agent in sorted(alive agents):
    obs  = Observation(kind="activity", family="activity", context=(1, F) need/affect vector,
                       n_actions=len(ACTIVITY_ACTIONS), action_mask=stage-dependent)
    dec  = policy.decide(obs, rng=streams.get("agent", id, "activity"))
    ... world executes activity → school/tutoring/work/self-study/social episodes ...
    task episodes go through hpe.tasks.practice(policy, symbol_world, family, n, rng, mode=...)
    activity reward → policy.learn(obs, dec.action, Feedback(mode="bandit", reward=r))
    social episodes → Observation(kind="social"), Demonstration(...) → learn_from_demonstration
    policy.tick_lifecycle(age, dt_years)
  yearly: hpe.tasks.assess(policy, symbol_world, family, age) for every family → CapabilityState
```

* `ACTIVITY_ACTIONS = ["rest", "study", "work", "socialize"]` (index order is part of the contract).
* Activity context vector (1 slot, 8 features, in this order):
  `[energy, stress, social_connection, resources_norm, age_norm, employed, enrolled, 1.0]`.
* Practice decision: `kind="practice"`, `family="practice"`, `n_actions=len(TRAINING_FAMILIES)`,
  context `(1, 2*len(TRAINING_FAMILIES)+1)` = `[recent_reward_f..., practice_share_f..., 1.0]`.
* Social decision: `kind="social"`, `family="trust_game"`, `n_actions=2` (0 = withhold, 1 = cooperate),
  context `(1, 4)` = `[trust_in_partner, partner_is_family, partner_is_friend, 1.0]`.
* Task decision: `kind="task"`, `family ∈ ALL_FAMILIES`, context `(n_slots, 12)` of ±1/0 features.

## 2. Policy classes (all implement the same protocol, same inputs, same budget)

| `policy_class` | Owner | Learns parameters? | Episodic memory? | Shared representation? | Social learning? |
|---|---|---|---|---|---|
| `static_persona` | evaluation | no (fixed trait-derived heuristic) | no | no | no |
| `memory_only` | evaluation | no (fixed retrieval rule over stored episodes) | yes (bounded) | no | stores observed demos as memories |
| `plain_rl` | evaluation | yes (independent per-family online learner) | no | no | no |
| `developmental` | development | yes | yes (replay, bounded) | yes (ablatable) | yes, trust-weighted (ablatable) |
| `random` | contracts | no | no | no | no |

Budget matching: every class sees the identical observation/feedback stream in learning-lab
experiments (same item sequence via common random numbers) and the same world inputs in society
experiments. `ComputeCounter` and wall time are reported per class.

## 3. Measurement

* `hpe.tasks.assess(policy, world, family, age)` → `AssessmentResult` on the fixed held-out eval set
  (holdout symbols only, learning disabled, ties credited fractionally). This is the only skill metric.
* `hpe.tasks.transfer_probe(...)` → few-shot curve on a **clone** for an unseen family (`barter`).
* `TaskWorldConfig(transfer_mode="orthogonal")` is the negative-transfer control world.

## 4. HTTP API (backend implements, web consumes; models in `hpe.contracts.api`)

| Method & path | Request | Response | Notes |
|---|---|---|---|
| GET `/health` | — | `Health` | |
| POST `/runs` | `RunCreate` | `RunInfo` | background job or interactive world; quotas from env |
| GET `/runs` | — | `RunInfo[]` | includes stored real outputs from `reports/runs/` |
| GET `/runs/{id}` | — | `RunInfo` | job status + progress |
| POST `/runs/{id}/step` | `StepRequest` | `WorldTick` | interactive runs only |
| POST `/runs/{id}/control` | `RunControl` | `RunInfo` | pause / resume / reset / cancel |
| GET `/runs/{id}/agents` | `?offset&limit&q&alive` | `AgentPage` | |
| GET `/runs/{id}/agents/{agent_id}` | — | `AgentDetail` | timeline, skill curve, parameter history |
| GET `/runs/{id}/events` | `EventQuery` as query params | `EventPage` | |
| GET `/runs/{id}/frames` | `?tick_from&tick_to&every` | `FramesResponse` | map playback |
| GET `/runs/{id}/timeseries` | — | `TimeSeriesResponse` | |
| GET `/runs/{id}/download` | — | `application/gzip` JSONL | raw event log + manifest |
| POST `/experiments` | `ExperimentCreate` | `ExperimentInfo` | preregistration hash frozen at creation |
| POST `/experiments/{id}/run` | — | `ExperimentInfo` | job |
| GET `/experiments` | — | `ExperimentInfo[]` | |
| GET `/experiments/{id}/results` | — | `ExperimentInfo` | `result: ExperimentResult` |
| GET `/research/metrics` | — | `{experiments, runs, benchmarks}` summary | |
| GET `/research/claims` | — | `ClaimRecord[]` | the investor Evidence page |
| GET `/benchmarks` | — | `BenchmarkRecord[]` | measured scale results only |
| POST `/scenarios` (alias `/decisions/scenarios`) | `DecisionRequest` | `ScenarioResponse` | requires `consent_given` |
| POST `/decisions/reflect` | `Reflection` | `Reflection` | |
| GET `/evidence` / `/evidence/{id}` | — | `EvidenceRecord[]` / `EvidenceRecord` | |
| POST `/privacy/consent` | `ConsentRecord` | `ConsentRecord` | |
| POST `/privacy/export` | `PrivacyRequest` | `PrivacyExport` | |
| POST `/privacy/delete` | `PrivacyRequest` | `PrivacyDeleteResult` | |
| GET `/journey/{user_id}` | — | `{decisions, reflections}` | |

Spec §21 aliases (`/worlds`, `/worlds/{id}/run`, `/agents/{id}`) map onto `/runs`.

## 5. Stored run layout (`reports/runs/<run_id>/`, also `data/runs/<run_id>/`)

```
manifest.json          RunManifest
events.jsonl.gz        ExperienceEvent per line, seq order
ticks.jsonl.gz         WorldTick per line (frames stripped)
frames.jsonl.gz        {"tick": t, "frames": AgentFrame[]} sampled per LogConfig
final_agents.json      AgentState[]
edges.json             SocialEdge[]
map.json               WorldMap
outcomes.jsonl.gz      OutcomeObservation per line
checkpoints/<tick>.json.gz   World.snapshot()
```

## 6. Experiment artifact layout (`reports/experiments/<experiment_id>/`)

```
definition.json        ExperimentDefinition incl. preregistration_hash (written before any run)
runs/<run_id>/...      as §5 (may be pruned to manifest + outcomes for size; raw logs regenerable)
result.json            ExperimentResult (regenerated from raw logs/outcomes by the analysis module)
report.md              human-readable report with figures and limitations
figures/*.png|svg      regenerated plots
```
