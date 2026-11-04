# NEXT_STEPS

Prioritised by what most changes the answer to the investor's decisive question (spec §44).
Each item names the acceptance test. Nothing here is implied to be in progress after this session ends.

## 1. Fix the research hypothesis (H1 failed)

The developmental learner ties plain RL on within-family held-out skill and is worse under high label
noise. Candidate causes, each a falsifiable experiment:

1. **Task world too easy / too linear.** Every rule is a linear readout, so a raw-feature learner is
   near-optimal. Add families with non-linear, compositional and non-stationary rules (rules that change
   at "life events"), where reuse of a learned representation should matter. *Accept:* new preregistered
   H1b with ≥10 seeds.
2. **Developmental components are inert or harmful.** Plasticity schedule hurt (−0.012), curriculum was
   untestable in the fixed-schedule lab, memory added only +0.007. Redesign or remove; test curriculum
   in self-directed practice. *Accept:* ablation shows a component with a CI excluding 0 in the
   predicted direction, on new seeds.
3. **Label-noise fragility.** Developmental learner −0.046 vs plain RL at noise 0.15: add noise-robust
   updates (e.g. confidence-weighted, robust losses). *Accept:* sensitivity variant no longer worse.
4. **Tuning hygiene.** Re-tune the developmental learner on seeds ≥ 1000 only; re-run H1/Exp6 on 0–9.

## 2. Gate 4 — real human validity

* Exp 7 ran with all five classes: **primary (MCPL) failed, secondary (IGT) passed.** Exemplar-memory
  models — including our own `memory_only` baseline — fit 9-way cue learning far better than the
  parametric developmental learner, whose replay memory is used for consolidation but never for
  prediction. Next: add an instance/exemplar pathway to the learner's predictions (a hybrid, not a
  replacement), give every class an equal hyper-parameter grid, and re-test under a new preregistration
  on the same held-out participants. Keep the IGT result as the positive anchor and check it on a second
  bandit dataset.
* Add one longitudinal human learning dataset with verified licence (candidates to review:
  open educational-practice logs). Keep DfE-DB on watch until it publishes a licence.

## 3. Scale (Phase 4)

* Run `make bench-1k` and `make bench-10k` on a quiet machine; record in `reports/scale/`.
* Profile hot paths: per-episode `practice()` overhead, per-agent timeline growth in memory, event
  serialisation. Vectorise assessment across agents; move per-agent histories to the event store.
* Job queue for long API runs (current: thread pool, in-process).

## 4. Product (Future Self)

* User study design (IRB/ethics review first) for a narrow, low-stakes domain where reversible
  experiments are natural (e.g. study habits). Pre-specify outcomes.
* Authentication, encryption at rest, retention enforcement before any real-user data.
* Independent check of the two external citations in `hpe.futures` (see `reports/audit/`).

## 5. Engineering hygiene

* Contract v0.2 ADR: optional `OutcomeObservation.shots`, `ExperimentDefinition.protocol`, a
  human-dataset definition variant, figures endpoint, exclude `schema_version` from preregistration
  hashes, `RunManifest.event_stream_hash`, `SocialEdgeKind.ACQUAINTANCE`.
* Docker images: build and smoke-test on a machine with a Docker daemon (compose config validated only).
* Branch protection on the default branch (repository setting; requires owner).

## Funding-ready milestones

| Milestone (spec §30) | Status | Next evidence needed |
|---|---|---|
| A — 100 agents acquire unfamiliar task skills | Supported in synthetic world (held-out curves, ablations) | — |
| B — developmental beats static and memory-only under comparable compute | Beats static (+0.41) and memory-only (+0.076); **does not beat plain RL** | Item 1 above |
| C — correspondence with independent human data | Not yet (exp07 primary inconclusive in first pass) | Item 2 |
| D — shared representation transfers better than specialists | Supported in synthetic world (Exp 6), with negative transfer when structure differs | Replicate on non-linear task families |
| E — prospective human study shows improved decisions | Not started | Item 4 |
