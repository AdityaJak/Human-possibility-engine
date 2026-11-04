# Investor Technical Brief — Phase II

_Two pages. Every number links to a committed artifact via `RESEARCH_RESULTS_V2.md` and the registry.
Nothing here is a confirmatory result; the programme's decision is in `BREAKTHROUGH_STATUS.md`._

## What we set out to test

Whether developmental experience can produce reusable representations of learning and adaptation that
predict real human decisions better than strong non-developmental models (`RESEARCH_THESIS_V2.md`).
Four hypotheses (H1–H4), six gates with thresholds fixed before any experiment, frozen participant
partitions, equal tuning budgets, an independent critic with veto, and automated leakage guards.

## What we built this phase

* A hybrid parametric + exemplar learner with an ordinal output structure and a learned arbitration
  gate (Candidate A), diagnosed from where the Phase I learner lost likelihood.
* Five harder synthetic task families plus a candidate-free compositional family; Gate A controls
  that can fail (permuted feedback, age-script-only null).
* A nine-family human-validation harness over Psych-101 with a code-enforced test lock, equal tuning
  grids and stronger references: exemplar GCM, PVL-Delta, Bayesian rule learner, and a small GRU
  sequence model fitted on other participants.
* An instrumented social mechanism that repairs friendship formation through information exchange
  (with a reward-only control), and a validated causal estimator with heterogeneity and off-policy
  analyses on a real micro-randomised trial.
* A static, backend-free deployment of the whole dashboard and Future Self (the Python engine runs in
  the browser), verified end to end; publication waits on one repository setting.

## What the evidence says

**Where we win (dev, exploratory).** Candidate A beats the best exemplar reference on human
cue-learning (−0.040 nats, controls confirming both mechanisms matter) and beats plain RL on 4/5 new
synthetic families; learning is genuine under controls designed to fail it.

**Where we lose.** A 666-parameter sequence model fitted on *other people* beats every per-participant
learner we have in 8 of 9 human families — the Phase I IGT advantage does not survive it — and Candidate
A, the architecture built this phase, is 0.18–0.27 nats behind it on every pristine family (X-010); it
improves on the Phase I learner by at most 0.016 nats. No architecture, ours included, learns non-linear
compositional structure (parity/XOR) at the budgets used. Tailored intervention selection learned from 22
participants adds nothing over always-send.

**What we could not test.** Centaur-70B / Minitaur-8B cannot run here; published figures use a
different protocol and are not comparable. Affect mechanisms were not executed. The second,
independent human dataset Gate D requires is identified but not adapted.

## What this means for the thesis

**Decision (Stage 8): PIVOT.** The developmental premise is neither confirmed nor refuted; no gate has a
confirmatory pass and none was attempted, because the stop rule (freeze only when the candidate is within
the margin of the best reference on the tuning partition) was applied. Two things changed this phase: the
baseline bar is now a population-fitted model (which is also what Centaur is), and the next
architecture must therefore learn a *prior across people* and then adapt per person (Candidate F) —
or the question must be narrowed, explicitly and separately, to per-participant adaptation (D-ind).
Both are preregistrable; neither has been run. What an investor should take from this phase: the
evaluation machinery is now strong enough to reject our own architecture quickly and cheaply (≈ 3.5
CPU-hours for the whole phase), which is the precondition for any credible positive result later.

## Diligence pointers

Confirmatory ledger (`EXPERIMENT_REGISTRY.md` §4): two causal rows only. Guards:
`scripts/phase2_guard.py check-test-access` (clean), partitions SHA `7dc2e779…`. Independent review:
`INDEPENDENT_REVIEW_V2.md` (design findings acted on; results review pending — the critic lane was
stopped by the account's usage limit). Compute: ≈ 3 CPU-hours total, no GPU, no paid API.
