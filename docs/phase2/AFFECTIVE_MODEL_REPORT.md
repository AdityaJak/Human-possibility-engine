# Affective Model Report — Candidate D

**Status: NOT EXECUTED.** No affective mechanism was implemented or evaluated in Phase II. This file
exists so that the absence is explicit; it contains no results.

## What happened

The `research/affective` lane was launched twice (brief in `CANDIDATE_ARCHITECTURES.md`, Candidate D)
and both runs were terminated by the account's API usage-credit limit before any code was written
(HTTP 429, "out of usage credits"). No worktree artifacts exist. The director prioritised the
deployment and the human-validation critical path over re-attempting this lane inline.

## What was specified (and remains the plan)

Literature-grounded computational affect — mood as a leaky integrator of reward prediction errors,
arousal from recent surprise modulating exploration temperature and learning rate, frustration from
runs of negative prediction errors modulating persistence and retrieval weighting, perceived control
from outcome contingency — each individually ablatable, with a **matched-complexity control**
(`affect_random_modulators`: the same free parameters driven by seeded noise). Evaluation: synthetic
dev experiments with context shifts and setback-recovery blocks; a human dev-partition pilot on
families where affect is relevant (`lefebvre2017behavioural` optimistic reinforcement learning with
a fitted asymmetric-learning-rate reference model; `garcia2023experiential` / `plonsky2018when`
gains and losses). All of it exploratory until a preregistered confirmatory test.

## What the existing evidence says about affect

* Phase I world affect proxies (stress, energy, social connection, motivation) exist only as control
  signals for activity choice; they were never evaluated against human data.
* The Phase II human pilot (`HUMAN_VALIDATION_V2.md` §7) shows the developmental learner behind a
  population-fitted GRU on `casino` (lefebvre2017, +0.067 nats) and `garcia` (+0.194) — the families
  where affect mechanisms would be tested. Any affective candidate must therefore be compared against
  that GRU under D-pop, and against per-participant models under D-ind.

## Claims permitted

None. H3's affective half is untested. No statement about computational emotion, mood or motivation
in this system is supported by any experiment.
