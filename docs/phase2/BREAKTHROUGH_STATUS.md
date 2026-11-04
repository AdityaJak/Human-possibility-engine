# Breakthrough Status

**No scientific breakthrough is claimed.** This page is updated only from confirmatory ledger rows
(`EXPERIMENT_REGISTRY.md` §4) and the independent review.

| Gate | Status | Basis |
|---|---|---|
| A — genuine learning | supported in the synthetic world (Phase I) | ablations: freezing learning −0.41 |
| A — genuine learning (Phase II controls) | supported on dev (exploratory) | permuted-feedback gain ≈ 0 for all learners; frozen/age-script gains ≈ +0.39 (X-007); confirmatory run pending |
| B — strong-baseline advantage | **not supported; not yet evaluable in Phase II** | Phase I: +0.012 vs plain RL. Phase II: Candidate A ≤ best exemplar baseline on 3/5 v2 families (X-004); `conjunction` and the candidate-free `composition` family are unsolved by every architecture (X-008) |
| C — transfer | supported in the synthetic world, with negative transfer quantified | Exp 6 |
| D — human behavioural correspondence (D-pop) | **not passed; dev evidence points to failure** | Phase I: 1 of 2 families. Phase II dev pilot (X-009): a population-fitted GRU beats every per-participant learner in 8/9 families; Candidate A beats the exemplar reference on MCPL (dev, −0.040) but is untested against the GRU on pristine families (X-010 running) |
| D-ind — per-participant adaptation vs per-participant models (secondary; never reported as Gate D) | defined, not evaluated | thesis amendment after X-009 |
| E — causal decision improvement | not started | requires a prospective human study. Retrospective evidence (HeartSteps test, C-002): suggestions help (+0.167 [+0.018, +0.320] vs never) but a dev-learned tailored policy adds nothing over always-send (+0.021 [−0.107, +0.156]); no reproducible heterogeneity (C-001) |
| F — cross-domain generalisation | not tested | — |

Recommendation at this point: **PIVOT the architecture, keep the programme** — the environment and the
learner's prediction pathway are the identified faults, not the developmental premise, which remains
untested on adequate tasks. Final GO / PIVOT / NO-GO is issued at Stage 8 from confirmatory results.
