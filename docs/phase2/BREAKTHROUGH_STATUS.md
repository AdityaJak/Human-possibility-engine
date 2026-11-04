# Breakthrough Status

**No scientific breakthrough is claimed.** This page is updated only from confirmatory ledger rows
(`EXPERIMENT_REGISTRY.md` §4) and the independent review.

| Gate | Status | Basis |
|---|---|---|
| A — genuine learning | supported in the synthetic world (Phase I) | ablations: freezing learning −0.41 |
| B — strong-baseline advantage | **not supported** | H1: +0.012 vs plain RL, CI includes 0 |
| C — transfer | supported in the synthetic world, with negative transfer quantified | Exp 6 |
| D — human behavioural correspondence | **not passed** | 1 of 2 families (IGT best of 8; MCPL worse than exemplar models by +0.23 nats) |
| E — causal decision improvement | not started | requires a prospective human study. Retrospective evidence (HeartSteps test, C-002): suggestions help (+0.167 [+0.018, +0.320] vs never) but a dev-learned tailored policy adds nothing over always-send (+0.021 [−0.107, +0.156]); no reproducible heterogeneity (C-001) |
| F — cross-domain generalisation | not tested | — |

Recommendation at this point: **PIVOT the architecture, keep the programme** — the environment and the
learner's prediction pathway are the identified faults, not the developmental premise, which remains
untested on adequate tasks. Final GO / PIVOT / NO-GO is issued at Stage 8 from confirmatory results.
