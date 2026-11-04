# Breakthrough Status

**No scientific breakthrough is claimed.** This page is updated only from confirmatory ledger rows
(`EXPERIMENT_REGISTRY.md` §4) and the independent review.

| Gate | Status | Basis |
|---|---|---|
| A — genuine learning | supported in the synthetic world (Phase I) | ablations: freezing learning −0.41 |
| A — genuine learning (Phase II controls) | supported on dev (exploratory) | permuted-feedback gain ≈ 0 for all learners; frozen/age-script gains ≈ +0.39 (X-007); confirmatory run pending |
| B — strong-baseline advantage | **not supported; not yet evaluable in Phase II** | Phase I: +0.012 vs plain RL. Phase II: Candidate A ≤ best exemplar baseline on 3/5 v2 families (X-004); `conjunction` and the candidate-free `composition` family are unsolved by every architecture (X-008) |
| C — transfer | supported in the synthetic world, with negative transfer quantified | Exp 6 |
| D — human behavioural correspondence (D-pop) | **not passed; dev evidence contradicts it for every architecture we have** | Phase I: 1 of 2 families. Phase II dev pilots: a population-fitted GRU beats every per-participant learner in 8/9 families (X-009); Candidate A is 0.18–0.27 nats behind it on all three pristine families (X-010). Stop rule applied: no freeze, no test-partition run |
| D-ind — per-participant adaptation vs per-participant models (secondary; never reported as Gate D) | **not met on dev** | Candidate A beats the exemplar and memory-only references everywhere but wins no family against the best per-participant reference by ≥ 0.02 (bandit2 +0.052 behind PVL-Delta; garcia +0.007; described −0.009, n.s.) (X-010) |
| E — causal decision improvement | not started | requires a prospective human study. Retrospective evidence (HeartSteps test, C-002): suggestions help (+0.167 [+0.018, +0.320] vs never) but a dev-learned tailored policy adds nothing over always-send (+0.021 [−0.107, +0.156]); no reproducible heterogeneity (C-001) |
| F — cross-domain generalisation | not tested | — |

## Stage 8 decision (2026-11-04): PIVOT

**Not GO.** No gate has a confirmatory pass; the one architecture built this phase (Candidate A) is
0.18–0.27 nats behind a 666-parameter sequence model fitted on other participants in every pristine
human family, and is at best a marginal improvement on the Phase I learner against per-participant
models. Nothing here would survive an independent replication as evidence that developmental
experience predicts human decisions better than strong non-developmental models.

**Not NO-GO.** The foundational hypothesis has not been tested on adequate tasks: every learner we
fielded (developmental or not) learns one person from scratch, and the family that beats them all does
so by carrying population regularities. The two things that would make the premise testable are known
and preregistrable: (1) a developmental learner that acquires a prior *across people* and then adapts
per person (Candidate F), scored under the same equal-tuning protocol against the GRU; (2) a separately
labelled per-participant sub-question (D-ind) with its own margin. Neither has been run. Harder
synthetic families (conjunction, composition) are unsolved by every architecture, so Gate B cannot yet
discriminate a developmental advantage from a general failure.

**PIVOT means:** the Candidate A line is closed (it does not beat the bar that matters); the next
architecture must learn from other people before it learns from the person; the environment work
(Gate B families, Gate A controls) and the human-validation harness are kept as they are the only parts
that produced decisive answers. Conditions for reopening a GO case: a candidate within the preregistered
margin of the GRU on tune for the primary pristine family, a frozen preregistration, a clean test-partition
run, and the critic's results review. The decision is the director's from exploratory evidence; it is not
a confirmatory result and should not be quoted as one.
