# Research Results v2 — Phase II

_Compiled by the director from the exploratory and confirmatory ledgers in `EXPERIMENT_REGISTRY.md`;
every number traces to the artifact directory named in its ledger row. Phase I results stand
unchanged in `RESEARCH_RESULTS.md`. Exploratory = dev/tune partitions, selection-optimistic; it
supports no claim._

## 1. One-page summary

| Question | Evidence | Verdict |
|---|---|---|
| Is Phase I reproducible? | all 8 simulation results re-analysed byte-identically (X-001) | yes |
| Why did the Phase I learner lose on human cue learning? | loss concentrated in the most predictable participants (r = +0.77 with predictability); categorical head cannot express the graded ordinal mapping (X-002) | diagnosed |
| Does Candidate A (hybrid parametric + exemplar, ordinal output, learned gate) fix it? | MCPL dev: −0.040 nats vs instance-based [−0.062, −0.016]; fixed-gate and no-ordinal controls lose the advantage; IGT advantage retained (X-003) | yes on dev, vs per-participant references |
| Does it beat strong baselines on harder synthetic tasks? | ≤ best exemplar baseline on 3/5 v2 families; `conjunction` and the candidate-free `composition` family unsolved by every architecture (X-004, X-008) | **no** |
| Is learning genuine (Gate A controls)? | permuted-feedback gain ≈ 0 for every learner; frozen-copy / age-script gains ≈ +0.39 (X-007) | yes on dev |
| What is the strongest human-prediction baseline? | a population-fitted GRU beats every per-participant learner in 8/9 Psych-101 families, +0.07 to +0.45 nats (X-009) | the bar moved |
| Does Candidate A approach the GRU on pristine families? | no: hybrid − GRU = +0.196 (bandit2), +0.178 (garcia), +0.269 (described), all CIs exclude 0; ≤ 0.016 better than the Phase I learner; level with PVL-Delta; beats exemplar and memory-only references (X-010) | **no** — stop rule applied, no freeze |
| Is the social pathway repaired? | friendships +1.40 vs Phase I from information exchange (reward-only control 0.03); information worth +0.021 accuracy; misinformation harm removed by reliability-tracked trust; connected − isolated still ≈ 0 (family substitutes) | mechanism repaired; H3 contrast not an information contrast in this world |
| Is the causal estimator valid; does tailoring help? | WCLS coverage 0.93–0.96, bias ≤ 0.003 (X-005); test: no heterogeneity (C-001), suggestions help (+0.167) but tailoring adds nothing over always-send (+0.021 [−0.107, +0.156]) (C-002) | estimator valid; Gate E untouched |
| Affect? | not executed (credit limit) | no evidence |
| State of the art? | Centaur/Minitaur not runnable here; published figures non-comparable; Simile undocumented; our learners lose to a 2k-parameter GRU | **no SOTA claim** |

## 2. Gate status (confirmatory ledger)

No gate-bearing confirmatory evaluation has been run, by design: the stop rule was applied after X-010.
Gates A and C are supported on dev; B is not evaluable (hard families unsolved by everyone); D-pop is
contradicted by dev evidence for every architecture we have, Candidate A included (X-009, X-010); D-ind
is contradicted on dev for Candidate A (no family won by ≥ 0.02); E not started; F not tested. Stage 8
decision: **PIVOT** (`BREAKTHROUGH_STATUS.md`).

## 3. Detailed results

Human data (Psych-101, dev/tune only): `HUMAN_VALIDATION_V2.md` §6–§7 and
`reports/phase2/human/p2_human_dev_pilot/`; Candidate A on MCPL/IGT: `reports/phase2/cognitive/human_dev/`.
Synthetic: `reports/phase2/cognitive/synthetic_dev/`, `reports/phase2/evaluation_v2/`.
Social: `reports/phase2/social/`, `SOCIAL_MECHANISM_NOTES.md`. Causal: `reports/phase2/causal/`,
`CAUSAL_VALIDITY_REPORT.md`. Competitive context: `STATE_OF_THE_ART_BENCHMARK.md`.

## 4. Negative results (reported with equal prominence)

1. No architecture learns parity or XOR-type composition at the budgets used (X-004, X-008).
2. Per-participant online learners — all of ours — lose to a population-fitted sequence model on
   8/9 human families (X-009).
3. Connected vs isolated societies still do not differ in skill after the social repair (social lane).
4. No heterogeneous treatment effect replicates on HeartSteps test; tailoring ≈ always-send (C-001/2).
5. Phase I's IGT advantage disappears against the stronger baseline set (X-009).
6. Candidate A, the architecture built this phase, is 0.18–0.27 nats behind the population-fitted GRU on
   every pristine human family and does not beat the best per-participant model by the preregistered
   margin anywhere (X-010).

## 5. Compute

Phase II CPU used: cognitive 0.36 h, human-validation ≈ 1.04 h, evaluation-v2 ≈ 0.5 h, social
≈ 1.03 h, causal 0.07 h, director diagnostics < 0.1 h, pilot B 0.44 h; ≈ 3.5 CPU-h in all; 0 GPU, 0 paid
API calls.
