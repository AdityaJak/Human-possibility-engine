# Phase III Research Thesis — Population-Trained Developmental Intelligence and Individual Adaptation

_Charter written by the Research Director on 2026-11-04, **before any Phase III model was fitted**.
Gates, thresholds, counted families and statistics below are fixed. Any later change is an amendment:
dated, justified, recorded in `EXPERIMENT_REGISTRY_V3.md`, and never made after a confirmatory result
has been seen. Phase I and II results stand unchanged._

## 1. Where Phase II left us

* The Phase II decision was **PIVOT** (`docs/phase2/BREAKTHROUGH_STATUS.md`). Candidate A is closed.
* A 666-parameter GRU fitted on *other people* beats every per-participant learner in 8/9 human
  families (X-009) and Candidate A by 0.18–0.27 nats/choice on all three pristine families (X-010).
* A recurrent population model already adapts to the person in context: its hidden state is updated by
  every choice and outcome. Any Phase III claim of "individual adaptation" must therefore add value
  **beyond in-context recurrent adaptation**, not merely beyond a static population average.

## 2. Hypotheses

**H3 (core, falsifiable).** An architecture that learns transferable structure across people and
continually updates an individual representation from the person's own experience predicts held-out
human choices better than (a) strong population-only sequence models and (b) strong individual-only
models, on appropriate human learning tasks.

Sub-hypotheses, one per gate:

| | Hypothesis | Gate |
|---|---|---|
| H3-A | The advantage is caused by learning (population training on real contingencies; personalisation on the person's own history), not by scripts or leakage | A |
| H3-B | Explicit individual adaptation improves on the frozen population model *and* on conventional adapters (fine-tuning, Bayesian updates, recurrent state, exemplar memory) | B |
| H3-C | The resulting model beats the strongest tune-selected baseline on genuinely held-out people | C |
| H3-D | Structure learned on some task families transfers usefully to an unseen family | D |
| H3-E | The model adapts to change while limiting catastrophic forgetting better than established continual-learning methods | E |
| H3-F | Affective or social latent variables improve held-out prediction where such variables are observed | F |
| H3-G | Model-guided decisions improve real outcomes | G |

## 3. Data, partitions and independence (fixed now)

Source: Psych-101 (Binz et al., 2024; Apache-2.0), file `prompts_training.jsonl`
(sha256 `7a370a94…`). Local Phase III subset: 18 study prefixes, 4,088 transcripts, sha256
`76f335e0…` (never committed).

| Family | Studies (independent publications) | Partition source | Test n | Role |
|---|---|---|---|---|
| `restless4` | bahrami2020four (4-armed restless bandit) | Phase III file, seed 20261106 | 347 | **counted, primary** |
| `garcia` | garcia2023experiential exp1–4 | Phase II file, seed 20261105, test never accessed | 140 | **counted** |
| `described` | plonsky2018when | Phase II file, test never accessed | 87 | **counted** |
| `two_step` | kool2016when, kool2017cost, zorowitz2023data | Phase III file | 273 | **counted only if its parser passes validation on dev before the freeze**; otherwise excluded, decided before any test access |
| `bandit2`, `srl3`, `horizon`, `casino`, `wpt` | gershman2018, gershman2020, wilson2014, lefebvre2017, speekenbrink2008 | Phase II file | 9–88 | secondary (reported, not counted) |
| `horizon_external` | sadeghiyeh2020, somerville2017, feng2021 | Phase III file | 73 | external-replication analysis: models fitted on wilson2014 only; dev/tune of these studies used for parser validation only |
| `mcpl`, `igt` | collsiöö2023, steingroever2015 | Phase II file | — | **contaminated** for confirmatory claims (Phase I scored test participants); development data only |

Independence: each counted family comes from a different original publication with disjoint
participants. All reach us through one compilation (Psych-101), which shares a text format and
preprocessing; this is stated wherever Gate C is reported. A dataset outside Psych-101 is sought for
the strongest claim (`HUMAN_DATA_VALIDATION.md`); none is assumed.

Contamination of external models: every participant in `prompts_training.jsonl` was in Centaur's
training data, so Centaur/Minitaur scores on these participants would be biased in their favour. They
are not valid comparators on these data (see `ADAPTATION_BENCHMARKS.md`).

Partition rules: dev and tune may be read by every lane. Test transcripts are never parsed unless a
freeze record `reports/phase3/freeze/<experiment_id>.json` exists, signed off by the critic. Each frozen
preregistration is run once.

## 4. Model classes

Candidates (design team; definitions and maths in `POPULATION_ADAPTATION_ARCHITECTURES.md`):
F1 population-pretrained learner with an explicit individual latent adapted online (slow trait vs fast
context); F2 population world model + individual policy; F3 hierarchical episodic–parametric model with
learned arbitration; F4 continual meta-learner trained on changing task sequences; F5 cognitive–affective–
social extension of the most promising of F1–F4 (only where affect/social variables are observed).

Comparator pool (evaluation team; `ADAPTATION_BENCHMARKS.md`), every member tuned under the same caps:
1. Phase II numpy `ref_gru` (as committed) and its torch re-implementation `pop_gru` at hidden 16/32/64;
2. `pop_lstm` at matched capacity; 3. a small causal transformer `pop_tf`;
4. `pop_gru` + per-person online fine-tuning (head-only and full); 5. `pop_gru` + Bayesian head update
with an empirical-Bayes population prior; 6. `pop_gru` + exemplar-memory mixture; 7. first-order
meta-learned (Reptile) GRU + fine-tuning; 8. hierarchical Bayesian cognitive models (family-appropriate
learner, population prior over parameters fitted on dev, per-person online posterior); 9. the Phase II
per-person models (PVL-Delta, exemplar, developmental, hybrid exemplar) as reported references.

**Encoding v3** (shared by every sequence model): current context features, available-action mask,
previous action, previous outcome of every option with an observed-outcome mask, previous chosen
outcome, previous context, block-start and instructed-trial flags. Counterfactual feedback that
per-person models already received in Phase II is now given to sequence models too.

**Causality.** Every model must pass `hpe.population.core.conformance.check_causal`: perturbing any
future choice or outcome leaves all earlier predictions bit-identical.

## 5. Metric and statistics (all gates using human data)

* Metric: per participant, mean negative log-likelihood (nats) of their free choices under teacher
  forcing, after lapse mixing `p' = (1−λ)p + λ/K` with λ selected on tune from
  {0, .01, .02, .05, .1, .2, .3}. Instructed trials are inputs, not scored.
* Contrast: paired difference (model − comparator) per participant, averaged over the scored partition;
  95 % percentile bootstrap CI (2,000 resamples, seed from `hpe.core.rng`); one-sided bootstrap p-value.
* Multiplicity: Holm across the counted families within a gate (and across candidates if two are
  frozen); a family "passes" only if the Holm-adjusted p < 0.05 **and** the point estimate meets the margin.
* Selection: the comparator for each family is the pool member with the lowest **tune** NLL, fixed at
  the freeze. Comparator and candidate get identical information, partitions, lapse grid and caps.
* Calibration (ECE), top-1 accuracy, NLL by trial quarter (sample efficiency), parameters and
  CPU-seconds are reported for every model; they do not decide gates.

Margin justification: Phase II paired SDs of NLL differences between similar models were 0.03–0.10
nats/choice; with test n = 87–347 the standard error is 0.002–0.011, so 0.02 nats/choice (≈ 2–4 nats per
participant, a likelihood ratio of ≥ 7 per person) is both meaningful and detectable. Adaptation effects
on top of a recurrent model are expected to be smaller, hence the smaller Gate B margins.

## 6. Gates (thresholds fixed now)

| Gate | Pass condition (confirmatory, held-out test participants or untouched seeds) |
|---|---|
| **A — learned capability** | (A1) candidate trained on real dev data beats the same architecture trained on dev data with outcomes permuted within participant by ≥ 0.02 nats/choice in ≥ 2 counted families; (A2) adapting on the person's own history beats adapting on a mismatched participant's history (same family, same length) with Holm p < 0.05 in ≥ 2 counted families; (A3) the critic's code audit finds no family-specific outcome rule or test access |
| **B — individual adaptation** | (B1) adaptation on vs the same frozen population model (recurrent state still running): ≥ 0.01 nats/choice better in ≥ 2 counted families incl. the primary; (B2) candidate's adaptation vs the best conventional adapter on the same backbone (fine-tune head/full, Bayesian head, exemplar mixture, Reptile): ≥ 0.005 better in ≥ 2 counted families. B passes only if B1 and B2 pass |
| **C — human behavioural prediction** | candidate vs the strongest tune-selected comparator: ≥ 0.02 nats/choice better with Holm p < 0.05 in ≥ 2 counted families from independent studies, including the primary, and not worse by > 0.02 in any counted family. All families reported |
| **D — cross-domain transfer** | leave-one-family-out: pretrain on the other families' dev data, adapt with k = 16 target-family dev participants; beat both a specialist of the same class trained on those 16 and a pretrained `pop_gru` fine-tuned on them by ≥ 0.02 nats/choice (Holm) in ≥ 2 counted target families |
| **E — continual learning** | on the synthetic switching suite (`ADAPTATION_BENCHMARKS.md`, reference solution known): retention after returning to a previously seen context better than online fine-tuning, EWC and experience replay (paired CI over ≥ 20 untouched seeds excludes 0) while post-switch adaptation is not slower than the best of them by more than 10 %. Human non-stationary families (`restless4`, `two_step`) are reported by trial quarter but do not decide E |
| **F — affective/social contribution** | on a licensed human dataset where affect or social variables are observed: the mechanism beats a matched-complexity control (same added parameters fed permuted variables) by ≥ 0.02 nats/choice, CI excluding 0, on held-out participants. Untestable without such data |
| **G — causal decision support** | improvement in a real outcome in an ethically reviewed prospective randomised study. Prediction results cannot pass it. Not executable in Phase III |

Exploratory results (dev/tune, untouched-seed pilots) never pass a gate.

## 7. Stop rules and freeze

* A candidate is frozen for Gate C only if, on **tune**, it beats the best tune comparator by ≥ 0.02 in
  ≥ 2 counted families including the primary. For Gate B the tune rule is B1 ≥ 0.01 in ≥ 2 families.
  Otherwise no test access for that gate; the stop is recorded.
* At most two candidates may be frozen; multiplicity is then corrected across both.
* Freeze record contents: experiment id, preregistration hash, code commit, model configs and hashes,
  comparator configs, lapse values, critic approval reference. The confirmatory run is executed once by the
  director's harness, never by a design lane.

## 8. Decision rules (Stage 8)

* **GO**: a candidate passes A, B and C (its claimed advance) and survives the critic's results review.
  Synthetic-only passes (E) are reported but cannot by themselves justify GO.
* **PIVOT**: no gate-bearing pass, but dev/tune evidence identifies a specific mechanism with a measured,
  ablation-supported effect worth a new preregistration.
* **NO-GO** (for this hypothesis): every candidate fails the tune stop rule and no ablation shows a
  supported mechanism after the budget below is spent.
* A positive result that vanishes against the strongest adaptive baseline is not a breakthrough. A model
  that needs much more compute must show value per unit of compute; costs are always reported.

## 9. Budgets

CPU only (4 cores, 15.7 GiB). Per model and family: ≤ 24 configurations and ≤ 45 CPU-minutes of fitting;
identical caps for candidates and comparators. Phase III total ≤ 60 CPU-hours. No GPU, no paid API, no
language-model calls in any model. Every attempt, including abandoned ones, is logged in
`EXPERIMENT_REGISTRY_V3.md`.

## 10. Lanes and ownership (AGENTS.md §1 addendum)

| Lane | Owns | Must not |
|---|---|---|
| director (orchestrator) | `packages/hpe/population/core/`, `docs/phase3/` (except lane reports), `experiments/phase3/`, `reports/phase3/{partitions,freeze,confirmatory}/`, `scripts/phase3_*.py` | write candidate code |
| `research/adaptation` (design) | `packages/hpe/population/candidates/`, `tests/population/candidates/`, `reports/phase3/candidates/`, `POPULATION_ADAPTATION_ARCHITECTURES.md` | read test; edit core or baselines |
| `science/validation-v3` (evaluation) | `packages/hpe/population/baselines/`, `packages/hpe/population/families/`, `tests/population/baselines/`, `reports/phase3/baselines/`, `ADAPTATION_BENCHMARKS.md`, `HUMAN_DATA_VALIDATION.md` | edit candidates |
| `research/continual-transfer` | `packages/hpe/population/continual/`, `reports/phase3/continual/`, `TRANSFER_LEARNING_RESULTS.md` | edit core |
| `research/affective-social` | `packages/hpe/population/affect/`, `reports/phase3/affect/`, `AFFECTIVE_SOCIAL_VALIDATION.md` | edit core |
| `research/causal` | `CAUSAL_DECISION_RESEARCH.md`, `reports/phase3/causal/` | — |
| `audit/critic-v3` | `reports/phase3/critic/`, `INDEPENDENT_SCIENTIFIC_AUDIT.md`; veto on freezes | production code |

No lane certifies its own claim; the critic reviews design before any freeze and results after.
