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

## 11. Amendment A3-1 (2026-11-04) — after the critic's design review; no confirmatory data seen

The independent critic's design review (`INDEPENDENT_SCIENTIFIC_AUDIT.md`) found seven items blocking
every freeze and two blocking any Gate A2/B freeze. Changes, each implemented in core code with tests:

1. **One joint freeze per candidate** (S-2). Gates A2, B and C for a candidate are preregistered together
   and run once on the same test participants. At most two candidate freezes in Phase III; multiplicity is
   then Holm-corrected across both. A modified candidate is never re-frozen on the same participants.
2. **Gate B1 uses a separately trained no-adaptation twin** (G-1): `pop_gru` with the candidate's backbone
   configuration, trained by core code on the same dev data. The candidate's own `frozen` mode is reported
   as a diagnostic only. Adaptation modes are owned by the harness (`core/adaptation.py`): adapt = the
   person's own prefixes; frozen = the prior state; donor = a **length-matched online** schedule over another
   participant's prefixes (G-3).
3. **Gate B2 is specified** (G-2, G-6): same backbone configuration as the candidate; adapter pool
   {`ft_head`, `ft_full`, `bayes_head`, `exemplar_mix`, `reptile_gru`, `cavia_gru`}; the best adapter per
   family is chosen on tune; B2, like B1, requires the primary family.
4. **Gate rules are code** (`core/gates.py`, S-1): counted families and the primary come from the freeze
   record; a counted family without results enters Holm with p = 1 and fails.
5. **p-values** (S-4): each family's p is max(one-sided bootstrap p with 20,000 resamples, one-sided paired
   t-test p).
6. **Comparator adequacy before any freeze** (F-1): `pop_gru` must reach tune NLL ≤ the Phase II `ref_gru` in
   every family where one exists; ≤ 25 % of the selected comparator's fits may stop at the epoch cap; a best
   comparator selected at the largest capacity in its grid gets one extended search (still ≤ 24 configs).
   Comparator grids use the full 24-configuration cap. The pool adds `pop_gru_ens` (5-seed ensemble).
7. **Configuration counting** (F-3): all tune-evaluated configurations of a candidate architecture across
   exploratory runs and variants are counted; above 24, comparators receive an equal-count search first.
8. **Counted families are explicit** (L-4): `two_step` counts only if before the freeze its parser reaches
   ≥ 95 % parse rate on dev, the encoding-leak check is clean and the critic signs off; the freeze record lists
   the counted families.
9. **Isolation and causality** (L-1, L-2, L-3): every participant is predicted by a fresh model copy with an
   opaque key; the strict check (key independence, no mutation, prefix truncation with a flipped action in
   every mode and the donor path, 20 sequences per family, tolerance 1e-5 log-prob — measured float noise
   1.2e-7) excludes violators from all comparisons.
10. **Test lock** (T-1, T-2, T-4, T-6): fixed freeze directory; committed freeze record plus a separate committed
    critic approval; matching preregistration hash; code unchanged since the frozen commit; clean tree; a
    single-run access log; two processes (fit stage on dev/tune persists hashed weights; test stage loads them);
    pinned partition-file hashes; overlapping family names refused; the exposure list fails closed; Phase II
    tooling can no longer unlock any test partition.
11. **Determinism and compute** (F-2, F-4): fits stopped by the CPU budget are flagged and cannot be frozen;
    prediction cost (ms per trial) is reported for every model.
12. **Gate D and Gate E** (G-4, G-5): their designs will be fixed by a further amendment before the
    continual-transfer lane runs any experiment; until then neither gate can be frozen.

Power (S-3): at plausible spreads B2 is under-powered (pass probability 0.07–0.92 at 1.5× the margin
depending on the paired SD); under no effect every rule passes with probability ≤ 0.002. Margins are not
changed; the limitation is reported with any B result.

## 12. Amendment A3-2 (2026-11-04) — Gate D and Gate E designs (critic G-4, G-5); no data seen for either

**Gate D — cross-domain transfer (leave-one-family-out).**

* Targets: the counted families (restless4, garcia, described; two_step if counted). Sources for target T: the
  dev partitions of every other generic family (bandit2, srl3, horizon, casino, wpt, garcia, described,
  restless4, two_step when registered) except T. MCPL/IGT are excluded (different trial format).
* **History-only inputs**: the current- and previous-context feature slots are zeroed for every model in Gate D,
  in pretraining and on the target, because feature slots mean different things across families (G-4).
* k ∈ {0, 16, 64} target participants, drawn from the target's dev partition by a seeded sampler, 5 draws per
  k, scores averaged over draws per test participant. Fine-tuning epochs and learning rate are selected on the
  target's tune partition from a fixed grid of ≤ 6 settings shared by all models.
* Models at equal budget: the candidate pretrained on the sources then adapted on k; a **specialist** of the
  same class trained only on the k target participants; a **pretrained `pop_gru`** on the sources fine-tuned on
  k; the **strongest low-data baseline** `hb_cog` with its population prior fitted on the k target participants.
* Pass (confirmatory): at k = 16 the candidate beats the specialist, the pretrained `pop_gru` and `hb_cog` by
  ≥ 0.02 nats/choice each (Holm over counted targets, max(bootstrap, t) p) in ≥ 2 counted targets. k = 0 and
  k = 64 are reported. Tune stop rule: the same margins on tune.

**Gate E — continual learning (synthetic switching suite with a known reference solution).**

* Generator (`hpe.population.continual`): sequences of 300 trials; K = 4 options; 3 latent contexts per
  sequence, each a Bernoulli reward vector (one option at 0.8, the others drawn from [0.15, 0.45]); blocks of
  20–40 trials; context order revisits earlier contexts (≥ 2 returns per sequence); context identity is not
  observed. Reference solutions: (i) an oracle that knows the context and its reward vector (upper bound);
  (ii) the Bayes filter over contexts with Beta posteriors (the optimal learner given the generative model).
* Learners within a sequence: online fine-tuning of a small network (no protection), EWC, experience replay,
  the meta-learned recurrent learner (F4, weights frozen at test, adapting in context), and any candidate.
* Metrics: post-switch adaptation (mean reward in the 10 trials after a switch, as a fraction of the oracle's);
  savings on return to a previously seen context (first 10 trials on return minus first 10 trials on first
  exposure); forgetting (reward on a return minus reward at the end of that context's previous block).
* Seeds: development 0–199; **confirmatory 3000–3049, untouched** (no generation or evaluation of seeds ≥ 3000
  before a freeze). Pass: the candidate's savings exceed fine-tuning, EWC and replay (paired 95 % CI over the 50
  confirmatory seeds excludes 0, Holm over the three) and its post-switch adaptation is not more than 10 %
  below the best of them. A synthetic pass does not establish human validity.
