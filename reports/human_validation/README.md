# Human-data validation (Gate 4) and causal grounding (Experiment 7)

Everything in this folder comes from **real, externally sourced human data** with verified licences
(see `docs/DATA_SOURCES.md`). Nothing here is a simulation result, and nothing here validates the
simulator's world. Raw external data is never committed; only derived metrics are.

| Experiment | Data | Question | Preregistration | Status |
|---|---|---|---|---|
| [`exp07_human_heldout`](exp07_human_heldout/report.md) | Psych-101 (Apache-2.0): MCPL (232 participants) primary, Iowa Gambling Task (511) secondary | Which policy classes predict held-out participants' learning choices? | `experiments/exp07_human_heldout.yaml`, hash `f30610430c3b847f` | **inconclusive** (primary contrast not evaluable: developmental class not in this build); secondary contrasts passed |
| [`exp07b_causal_grounding`](exp07b_causal_grounding/report.md) | HeartSteps V1 micro-randomised trial (CC-BY-4.0), 37 participants, 6,590 available decision points | Does our causal-estimation pipeline recover the published suggestion effect? | `experiments/exp07b_causal_grounding.yaml`, hash `db680486f77f864f` | **passed** |

## Headline numbers (copied from the generated `result.json` files)

### exp07b — HeartSteps V1 (real randomised data)

| term | our estimate | 95% participant-cluster bootstrap CI | p (sandwich, t36) | published (abstract, PMID 30192907) |
|---|---|---|---|---|
| any suggestion vs none | **+12.2%** | [−0.5%, +27.4%] | 0.083 | +14% (p = .06) — inside our CI |
| walking suggestion vs none | **+21.2%** | [+4.2%, +42.0%] | 0.024 | +24% (p = .02) — inside our CI |
| anti-sedentary vs none | +1.4% | [−13.3%, +20.8%] | 0.869 | "no detectable effect" |

Sensitivity: complete-case outcome gives +16.1% [+2.0, +32.8] (p = 0.043); no control covariates
+10.3% [−3.1, +25.9]; nominal 0.6/0.3/0.3 centring changes nothing material. An exploratory
time-moderated model shows the effect declining with study day (slope −0.0126 log-units/day,
bootstrap CI [−0.0246, −0.0003]), qualitatively matching the paper's statement that the effect
diminished over time. Caveats: the public data has 37 participants (the paper enrolled 44), and the
paper's exact control-covariate set could not be verified (only the abstract was accessible).

### exp07 — held-out human choice prediction

**Primary contrast (developmental vs strongest non-developmental model): NOT EVALUATED.** The run was
produced in the `science/human-validation` worktree, where only the reference `random` policy is
registered; `static_persona`, `memory_only`, `plain_rl` and `developmental` are built by other lanes
and were reported as *unavailable*, not imputed. Status is therefore **inconclusive** by the
preregistered rule. Re-running the same command after integration evaluates them automatically.

What *was* measured — teacher-forced held-out NLL per choice (lower is better), primary split seed
20261104, 95% bootstrap CI over held-out participants:

| model | MCPL (69 held-out; chance 2.1972) | IGT (153 held-out; chance 1.3863) |
|---|---|---|
| `ref_instance_based` (adaptive, memory of instances) | **1.5052** [1.3876, 1.6162] | **1.2809** [1.2532, 1.3066] |
| `ref_population_static` (static; cue→choice table from *other* participants) | 1.6023 [1.5158, 1.6927] | n/a |
| `ref_delta_rule` (adaptive, Rescorla–Wagner / LMS) | 1.8020 [1.7219, 1.8858] | 1.3041 [1.2843, 1.3239] |
| `ref_bias` (static choice frequencies) | 1.9363 [1.9067, 1.9652] | 1.3504 [1.3410, 1.3593] |
| `random` | 2.1972 (= chance) | 1.3863 (= chance) |

Secondary preregistered contrast *best adaptive − best static* (both chosen on TRAIN):

| split seed | MCPL | IGT |
|---|---|---|
| 20261104 (primary) | −0.0970 [−0.1359, −0.0545] passed | −0.0696 [−0.0949, −0.0455] passed |
| 1 (robustness) | −0.0801 [−0.1232, −0.0375] passed | −0.0809 [−0.1026, −0.0597] passed |
| 2 (robustness) | −0.0981 [−0.1444, −0.0528] passed | −0.0973 [−0.1259, −0.0692] passed |

Interpretation, limited to what was tested: on held-out real participants, a model that adapts to each
person's own experience predicts their next choice better than the best fixed policy, including one
fitted to other people's cue→choice behaviour. This is evidence about *reference cognitive models*,
not yet about this repository's developmental learner. Limitations: several fitted values sit on the
edge of the preregistered grids (MCPL instance model h = s = 0.5; delta-rule lapse 0.3; IGT
instance lapse 0.3), so the reference models may be under-fit; registry policy classes will receive
only lapse (and IGT reward-scale) fitting, which disadvantages them relative to the gridded reference
models; teacher-forced one-step likelihood is not generative realism.

## Reproduce

```bash
python datasets/adapters/psych101.py       # download (~859 MB) + filter + verify sha256
python datasets/adapters/heartsteps.py     # download (~4.4 MB) + verify sha256
python -m hpe.evaluation.human run experiments/exp07_human_heldout.yaml
python -m hpe.evaluation.human run experiments/exp07b_causal_grounding.yaml
python -m hpe.evaluation.human analyze reports/human_validation/exp07_human_heldout   # regenerate from logs
```

Each folder contains `definition.json` (written before analysis), the derived logs
(`fits.json` + `outcomes.jsonl.gz`, or `estimates.json`), and `result.json` / `report.md` /
`figures/` regenerated from those logs. `tests/evaluation/human/test_human_pipeline.py` checks that the
committed results regenerate identically.

## Attribution

* Binz, M. et al. (2024). *Centaur: a foundation model of human cognition.* arXiv:2410.20268 — Psych-101 dataset (Apache-2.0). MCPL data: collsiöö2023MCPL; IGT data: steingroever2015data (as compiled in Psych-101).
* Klasnja, P. et al. (2019). *Efficacy of Contextually Tailored Suggestions for Physical Activity: A Micro-randomized Optimization Trial of HeartSteps.* Annals of Behavioral Medicine 53(6):573–582. doi:10.1093/abm/kay067 — HeartSteps V1 data (CC-BY-4.0).
