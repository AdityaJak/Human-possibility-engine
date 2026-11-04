# Human Validation v2 — protocol, families, baselines, dev pilot

_Lane `research/human-validation-v2`. Every number in §6–§7 is produced by
`python -m hpe.evaluation.human run2 experiments/phase2/human/dev_pilot.yaml` (log:
`reports/phase2/human/p2_human_dev_pilot/run.log`) and rendered by `scripts/phase2_sota_table.py`.
Nothing here is a confirmatory result: the report partition is **tune**, on which hyper-parameters
were also selected._

## 1. Protocol

| Element | Rule | Where enforced |
|---|---|---|
| Partitions | participant-level dev 40 % / tune 20 % / test 40 %, split seed 20261105, stratified by experiment file; frozen in `reports/phase2/partitions/psych101.json` (test SHA-256 `7dc2e779…`) | `hpe.evaluation.human.partitions.Partitions.load` re-verifies the digest on every load |
| Test lock | records of the test partition are skipped **before parsing** unless `reports/phase2/freeze/<experiment_id>.json` exists with `experiment_id, preregistration_hash, code_version, frozen_at` | `partitions.check_partition_access`, `iter_partition_records`; tests `test_v2_partitions_models.py::test_test_partition_is_locked_without_freeze_record`, `…opens_only_with_valid_freeze_record`, `…tampered_partitions_file_is_rejected` |
| Equal tuning | every model class: ≤ 24 hyper-parameter configurations (reward-scale variants count), committed in the YAML before tuning; the same lapse grid for all; selection by mean per-participant NLL on tune; fits at a grid edge are flagged | `harness2.evaluate_model2` refuses larger grids (`error: grid has N configurations > 24`) |
| Fitting | models that need data (pooled frequencies, population table, GRU) are fitted on **dev** only; the GRU early-stops on a held-out fifth of dev, never on tune | `models2.fit_gru` |
| Scoring | teacher forcing: predict p(human choice), then learn from the human's action and the outcome the human saw; instructed (forced) trials are learned from, never scored; per-choice NLL in nats, lapse-mixed | `harness2.run_participant2`, `score_participant` |
| Reported | NLL mean with participant-level bootstrap 95 % CI; Δ vs chance (ln K); top-1 accuracy; top-label ECE (10 bins); NLL by trial quartile (sample efficiency); NLL by experiment file (robustness); NLL by participant-predictability tercile (subgroups); parameter count; CPU seconds | `runner2.analyze_v2`, `harness2.expected_calibration_error` |
| Regeneration | `python -m hpe.evaluation.human analyze2 <dir>` rebuilds summary.json and report.md from fits.json + outcomes.jsonl.gz only | verified byte-identical on the smoke run |

Phase I contact disclosure: every MCPL and IGT participant was fitted on or scored in Phase I; the seven
new families below are pristine. A second dataset (§8) is still required for the "independent
population" clause of Gate D.

## 2. Families, parsers and encodings (Psych-101, Apache-2.0)

Parsers: `hpe.evaluation.human.psych101_v2` (new families) and `psych101` (Phase I MCPL/IGT). Every
family becomes `GenericTrial` records with a 12-feature context in [−1, 1] and a feedback mode, encoded
identically for all models by `models2.encode_generic`. Validation on dev+tune transcripts
(`reports/phase2/human/p2_human_dev_pilot/validation.json`).

| Paradigm key | Psych-101 files | Task | Actions | Context features (12) | Feedback | Reward map to [0, 1] | Notes |
|---|---|---|---|---|---|---|---|
| `mcpl` | collsiöö2023MCPL exp1–3 | multiple-cue probability learning | 9 ordinal levels | one-hot cue1 (5), one-hot cue2 (5), scaled cues (2) — Phase I | full while shown, then none | — | Phase I encoding |
| `igt` | steingroever2015data exp1–3 | Iowa gambling task | 4 decks | constant | bandit | 0.5 + net/(2·scale), scale ∈ grid | Phase I encoding |
| `bandit2` | gershman2018deconstructing exp1–2 | two slot machines (one risky, one always 0), 20 games × 10 trials | 2 | bias, trial-in-game, game-start flag | bandit | (points + 3)/6 | machine order from the instruction text |
| `srl3` | gershman2020reward exp1 | stimulus → response learning, 2–6 stimuli, 13 games, 3 responses | 3 | one-hot stimulus (8), bias | bandit | (points + 1)/2 | some variants give −1 |
| `horizon` | wilson2014humans exp1–5, waltz2020differential exp1 | 4 instructed + 1 or 6 free choices per game | 2 | bias, horizon flag, trial-in-game, game-start | bandit | points/100 | instructed trials forced (unscored) |
| `casino` | lefebvre2017behavioural exp1–2 | 4 contexts × 2 machines, probabilistic 0/0.5 or ±0.5 | 2 | one-hot casino (4), bias | bandit | (x − min)/(max − min) from the header | machine pair per casino fixed per participant |
| `wpt` | speekenbrink2008learning exp1 | weather prediction, 1–3 of 4 cards | 2 (rainy/fine) | card presence ±1 (4), bias | full (correct outcome) | hit = 1 | verdict text cross-checked against parsed outcome |
| `garcia` | garcia2023experiential exp1–4 | part 1: letter pairs with counterfactual feedback; part 2: letter vs described option, no feedback; part 3 judgements excluded | 2 | 4-bit code per option letter (8), described flag, described p(win), part flag, bias | part 1 full (both outcomes), part 2 none | (x + 1)/2 | letters differ per participant; codes are participant-salted hashes |
| `described` | plonsky2018when exp1 | described gambles, 5 no-feedback + 20 full-feedback repetitions per problem | 2 | per option: EV, SD, min, max (/100), p(gain), unknown-probability flag | none then full (both outcomes) | (x + 100)/200 | "unknown chance" → uniform placeholder + flag; problems whose two options carry the same letter (a transcript defect) are excluded and counted |

Dev+tune parse counts (from the parser-validation run): bandit2 48, srl3 20 (after the −1-point fix),
horizon 154, casino 47, wpt 14, garcia 206, described 126 (3 participants excluded for the identical-letter
defect; see validation.json). Trials per participant range from 96 (casino) to 2,400 (horizon exp1).

Block structure (new game / new problem) is exposed only through `obs.meta["block_start"]`, which
reference models may use (PVL-Delta resets values per game) and registry policy classes do not read.
This is a documented asymmetry, not an oversight: the registry classes are being tested as they are.

## 3. Models

Registry policy classes (`hpe.contracts.registry`): `random`, `static_persona`, `memory_only`,
`plain_rl`, `developmental`; `hybrid_exemplar` is listed so the cognitive-architecture lane's Candidate A
is picked up automatically when registered (reported "unavailable" until then).

Reference models (`hpe.evaluation.human.models2`; Phase I ones in `models`):

| Name | Family / reference | Free parameters (fitted on tune via grid) | Where it applies |
|---|---|---|---|
| `ref_bias` | pooled choice frequencies of dev participants | smoothing | all |
| `ref_population_static` | fixed P(choice | cue1, cue2) from dev participants | smoothing | mcpl |
| `ref_delta_rule` | Rescorla–Wagner / LMS criterion learner; RW + softmax for IGT (Phase I) | α, σ / α, β | mcpl, igt |
| `ref_instance_based` | kernel retrieval over remembered (cues, outcome) instances (Phase I) | h, s, κ / decay, β | mcpl, igt |
| `ref_exemplar` | generalized context model with attention-weighted similarity (Nosofsky, 1986) and error-driven attention learning (ALCOVE; Kruschke, 1992), recency decay, softmax over similarity-weighted outcome evidence | c, β, decay, lr_att (+12 learned attention weights) | all |
| `ref_pvl_delta` | prospect valence learning with delta rule (Ahn, Busemeyer, Wagenmakers & Stout, 2008; Ahn et al., 2014); outcomes = 2·reward − 1 | α (utility curvature), λ (loss aversion), A (learning rate), c (sensitivity, θ = 3^c − 1) | bandit-style families (and, as a control, every family) |
| `ref_bayes_rule` | Bayesian linear rule learner (Bayesian LMS / Kalman learner; Speekenbrink & Shanks, 2010), Gaussian posterior over weights, predictive distribution binned onto the ordered responses | τ² (prior variance), σ (noise) | full-feedback families: mcpl, wpt, garcia, described |
| `ref_gru` | GRU (Cho et al., 2014) with ≤ 2 k parameters, inputs = context + previous action/reward/correct answer, trained by Adam with truncated BPTT, early-stopped on a fifth of dev; the "modern sequence-model baseline" | hidden ∈ {8, 12}, lr ∈ {3e-3, 1e-2}; weights fitted on dev | all |

Gradient correctness of the GRU is unit-tested against finite differences
(`test_gru_gradients_match_finite_differences`).

## 4. Equal-tuning grids

Committed in `experiments/phase2/human/dev_pilot.yaml` before the pilot ran (preregistration hash
`ee15cd6f2150f602`). Registry classes: `developmental` {lr_head × lr_encoder × exploration_base} = 12,
`memory_only` {memory_capacity × beta} = 9, `plain_rl` {lr} = 4; references 4–24 configurations;
IGT grids are halved so that × 2 reward scales ≤ 24. Shared lapse grid {0, .01, .02, .05, .1, .2, .3}.

## 5. What Gate D requires (from `RESEARCH_THESIS_V2.md`)

On the frozen **test** partition, after a freeze record: candidate NLL ≤ best fitted reference − 0.02
nats/choice with a paired 95 % CI excluding 0 in ≥ 2 of 3 preregistered families; not worse by > 0.02 in
any; ECE reported; ≥ 2 datasets. The dev pilot below informs which families to preregister and which
references are strongest; it proves nothing.

## 6. Dev-pilot results (EXPLORATORY — report partition = tune)

Generated by `scripts/phase2_sota_table.py` from `reports/phase2/human/p2_human_dev_pilot/summary.json`
(full per-family tables with CIs, ECE, quartiles, grid edges and CPU seconds: `report.md` in the same
directory). Reproduced verbatim:

# Directly reproduced comparison — `p2_human_dev_pilot` (report partition: **tune**)

_Generated by `scripts/phase2_sota_table.py` from `reports/phase2/human/p2_human_dev_pilot/summary.json`; code `a7f9558727fa14f8b10d01755301d65b8f7753de+dirty`; CPU 3743.7 s; 0 LLM calls. Every model saw identical observations and feedback (teacher forcing); grids ≤ 24 configurations each; shared lapse grid._

## bandit2 (exploratory) — n = {'dev': 32, 'tune': 16}; chance NLL 0.6931

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 0.3894 [0.330, 0.464] | -0.304 | 0.825 | 0.063 | 0.462 | 0.356 | 0.351 | 0.389 | 666 | 35.04 | ok (grid edge: hidden, lr) |
| ref_pvl_delta | reference | 0.5339 [0.474, 0.591] | -0.159 | 0.699 | 0.105 | 0.553 | 0.539 | 0.504 | 0.539 | 4 | 6.4 | ok (grid edge: A, alpha, c, lam) |
| developmental | ours | 0.5618 [0.519, 0.609] | -0.131 | 0.729 | 0.084 | 0.604 | 0.557 | 0.509 | 0.576 | — | 7.41 | ok (grid edge: exploration_base, lr_encoder, lr_head) |
| ref_exemplar | reference | 0.6103 [0.575, 0.647] | -0.083 | 0.662 | 0.099 | 0.641 | 0.611 | 0.557 | 0.632 | 16 | 8.82 | ok (grid edge: beta, c, decay) |
| ref_bias | reference | 0.6807 [0.658, 0.703] | -0.012 | 0.583 | 0.176 | 0.687 | 0.679 | 0.681 | 0.676 | 1 | 0.66 | ok |
| plain_rl | ours (baseline) | 0.6823 [0.661, 0.702] | -0.011 | 0.569 | 0.123 | 0.684 | 0.690 | 0.659 | 0.696 | — | 2.83 | ok |
| random | chance | 0.6931 [0.693, 0.693] | -0.000 | 0.500 | 0.000 | 0.693 | 0.693 | 0.693 | 0.693 | — | 0.81 | ok |
| memory_only | ours (baseline) | 0.7752 [0.694, 0.860] | +0.082 | 0.599 | 0.197 | 0.787 | 0.786 | 0.672 | 0.855 | — | 8.47 | ok (grid edge: beta, memory_capacity) |
| static_persona | ours (baseline) | 0.8236 [0.711, 0.972] | +0.130 | 0.542 | 0.236 | 0.855 | 0.864 | 0.761 | 0.815 | — | 0.85 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_bayes_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_bayes_rule needs full (corrective) feedback |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.1723 | [+0.1152, +0.2263] | 0.0005 | 1.57 | 16 |
| developmental − ref_gru | +0.1723 | [+0.1178, +0.2228] | 0.0005 | 1.57 | 16 |
| ref_gru − ref_bias | -0.2912 | [-0.3447, -0.2253] | 0.0005 | -2.27 | 16 |

## casino (exploratory) — n = {'dev': 31, 'tune': 16}; chance NLL 0.6931

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 0.5562 [0.485, 0.631] | -0.137 | 0.702 | 0.111 | 0.696 | 0.550 | 0.502 | 0.476 | 666 | 22.2 | ok (grid edge: hidden, lr) |
| developmental | ours | 0.6226 [0.598, 0.651] | -0.071 | 0.651 | 0.124 | 0.670 | 0.634 | 0.597 | 0.590 | — | 4.51 | ok (grid edge: exploration_base, lr_encoder, lr_head) |
| plain_rl | ours (baseline) | 0.6233 [0.589, 0.659] | -0.070 | 0.691 | 0.139 | 0.657 | 0.625 | 0.606 | 0.606 | — | 1.88 | ok |
| memory_only | ours (baseline) | 0.6369 [0.540, 0.740] | -0.056 | 0.693 | 0.143 | 0.620 | 0.628 | 0.649 | 0.650 | — | 5.3 | ok |
| ref_exemplar | reference | 0.6548 [0.628, 0.690] | -0.038 | 0.644 | 0.149 | 0.703 | 0.642 | 0.639 | 0.635 | 16 | 3.49 | ok (grid edge: beta, c, decay) |
| ref_pvl_delta | reference | 0.6791 [0.654, 0.700] | -0.014 | 0.584 | 0.096 | 0.678 | 0.674 | 0.680 | 0.685 | 4 | 4.79 | ok (grid edge: A, alpha, c, lam) |
| random | chance | 0.6931 [0.693, 0.693] | +0.000 | 0.500 | 0.000 | 0.693 | 0.693 | 0.693 | 0.693 | — | 0.44 | ok |
| ref_bias | reference | 0.6950 [0.690, 0.701] | +0.002 | 0.480 | 0.125 | 0.694 | 0.696 | 0.695 | 0.695 | 1 | 0.34 | ok |
| static_persona | ours (baseline) | 0.9507 [0.810, 1.087] | +0.258 | 0.455 | 0.358 | 0.882 | 0.945 | 0.960 | 1.015 | — | 0.68 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_bayes_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_bayes_rule needs full (corrective) feedback |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.0665 | [+0.0055, +0.1235] | 0.032 | 0.54 | 16 |
| developmental − ref_gru | +0.0665 | [+0.0060, +0.1250] | 0.03 | 0.54 | 16 |
| ref_gru − ref_bias | -0.1389 | [-0.2062, -0.0631] | 0.0005 | -0.91 | 16 |

## described (exploratory) — n = {'dev': 85, 'tune': 41}; chance NLL 0.6931

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 0.3720 [0.331, 0.412] | -0.321 | 0.836 | 0.041 | 0.401 | 0.380 | 0.375 | 0.332 | 666 | 224.0 | ok (grid edge: hidden, lr) |
| ref_pvl_delta | reference | 0.6505 [0.636, 0.666] | -0.043 | 0.615 | 0.073 | 0.647 | 0.642 | 0.658 | 0.655 | 4 | 42.23 | ok (grid edge: A, alpha, c, lam) |
| developmental | ours | 0.6552 [0.641, 0.670] | -0.038 | 0.623 | 0.062 | 0.661 | 0.642 | 0.666 | 0.653 | — | 50.1 | ok (grid edge: exploration_base, lr_encoder) |
| plain_rl | ours (baseline) | 0.6636 [0.649, 0.679] | -0.030 | 0.611 | 0.070 | 0.656 | 0.661 | 0.674 | 0.664 | — | 22.13 | ok |
| ref_bayes_rule | reference | 0.6690 [0.656, 0.682] | -0.024 | 0.601 | 0.077 | 0.641 | 0.675 | 0.667 | 0.692 | 15 | 62.25 | ok (grid edge: sigma, tau2) |
| ref_bias | reference | 0.6781 [0.673, 0.683] | -0.015 | 0.587 | 0.046 | 0.676 | 0.677 | 0.678 | 0.682 | 1 | 4.53 | ok |
| ref_exemplar | reference | 0.6799 [0.677, 0.683] | -0.013 | 0.598 | 0.078 | 0.680 | 0.677 | 0.683 | 0.680 | 16 | 304.53 | ok (grid edge: beta, c, decay, lr_att) |
| random | chance | 0.6931 [0.693, 0.693] | +0.000 | 0.500 | 0.000 | 0.693 | 0.693 | 0.693 | 0.693 | — | 4.49 | ok |
| memory_only | ours (baseline) | 0.7757 [0.745, 0.805] | +0.083 | 0.613 | 0.200 | 0.749 | 0.792 | 0.796 | 0.766 | — | 110.49 | ok (grid edge: beta, memory_capacity) |
| static_persona | ours (baseline) | 0.8841 [0.838, 0.929] | +0.191 | 0.506 | 0.276 | 0.885 | 0.895 | 0.910 | 0.847 | — | 7.03 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.2832 | [+0.2367, +0.3285] | 0.0005 | 1.86 | 41 |
| developmental − ref_gru | +0.2832 | [+0.2365, +0.3310] | 0.0005 | 1.86 | 41 |
| ref_gru − ref_bias | -0.3061 | [-0.3467, -0.2629] | 0.0005 | -2.16 | 41 |

## garcia (exploratory) — n = {'dev': 138, 'tune': 68}; chance NLL 0.6931

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 0.4467 [0.411, 0.482] | -0.246 | 0.771 | 0.065 | 0.546 | 0.505 | 0.397 | 0.338 | 666 | 178.57 | ok (grid edge: hidden, lr) |
| ref_bayes_rule | reference | 0.6120 [0.587, 0.640] | -0.081 | 0.650 | 0.115 | 0.587 | 0.559 | 0.610 | 0.692 | 15 | 38.6 | ok (grid edge: sigma, tau2) |
| ref_pvl_delta | reference | 0.6175 [0.592, 0.644] | -0.076 | 0.637 | 0.086 | 0.603 | 0.563 | 0.611 | 0.693 | 4 | 32.25 | ok (grid edge: A, alpha, c, lam) |
| plain_rl | ours (baseline) | 0.6258 [0.605, 0.648] | -0.067 | 0.644 | 0.107 | 0.600 | 0.576 | 0.622 | 0.705 | — | 10.08 | ok (grid edge: lr) |
| developmental | ours | 0.6405 [0.612, 0.669] | -0.053 | 0.646 | 0.135 | 0.588 | 0.564 | 0.645 | 0.765 | — | 24.64 | ok (grid edge: exploration_base, lr_encoder, lr_head) |
| ref_exemplar | reference | 0.6427 [0.626, 0.662] | -0.050 | 0.645 | 0.115 | 0.587 | 0.579 | 0.658 | 0.747 | 16 | 73.31 | ok (grid edge: beta, c, decay, lr_att) |
| memory_only | ours (baseline) | 0.6511 [0.613, 0.689] | -0.042 | 0.645 | 0.138 | 0.641 | 0.618 | 0.637 | 0.708 | — | 36.44 | ok (grid edge: beta, memory_capacity) |
| ref_bias | reference | 0.6515 [0.636, 0.667] | -0.042 | 0.643 | 0.088 | 0.616 | 0.600 | 0.658 | 0.732 | 1 | 2.67 | ok |
| random | chance | 0.6931 [0.693, 0.693] | +0.000 | 0.500 | 0.000 | 0.693 | 0.693 | 0.693 | 0.693 | — | 3.16 | ok |
| static_persona | ours (baseline) | 0.9017 [0.845, 0.957] | +0.209 | 0.516 | 0.296 | 0.886 | 0.867 | 0.925 | 0.929 | — | 4.02 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.1937 | [+0.1666, +0.2192] | 0.0005 | 1.72 | 68 |
| developmental − ref_gru | +0.1937 | [+0.1665, +0.2192] | 0.0005 | 1.72 | 68 |
| ref_gru − ref_bias | -0.2048 | [-0.2333, -0.1770] | 0.0005 | -1.70 | 68 |

## horizon (exploratory) — n = {'dev': 103, 'tune': 51}; chance NLL 0.6931

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 0.4589 [0.415, 0.500] | -0.234 | 0.783 | 0.057 | 0.480 | 0.467 | 0.441 | 0.447 | 666 | 296.72 | ok (grid edge: hidden, lr) |
| ref_pvl_delta | reference | 0.5294 [0.485, 0.576] | -0.164 | 0.765 | 0.103 | 0.548 | 0.531 | 0.512 | 0.526 | 4 | 92.22 | ok (grid edge: A, alpha, c, lam) |
| developmental | ours | 0.6276 [0.615, 0.641] | -0.066 | 0.652 | 0.056 | 0.662 | 0.618 | 0.618 | 0.612 | — | 101.67 | ok (grid edge: exploration_base, lr_encoder, lr_head) |
| ref_exemplar | reference | 0.6318 [0.624, 0.640] | -0.061 | 0.680 | 0.114 | 0.636 | 0.632 | 0.629 | 0.630 | 16 | 447.1 | ok (grid edge: beta, c, decay) |
| plain_rl | ours (baseline) | 0.6879 [0.685, 0.690] | -0.005 | 0.557 | 0.038 | 0.689 | 0.683 | 0.685 | 0.695 | — | 52.28 | ok (grid edge: lr) |
| ref_bias | reference | 0.6912 [0.690, 0.692] | -0.002 | 0.540 | 0.042 | 0.692 | 0.689 | 0.690 | 0.693 | 1 | 7.64 | ok |
| random | chance | 0.6931 [0.693, 0.693] | +0.000 | 0.500 | 0.000 | 0.693 | 0.693 | 0.693 | 0.693 | — | 7.76 | ok |
| memory_only | ours (baseline) | 0.7851 [0.769, 0.803] | +0.092 | 0.574 | 0.189 | 0.792 | 0.755 | 0.774 | 0.820 | — | 279.98 | ok (grid edge: beta, memory_capacity) |
| static_persona | ours (baseline) | 0.8531 [0.812, 0.895] | +0.160 | 0.504 | 0.222 | 0.834 | 0.866 | 0.868 | 0.845 | — | 9.9 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_bayes_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_bayes_rule needs full (corrective) feedback |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.1687 | [+0.1338, +0.2054] | 0.0005 | 1.29 | 51 |
| developmental − ref_gru | +0.1687 | [+0.1334, +0.2007] | 0.0005 | 1.29 | 51 |
| ref_gru − ref_bias | -0.2323 | [-0.2758, -0.1887] | 0.0005 | -1.46 | 51 |

## igt (secondary) — n = {'dev': 204, 'tune': 103}; chance NLL 1.3863

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 1.1001 [1.037, 1.164] | -0.286 | 0.520 | 0.099 | 1.197 | 1.119 | 1.086 | 0.998 | 780 | 140.68 | ok (grid edge: hidden, lr) |
| developmental | ours | 1.1831 [1.128, 1.238] | -0.203 | 0.455 | 0.111 | 1.263 | 1.209 | 1.166 | 1.093 | — | 60.62 | ok (grid edge: exploration_base, lr_encoder) |
| memory_only | ours (baseline) | 1.2840 [1.241, 1.322] | -0.102 | 0.431 | 0.151 | 1.297 | 1.287 | 1.296 | 1.256 | — | 59.8 | ok (grid edge: beta, memory_capacity) |
| ref_instance_based | reference | 1.2890 [1.257, 1.316] | -0.097 | 0.396 | 0.137 | 1.320 | 1.314 | 1.301 | 1.221 | 3 | 49.52 | ok (grid edge: decay) |
| ref_pvl_delta | reference | 1.2973 [1.267, 1.324] | -0.089 | 0.400 | 0.143 | 1.285 | 1.286 | 1.311 | 1.308 | 4 | 28.05 | ok (grid edge: A, alpha, lam) |
| ref_exemplar | reference | 1.3110 [1.298, 1.324] | -0.075 | 0.461 | 0.177 | 1.311 | 1.317 | 1.317 | 1.299 | 16 | 50.47 | ok (grid edge: beta, c, decay) |
| ref_delta_rule | reference | 1.3142 [1.288, 1.337] | -0.072 | 0.381 | 0.149 | 1.298 | 1.313 | 1.332 | 1.314 | 2 | 45.69 | ok (grid edge: alpha) |
| plain_rl | ours (baseline) | 1.3369 [1.320, 1.352] | -0.049 | 0.395 | 0.141 | 1.350 | 1.335 | 1.331 | 1.331 | — | 23.84 | ok |
| ref_bias | reference | 1.3595 [1.350, 1.369] | -0.027 | 0.306 | 0.119 | 1.376 | 1.358 | 1.356 | 1.348 | 3 | 4.97 | ok |
| random | chance | 1.3863 [1.386, 1.386] | +0.000 | 0.250 | 0.000 | 1.386 | 1.386 | 1.386 | 1.386 | — | 4.6 | ok |
| static_persona | ours (baseline) | 1.8336 [1.763, 1.898] | +0.447 | 0.248 | 0.413 | 1.821 | 1.830 | 1.837 | 1.847 | — | 5.86 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_bayes_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_bayes_rule needs full (corrective) feedback |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.0830 | [+0.0644, +0.1021] | 0.0005 | 0.84 | 103 |
| developmental − ref_gru | +0.0830 | [+0.0645, +0.1023] | 0.0005 | 0.84 | 103 |
| ref_gru − ref_bias | -0.2594 | [-0.3196, -0.1993] | 0.0005 | -0.83 | 103 |

## mcpl (primary) — n = {'dev': 93, 'tune': 47}; chance NLL 2.1972

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 1.2481 [1.096, 1.397] | -0.949 | 0.599 | 0.124 | 1.686 | 1.279 | 1.067 | 0.955 | 1065 | 128.86 | ok (grid edge: hidden, lr) |
| memory_only | ours (baseline) | 1.4259 [1.268, 1.587] | -0.771 | 0.597 | 0.225 | 1.899 | 1.490 | 1.220 | 1.090 | — | 32.8 | ok (grid edge: beta, memory_capacity) |
| ref_instance_based | reference | 1.4414 [1.317, 1.573] | -0.756 | 0.598 | 0.239 | 1.852 | 1.469 | 1.254 | 1.187 | 3 | 13.1 | ok (grid edge: h, s) |
| ref_population_static | reference | 1.5684 [1.469, 1.683] | -0.629 | 0.398 | 0.147 | 1.876 | 1.605 | 1.422 | 1.367 | 200 | 3.89 | ok |
| developmental | ours | 1.6927 [1.606, 1.787] | -0.505 | 0.393 | 0.146 | 2.075 | 1.777 | 1.517 | 1.399 | — | 20.4 | ok (grid edge: exploration_base, lr_encoder) |
| ref_bayes_rule | reference | 1.7306 [1.554, 1.905] | -0.467 | 0.488 | 0.253 | 1.955 | 1.741 | 1.623 | 1.601 | 15 | 26.37 | ok (grid edge: sigma, tau2) |
| ref_delta_rule | reference | 1.7358 [1.586, 1.883] | -0.461 | 0.490 | 0.223 | 1.932 | 1.743 | 1.644 | 1.623 | 2 | 27.57 | ok (grid edge: sigma) |
| plain_rl | ours (baseline) | 1.7609 [1.704, 1.827] | -0.436 | 0.321 | 0.111 | 2.044 | 1.837 | 1.614 | 1.544 | — | 9.08 | ok |
| ref_exemplar | reference | 1.8369 [1.777, 1.899] | -0.360 | 0.388 | 0.168 | 2.141 | 1.848 | 1.692 | 1.662 | 16 | 73.93 | ok (grid edge: beta, c, decay, lr_att) |
| ref_bias | reference | 1.9506 [1.917, 1.987] | -0.247 | 0.288 | 0.077 | 1.987 | 1.953 | 1.907 | 1.955 | 8 | 1.73 | ok |
| ref_pvl_delta | reference | 2.1305 [2.098, 2.163] | -0.067 | 0.222 | 0.068 | 2.209 | 2.158 | 2.081 | 2.073 | 4 | 19.96 | ok (grid edge: A, alpha, c, lam) |
| random | chance | 2.1972 [2.197, 2.197] | -0.000 | 0.111 | 0.000 | 2.197 | 2.197 | 2.197 | 2.197 | — | 1.96 | ok |
| static_persona | ours (baseline) | 2.8624 [2.819, 2.906] | +0.665 | 0.105 | 0.465 | 2.837 | 2.856 | 2.893 | 2.864 | — | 2.75 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.4446 | [+0.3550, +0.5361] | 0.0005 | 1.44 | 47 |
| developmental − ref_gru | +0.4446 | [+0.3618, +0.5307] | 0.0005 | 1.44 | 47 |
| ref_gru − ref_bias | -0.7025 | [-0.8523, -0.5463] | 0.0005 | -1.29 | 47 |

## srl3 (exploratory) — n = {'dev': 30, 'tune': 15}; chance NLL 1.0986

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | reference | 0.7411 [0.698, 0.785] | -0.358 | 0.716 | 0.056 | 0.754 | 0.748 | 0.726 | 0.736 | 723 | 104.51 | ok (grid edge: hidden, lr) |
| developmental | ours | 0.8544 [0.822, 0.889] | -0.244 | 0.636 | 0.048 | 0.891 | 0.860 | 0.805 | 0.862 | — | 13.75 | ok (grid edge: exploration_base, lr_encoder) |
| ref_exemplar | reference | 0.9480 [0.936, 0.961] | -0.151 | 0.564 | 0.083 | 0.953 | 0.952 | 0.978 | 0.909 | 16 | 33.87 | ok (grid edge: beta, c, decay) |
| memory_only | ours (baseline) | 1.0017 [0.968, 1.039] | -0.097 | 0.596 | 0.160 | 0.999 | 1.025 | 0.928 | 1.055 | — | 29.14 | ok (grid edge: beta, memory_capacity) |
| ref_pvl_delta | reference | 1.0396 [1.028, 1.051] | -0.059 | 0.406 | 0.039 | 1.041 | 1.053 | 1.062 | 1.002 | 4 | 13.06 | ok (grid edge: A, alpha, c, lam) |
| plain_rl | ours (baseline) | 1.0780 [1.065, 1.092] | -0.021 | 0.437 | 0.100 | 1.073 | 1.083 | 1.069 | 1.087 | — | 7.02 | ok (grid edge: lr) |
| ref_bias | reference | 1.0986 [1.098, 1.099] | -0.000 | 0.325 | 0.029 | 1.098 | 1.099 | 1.100 | 1.097 | 2 | 0.98 | ok |
| random | chance | 1.0986 [1.099, 1.099] | -0.000 | 0.333 | 0.000 | 1.099 | 1.099 | 1.099 | 1.099 | — | 1.19 | ok |
| static_persona | ours (baseline) | 1.4763 [1.395, 1.555] | +0.378 | 0.327 | 0.385 | 1.474 | 1.500 | 1.464 | 1.467 | — | 1.43 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_bayes_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_bayes_rule needs full (corrective) feedback |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_gru | +0.1133 | [+0.0838, +0.1404] | 0.0005 | 1.95 | 15 |
| developmental − ref_gru | +0.1133 | [+0.0828, +0.1399] | 0.0005 | 1.95 | 15 |
| ref_gru − ref_bias | -0.3575 | [-0.4004, -0.3124] | 0.0005 | -4.04 | 15 |

## wpt (exploratory) — n = {'dev': 9, 'tune': 5}; chance NLL 0.6931

| model | group | NLL [95% CI] | Δ chance | top-1 | ECE | Q1 | Q2 | Q3 | Q4 | params | CPU s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| developmental | ours | 0.5659 [0.472, 0.700] | -0.127 | 0.714 | 0.084 | 0.564 | 0.618 | 0.515 | 0.566 | — | 2.1 | ok (grid edge: exploration_base, lr_encoder) |
| plain_rl | ours (baseline) | 0.5691 [0.468, 0.699] | -0.124 | 0.723 | 0.107 | 0.567 | 0.642 | 0.512 | 0.555 | — | 0.87 | ok |
| ref_bayes_rule | reference | 0.5733 [0.482, 0.693] | -0.120 | 0.720 | 0.128 | 0.553 | 0.666 | 0.523 | 0.551 | 15 | 2.43 | ok (grid edge: sigma, tau2) |
| ref_exemplar | reference | 0.5754 [0.489, 0.705] | -0.118 | 0.727 | 0.107 | 0.588 | 0.637 | 0.526 | 0.550 | 16 | 6.78 | ok (grid edge: beta, c, decay, lr_att) |
| memory_only | ours (baseline) | 0.5946 [0.457, 0.771] | -0.099 | 0.719 | 0.095 | 0.569 | 0.715 | 0.538 | 0.557 | — | 3.09 | ok (grid edge: beta, memory_capacity) |
| ref_gru | reference | 0.5995 [0.523, 0.710] | -0.094 | 0.699 | 0.128 | 0.621 | 0.650 | 0.562 | 0.565 | 666 | 9.13 | ok (grid edge: hidden, lr) |
| ref_bias | reference | 0.6928 [0.692, 0.693] | -0.000 | 0.523 | 0.036 | 0.694 | 0.692 | 0.692 | 0.693 | 1 | 0.15 | ok |
| random | chance | 0.6931 [0.693, 0.693] | -0.000 | 0.500 | 0.000 | 0.693 | 0.693 | 0.693 | 0.693 | — | 0.2 | ok |
| ref_pvl_delta | reference | 0.7059 [0.694, 0.719] | +0.013 | 0.519 | 0.075 | 0.689 | 0.700 | 0.723 | 0.712 | 4 | 2.08 | ok (grid edge: A, alpha, c, lam) |
| static_persona | ours (baseline) | 1.0480 [0.893, 1.217] | +0.355 | 0.404 | 0.382 | 1.076 | 1.038 | 1.053 | 1.025 | — | 0.23 | ok |
| hybrid_exemplar | ours | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: policy class 'hybrid_exemplar' not registered in this build (KeyError: |
| ref_delta_rule | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only |
| ref_instance_based | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT o |
| ref_population_static | reference | — | — | — | — | — | — | — | — | — | 0.0 | unavailable: ValueError: ref_population_static is defined for MCPL only |

| contrast | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental − ref_bayes_rule | -0.0074 | [-0.0118, -0.0015] | 0.014 | -1.08 | 5 |
| developmental − plain_rl | -0.0032 | [-0.0235, +0.0113] | 0.768 | -0.14 | 5 |
| ref_bayes_rule − ref_bias | -0.1195 | [-0.2104, +0.0007] | 0.056 | -0.86 | 5 |


## 6b. Dev pilot B (X-010) — Candidate A on the three pristine families (EXPLORATORY)

Run after the cognitive-architecture lane was merged (`hybrid_exemplar` registered), with the same
protocol as §6 (fit dev / select tune / report tune; same lapse grid; 16-configuration grid of the
candidate's own knobs, committed before the run): `reports/phase2/human/p2_human_dev_pilot_hybrid/`.
Wall 1615 s, CPU 1593 s, 0 LLM calls.

| Family (n tune) | `hybrid_exemplar` NLL | `ref_gru` | best per-participant reference | hybrid − GRU [95 % CI] | hybrid − `developmental` | hybrid − best per-participant reference |
|---|---|---|---|---|---|---|
| bandit2 (16) | 0.585 | 0.389 | `ref_pvl_delta` 0.534 | +0.196 [+0.135, +0.255] | +0.024 [+0.009, +0.038] | +0.052 [+0.000, +0.105] |
| garcia (68) | 0.625 | 0.447 | `ref_bayes_rule` 0.612 | +0.178 [+0.152, +0.202] | −0.016 [−0.023, −0.009] | +0.007 [−0.004, +0.019] vs PVL-Delta; behind the Bayesian rule learner (0.612) |
| described (41) | 0.641 | 0.372 | `ref_pvl_delta` 0.651 | +0.269 [+0.223, +0.317] | −0.014 [−0.018, −0.010] | −0.009 [−0.026, +0.009] (tie; hybrid is the best per-participant model here) |

Paired contrasts recomputed from the raw outcomes with `scripts/phase2_contrasts.py` (same pairing and
bootstrap as the runner; the runner's own `developmental − ref_gru` rows reproduce to the third decimal).
Candidate A beats `ref_exemplar` (−0.025 / −0.018 / −0.039) and `memory_only` (−0.190 / −0.026 / −0.134)
in every family (CIs exclude 0), i.e. it satisfies the "must beat memory_only" rule, but every one of its
four knobs is at a grid edge in every family (two-point grids), so these are lower bounds on what a
wider grid would give — for the candidate *and* for every reference, which received the same cap.

**Verdict for the freeze decision: Candidate A does not approach the GRU in any pristine family (gap
0.18–0.27 nats) and does not beat the best per-participant reference by ≥ 0.02 in any family.** The
stop rule applies; no confirmatory run was frozen (`EXPERIMENT_REGISTRY.md`, X-010).

## 7. Wins / losses / untested (per family, from §6 and §6b)

Rule: "wins" = paired 95 % CI against the best reference excludes 0 in our favour by ≥ 0.02 nats on the
tune partition (selection-optimistic for every model equally); "loses" = CI excludes 0 against us;
otherwise "tie". All contrasts are from `summary.json` (paired bootstrap over tune participants).

| Family (n tune) | Best model | `developmental` vs best reference | Verdict for our architecture | Also notable |
|---|---|---|---|---|
| mcpl (47) | `ref_gru` 1.248 | +0.445 [+0.355, +0.536] | **loses** | `memory_only` (1.426) and `ref_instance_based` (1.441) also beat it; `developmental` 1.693 |
| igt (103) | `ref_gru` 1.100 | +0.083 [+0.064, +0.102] | **loses** | second best (1.183); ahead of every cognitive model incl. PVL-Delta (1.297) |
| bandit2 (16) | `ref_gru` 0.389 | +0.172 [+0.115, +0.226] | **loses** | third after PVL-Delta (0.534); far ahead of plain RL (0.682) |
| srl3 (15) | `ref_gru` 0.741 | +0.113 [+0.084, +0.140] | **loses** | second best (0.854) |
| horizon (51) | `ref_gru` 0.459 | +0.169 [+0.134, +0.205] | **loses** | third after PVL-Delta (0.529); plain RL at chance (0.688) |
| casino (16) | `ref_gru` 0.556 | +0.067 [+0.006, +0.124] | **loses** (narrowly) | second best (0.623) |
| wpt (5) | `developmental` 0.566 | −0.007 [−0.012, −0.002] vs `ref_bayes_rule` | tie (below the 0.02 margin; n = 5) | all learning models within 0.04 of each other |
| garcia (68) | `ref_gru` 0.447 | +0.194 [+0.167, +0.219] | **loses** | behind Bayesian rule learner, PVL-Delta and plain RL |
| described (41) | `ref_gru` 0.372 | +0.283 [+0.237, +0.329] | **loses** | near-chance for every per-participant learner (0.65–0.68); only the population-fitted GRU is far from chance |

Reading, stated plainly:

1. **A small sequence model fitted to other participants predicts held-out people better than our
   per-participant online learner in 8 of 9 families.** The GRU has 1–2 k parameters fitted on dev
   participants; our policies tune ≤ 3 hyper-parameters and learn each participant from scratch.
   This is the same asymmetry that lets Centaur beat domain-specific cognitive models: population
   regularities (e.g. choice biases, response to described gambles) are learnable from other people.
   The `ref_gru − ref_bias` contrasts (−0.14 to −0.70 nats) measure how much of that is sequence
   structure beyond pooled frequencies.
2. **Against cognitive models, the developmental learner is competitive on bandit-style tasks**
   (second on IGT, srl3, casino; third on bandit2 and horizon behind PVL-Delta) and **weak on
   cue-combination tasks** (MCPL: behind instance-based and even our memory-only baseline; garcia).
   This reproduces the Phase I MCPL finding on fresh participants and extends it.
3. **The Phase I "IGT best of 8" result does not survive a stronger baseline.** With the GRU in the
   comparison the developmental learner is second on IGT (+0.083 nats behind).
4. **Sample efficiency (NLL by quartile, §6)**: per-participant learners improve from Q1 to Q4 in every
   bandit family; the GRU starts low already in Q1 because it carries population knowledge.
5. **Calibration**: ECE is lowest for the GRU in most families; `memory_only` is the worst-calibrated
   model everywhere (0.15–0.23).
6. **Candidate A (`hybrid_exemplar`, §6b)** loses to the GRU by 0.18–0.27 nats on all three pristine
   families; it is a marginal (≤ 0.016 nats) improvement on the Phase I learner on garcia/described,
   slightly worse on bandit2, and level with or behind PVL-Delta. **Untested**: `ref_instance_based` /
   `ref_delta_rule` are defined only for MCPL/IGT; `ref_bayes_rule` only where corrective feedback
   exists. No family has been evaluated on the test partition.

Implication for Gate D: as the baseline set now includes a population-fitted sequence model, the
candidate architectures must either (a) also use other participants (a developmental prior learned
across people), or (b) be compared on a preregistered sub-question where per-participant learning is
the scientific target (e.g. first-exposure behaviour, transfer after a rule change). Which of these is
chosen is a Research-Director decision and must be preregistered before any test-partition run.

## 8. Second-dataset search (licences verified from pages opened on 2026-11-04)

| Candidate | What | Licence (evidence) | Suitability |
|---|---|---|---|
| Psych-201 (`marcelbinz/Psych-201`) | 208,021 transcripts incl. Psych-101 flags, RTs, demographics, clinical questionnaires | Apache-2.0 (dataset card YAML `license: apache-2.0`) | Same source family as Psych-101 (not an independent population for the overlap); its `is_psych101: false` rows and questionnaire fields are relevant to the affective lane (H3). Not yet downloaded (540 MB). |
| OpenNeuro ds000005 "Mixed-gambles task" (Tom et al.) | accept/reject described gambles with gain/loss amounts; behavioural events in the BIDS `events.tsv` | PDDL 1.0 (dataset_description.json) | risky described choice; adapter needed; independent population. |
| OpenNeuro ds002041 | subjective-value decisions with behavioural events | CC0 (dataset_description.json) | candidate; task details not yet inspected. |
| OpenNeuro ds004636 | "Cognitive tasks … evaluating the construct …" battery with behavioural data | CC0 (dataset_description.json) | candidate; which tasks are included must be checked. |
| Zenodo search | API ignored our query terms (returned unrelated most-recent records) | — | not usable as a search method here; manual DOI lookup required. |
| DfE-DB | — | no licence; data not open (Phase I) | excluded |

None of these has been adapted yet; Gate D's "≥ 2 datasets" clause is therefore still unmet.
