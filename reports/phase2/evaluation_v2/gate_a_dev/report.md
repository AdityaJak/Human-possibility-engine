# Gate A controls on DEV seeds (exploratory)

Seeds 1000-1009 (n=10), 20 agents per seed, v1 families compare, signal, sort; learning-lab defaults, transfer/retention skipped. Produced by `scripts/gate_a_dev.py`; code `70437a063f67406b5fbbb341b18b8c65cefff80c`; CPU 11.0 min over 110 runs. **Dev seeds only — not a confirmatory result.**

## Held-out accuracy (mean over families; mean over seeds, 95% bootstrap CI)

| variant / arm | initial (tick 0) | final | updates / seed |
|---|---|---|---|
| main/developmental | 0.444 | +0.832 [+0.816, +0.849] | 26980 |
| main/developmental_frozen | 0.444 | +0.444 [+0.444, +0.444] | 0 |
| main/hybrid_exemplar | 0.444 | +0.830 [+0.815, +0.849] | 18000 |
| main/hybrid_exemplar_frozen | 0.444 | +0.444 [+0.444, +0.444] | 0 |
| main/plain_rl | 0.444 | +0.814 [+0.787, +0.840] | 18000 |
| main/plain_rl_frozen | 0.444 | +0.444 [+0.444, +0.444] | 18000 |
| main/age_script_only | 0.440 | +0.440 [+0.429, +0.453] | 0 |
| main/static_persona | 0.440 | +0.440 [+0.429, +0.453] | 0 |
| permuted_feedback/developmental | 0.444 | +0.437 [+0.431, +0.443] | 26980 |
| permuted_feedback/hybrid_exemplar | 0.444 | +0.433 [+0.424, +0.442] | 18000 |
| permuted_feedback/plain_rl | 0.444 | +0.446 [+0.436, +0.454] | 18000 |

## Gains (paired over seeds; estimate [95% CI], Cohen dz, sign-flip p)

| learner | frozen-copy gain (≥0.10 expected) | permuted-feedback gain (≤0.02 expected) | learned − permuted | age-script-only gain (≥0.10 required) | static-persona gain |
|---|---|---|---|---|---|
| developmental | +0.388 [+0.372, +0.404] dz=14.04 p=0.00195 | -0.007 [-0.014, -0.001] dz=-0.67 p=0.0723 | +0.395 [+0.380, +0.411] dz=14.53 p=0.00195 | +0.392 [+0.374, +0.412] dz=12.28 p=0.00195 | +0.392 [+0.374, +0.412] dz=12.28 p=0.00195 |
| hybrid_exemplar | +0.386 [+0.371, +0.404] dz=13.76 p=0.00195 | -0.011 [-0.020, -0.003] dz=-0.73 p=0.043 | +0.397 [+0.381, +0.414] dz=13.96 p=0.00195 | +0.390 [+0.371, +0.409] dz=11.90 p=0.00195 | +0.390 [+0.371, +0.409] dz=11.90 p=0.00195 |
| plain_rl | +0.369 [+0.343, +0.396] dz=8.17 p=0.00195 | +0.001 [-0.008, +0.009] dz=0.10 p=0.775 | +0.368 [+0.344, +0.392] dz=8.77 p=0.00195 | +0.373 [+0.341, +0.404] dz=7.00 p=0.00195 | +0.373 [+0.342, +0.405] dz=7.00 p=0.00195 |

## Dev pattern against the Gate A thresholds (exploratory, dev seeds)

| learner | frozen-copy gain ≥ 0.10, CI > 0 | permuted gain CI within ±0.02 | permuted point ≤ 0.02 | age-script gain ≥ 0.10, CI > 0 |
|---|---|---|---|---|
| developmental | True | True | True | True |
| hybrid_exemplar | True | False | True | True |
| plain_rl | True | True | True | True |

## Per-family gains

| learner | family | frozen-copy gain | permuted gain | age-script gain |
|---|---|---|---|---|
| developmental | compare | +0.393 [+0.372, +0.415] | +0.009 [-0.003, +0.021] | +0.394 [+0.371, +0.420] |
| developmental | signal | +0.372 [+0.337, +0.405] | -0.012 [-0.033, +0.009] | +0.364 [+0.325, +0.402] |
| developmental | sort | +0.398 [+0.370, +0.430] | -0.019 [-0.034, -0.005] | +0.418 [+0.391, +0.449] |
| hybrid_exemplar | compare | +0.384 [+0.363, +0.405] | -0.013 [-0.027, +0.001] | +0.385 [+0.361, +0.413] |
| hybrid_exemplar | signal | +0.360 [+0.327, +0.389] | -0.010 [-0.030, +0.009] | +0.352 [+0.318, +0.383] |
| hybrid_exemplar | sort | +0.412 [+0.379, +0.444] | -0.011 [-0.030, +0.010] | +0.432 [+0.399, +0.464] |
| plain_rl | compare | +0.383 [+0.358, +0.405] | +0.004 [-0.006, +0.018] | +0.383 [+0.352, +0.413] |
| plain_rl | signal | +0.360 [+0.330, +0.387] | +0.011 [-0.006, +0.028] | +0.352 [+0.311, +0.385] |
| plain_rl | sort | +0.366 [+0.320, +0.416] | -0.011 [-0.031, +0.010] | +0.385 [+0.337, +0.435] |

Freeze check (held-out action probabilities changed after 60 practice episodes with full feedback): developmental=yes, developmental_frozen=no, hybrid_exemplar=yes, hybrid_exemplar_frozen=no, plain_rl=yes, plain_rl_frozen=no, age_script_only=no, static_persona=no
