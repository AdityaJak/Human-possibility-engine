# `composition` family on DEV seeds (exploratory)

Seeds 1000-1009 (n=10), 20 agents per seed, 300 episodes per family, bandit feedback, label noise 0.05. Produced by `scripts/composition_dev.py`; code `70437a063f67406b5fbbb341b18b8c65cefff80c+dirty`; CPU 15.4 min over 80 runs. **Dev seeds only — not a confirmatory result.** Chance is 0.5 on every family (eval sets are exactly balanced).

Measured probe ceilings for `composition` (from `composition_ceiling.json`, eval-set accuracy, mean over seeds):

- shared: lstsq_raw 0.625 [0.575, 0.682], logistic_raw 0.672 [0.612, 0.745], logistic_quad 0.655 [0.573, 0.735], oracle_prims 1.000 [1.000, 1.000]
- control: lstsq_raw 0.505 [0.420, 0.587], logistic_raw 0.527 [0.437, 0.608], logistic_quad 0.518 [0.443, 0.598], oracle_prims 0.583 [0.523, 0.655]

## World `shared` (composition uses the taught primitives)

| arm | primitive_a | primitive_b | composition | composition_pair |
|---|---|---|---|---|
| plain_rl | +0.912 [+0.893, +0.929] (t0 0.50) | +0.526 [+0.467, +0.578] (t0 0.50) | +0.701 [+0.627, +0.775] (t0 0.50) | +0.739 [+0.646, +0.837] (t0 0.50) |
| memory_only | +0.892 [+0.872, +0.909] (t0 0.50) | +0.595 [+0.527, +0.668] (t0 0.50) | +0.684 [+0.605, +0.764] (t0 0.50) | +0.741 [+0.645, +0.837] (t0 0.50) |
| developmental | +0.965 [+0.955, +0.973] (t0 0.50) | +0.516 [+0.479, +0.548] (t0 0.50) | +0.704 [+0.631, +0.784] (t0 0.50) | +0.733 [+0.640, +0.831] (t0 0.50) |
| hybrid_exemplar | +0.976 [+0.967, +0.984] (t0 0.50) | +0.628 [+0.583, +0.674] (t0 0.50) | +0.709 [+0.632, +0.790] (t0 0.50) | +0.734 [+0.640, +0.834] (t0 0.50) |

## World `control` (specialist control: composed primitives independent of the taught ones)

| arm | primitive_a | primitive_b | composition | composition_pair |
|---|---|---|---|---|
| plain_rl | +0.910 [+0.891, +0.928] (t0 0.50) | +0.535 [+0.485, +0.579] (t0 0.50) | +0.643 [+0.526, +0.760] (t0 0.50) | +0.681 [+0.537, +0.812] (t0 0.50) |
| memory_only | +0.879 [+0.854, +0.901] (t0 0.50) | +0.577 [+0.520, +0.640] (t0 0.50) | +0.656 [+0.565, +0.744] (t0 0.50) | +0.682 [+0.550, +0.797] (t0 0.50) |
| developmental | +0.948 [+0.931, +0.964] (t0 0.50) | +0.526 [+0.489, +0.562] (t0 0.50) | +0.602 [+0.513, +0.688] (t0 0.50) | +0.656 [+0.514, +0.789] (t0 0.50) |
| hybrid_exemplar | +0.958 [+0.942, +0.971] (t0 0.50) | +0.611 [+0.566, +0.659] (t0 0.50) | +0.627 [+0.545, +0.705] (t0 0.50) | +0.655 [+0.519, +0.781] (t0 0.50) |

## Shared − control (paired over seeds)

| arm | composition | composition_pair |
|---|---|---|
| plain_rl | +0.058 [-0.029, +0.145] dz=0.39 | +0.058 [-0.046, +0.158] dz=0.33 |
| memory_only | +0.028 [-0.049, +0.101] dz=0.22 | +0.059 [-0.052, +0.171] dz=0.32 |
| developmental | +0.102 [+0.016, +0.175] dz=0.77 | +0.077 [-0.036, +0.184] dz=0.41 |
| hybrid_exemplar | +0.082 [+0.001, +0.157] dz=0.60 | +0.079 [-0.030, +0.180] dz=0.44 |

## Informativeness against the measured linear ceiling

- shared: linear ceiling 0.672; arms whose composition CI lies entirely above it: **none** (family uninformative for these learners at this budget)
- control: linear ceiling 0.527; arms whose composition CI lies entirely above it: memory_only, hybrid_exemplar
