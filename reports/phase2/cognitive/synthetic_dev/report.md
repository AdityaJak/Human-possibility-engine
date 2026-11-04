# Synthetic DEV experiment — Candidate A on v2 families (EXPLORATORY, Phase II)

seeds [1000, 1001, 1002, 1003, 1004] (dev), 6 agents/arm, 300 episodes/family, CPU 248 s. Held-out accuracy after the full budget (mean over seeds ± sd over seeds).

## shared structure

| arm | conjunction | context_switch | nonstationary | delayed | ordinal_cue |
|---|---|---|---|---|---|
| hybrid_exemplar | 0.616 ± 0.036 | 0.541 ± 0.072 | 0.272 ± 0.106 | 0.919 ± 0.036 | 0.197 ± 0.210 |
| hybrid_fixed_gate | 0.562 ± 0.068 | 0.504 ± 0.037 | 0.286 ± 0.106 | 0.917 ± 0.033 | 0.218 ± 0.233 |
| hybrid_exemplar_only | 0.616 ± 0.045 | 0.573 ± 0.073 | 0.307 ± 0.089 | 0.824 ± 0.052 | 0.319 ± 0.290 |
| developmental | 0.489 ± 0.032 | 0.417 ± 0.026 | 0.293 ± 0.123 | 0.899 ± 0.035 | 0.139 ± 0.211 |
| plain_rl | 0.494 ± 0.055 | 0.469 ± 0.051 | 0.280 ± 0.092 | 0.897 ± 0.054 | 0.154 ± 0.193 |
| memory_only | 0.643 ± 0.052 | 0.547 ± 0.076 | 0.301 ± 0.083 | 0.836 ± 0.069 | 0.270 ± 0.175 |

| family | hybrid − control | mean | 95% CI (paired by seed) |
|---|---|---|---|
| conjunction | hybrid_fixed_gate | +0.053 | [+0.020, +0.079] |
| conjunction | hybrid_exemplar_only | -0.001 | [-0.040, +0.033] |
| conjunction | developmental | +0.127 | [+0.108, +0.142] |
| conjunction | plain_rl | +0.121 | [+0.073, +0.167] |
| conjunction | memory_only | -0.027 | [-0.039, -0.015] |
| context_switch | hybrid_fixed_gate | +0.037 | [+0.007, +0.069] |
| context_switch | hybrid_exemplar_only | -0.032 | [-0.053, -0.011] |
| context_switch | developmental | +0.124 | [+0.068, +0.181] |
| context_switch | plain_rl | +0.072 | [+0.003, +0.141] |
| context_switch | memory_only | -0.006 | [-0.029, +0.025] |
| nonstationary | hybrid_fixed_gate | -0.014 | [-0.026, +0.002] |
| nonstationary | hybrid_exemplar_only | -0.034 | [-0.058, -0.003] |
| nonstationary | developmental | -0.021 | [-0.057, +0.016] |
| nonstationary | plain_rl | -0.008 | [-0.037, +0.019] |
| nonstationary | memory_only | -0.029 | [-0.066, +0.007] |
| delayed | hybrid_fixed_gate | +0.002 | [-0.003, +0.008] |
| delayed | hybrid_exemplar_only | +0.094 | [+0.069, +0.119] |
| delayed | developmental | +0.019 | [+0.001, +0.038] |
| delayed | plain_rl | +0.022 | [+0.005, +0.036] |
| delayed | memory_only | +0.083 | [+0.041, +0.140] |
| ordinal_cue | hybrid_fixed_gate | -0.022 | [-0.064, +0.021] |
| ordinal_cue | hybrid_exemplar_only | -0.122 | [-0.183, -0.064] |
| ordinal_cue | developmental | +0.058 | [+0.030, +0.105] |
| ordinal_cue | plain_rl | +0.042 | [+0.023, +0.066] |
| ordinal_cue | memory_only | -0.073 | [-0.156, +0.014] |

## specialist structure

| arm | conjunction | context_switch | nonstationary | delayed | ordinal_cue |
|---|---|---|---|---|---|
| hybrid_exemplar | 0.616 ± 0.036 | 0.503 ± 0.055 | 0.238 ± 0.129 | 0.939 ± 0.012 | 0.256 ± 0.194 |
| hybrid_fixed_gate | 0.562 ± 0.068 | 0.471 ± 0.052 | 0.233 ± 0.107 | 0.924 ± 0.020 | 0.292 ± 0.287 |
| hybrid_exemplar_only | 0.616 ± 0.045 | 0.561 ± 0.054 | 0.300 ± 0.089 | 0.854 ± 0.036 | 0.319 ± 0.290 |
| developmental | 0.489 ± 0.032 | 0.403 ± 0.041 | 0.221 ± 0.061 | 0.921 ± 0.032 | 0.204 ± 0.167 |
| plain_rl | 0.494 ± 0.055 | 0.444 ± 0.066 | 0.234 ± 0.084 | 0.907 ± 0.039 | 0.154 ± 0.193 |
| memory_only | 0.643 ± 0.052 | 0.508 ± 0.076 | 0.264 ± 0.069 | 0.850 ± 0.030 | 0.270 ± 0.175 |

| family | hybrid − control | mean | 95% CI (paired by seed) |
|---|---|---|---|
| conjunction | hybrid_fixed_gate | +0.053 | [+0.024, +0.079] |
| conjunction | hybrid_exemplar_only | -0.001 | [-0.039, +0.033] |
| conjunction | developmental | +0.127 | [+0.109, +0.142] |
| conjunction | plain_rl | +0.121 | [+0.075, +0.167] |
| conjunction | memory_only | -0.027 | [-0.039, -0.015] |
| context_switch | hybrid_fixed_gate | +0.032 | [+0.007, +0.052] |
| context_switch | hybrid_exemplar_only | -0.058 | [-0.089, -0.028] |
| context_switch | developmental | +0.099 | [+0.054, +0.146] |
| context_switch | plain_rl | +0.058 | [+0.026, +0.090] |
| context_switch | memory_only | -0.006 | [-0.023, +0.012] |
| nonstationary | hybrid_fixed_gate | +0.004 | [-0.034, +0.046] |
| nonstationary | hybrid_exemplar_only | -0.062 | [-0.098, -0.016] |
| nonstationary | developmental | +0.017 | [-0.051, +0.079] |
| nonstationary | plain_rl | +0.003 | [-0.043, +0.050] |
| nonstationary | memory_only | -0.027 | [-0.085, +0.038] |
| delayed | hybrid_fixed_gate | +0.014 | [-0.002, +0.032] |
| delayed | hybrid_exemplar_only | +0.085 | [+0.059, +0.126] |
| delayed | developmental | +0.018 | [-0.007, +0.047] |
| delayed | plain_rl | +0.032 | [+0.002, +0.067] |
| delayed | memory_only | +0.089 | [+0.063, +0.122] |
| ordinal_cue | hybrid_fixed_gate | -0.036 | [-0.127, +0.026] |
| ordinal_cue | hybrid_exemplar_only | -0.063 | [-0.178, +0.046] |
| ordinal_cue | developmental | +0.052 | [+0.008, +0.096] |
| ordinal_cue | plain_rl | +0.102 | [+0.053, +0.160] |
| ordinal_cue | memory_only | -0.014 | [-0.089, +0.071] |
