# Human DEV-partition comparison — IGT (EXPLORATORY, Phase II)

tune=103 dev=204; test participants never loaded. Lower NLL is better.

| model | dev NLL [95% CI] | top-1 | selected params | lapse | configs | CPU s |
|---|---|---|---|---|---|---|
| hybrid_exemplar | 1.1514 [1.1156, 1.1883] | 0.472 | `{'exemplar_beta': 3.0, 'gate_lr': 0.2, 'recency_decay': 0.9}` | 0.01 | 54 | 298 |
| developmental | 1.1531 [1.1167, 1.1893] | 0.471 | `{}` | 0.0 | 3 | 12 |
| hybrid_fixed_gate | 1.1534 [1.1147, 1.1889] | 0.470 | `{'exemplar_beta': 3.0, 'recency_decay': 0.9}` | 0.02 | 27 | 148 |
| hybrid_exemplar_only | 1.1648 [1.1282, 1.2019] | 0.467 | `{'exemplar_beta': 3.0, 'recency_decay': 0.9}` | 0.1 | 27 | 119 |
| memory_only | 1.2772 [1.2470, 1.3073] | 0.437 | `{}` | 0.3 | 3 | 17 |
| ref_instance_based | 1.2781 [1.2586, 1.2959] | 0.396 | `{'beta': 6.0, 'decay': 0.5}` | 0.2 | 48 | 94 |
| ref_delta_rule | 1.2969 [1.2815, 1.3127] | 0.390 | `{'alpha': 0.4, 'beta': 6.0}` | 0.2 | 48 | 86 |
| plain_rl | 1.3293 [1.3188, 1.3388] | 0.403 | `{}` | 0.0 | 3 | 14 |

| contrast (treatment − control, NLL; negative favours treatment) | mean | 95% CI | share better | n |
|---|---|---|---|---|
| hybrid_exemplar − ref_instance_based | -0.1266 | [-0.1537, -0.1004] | 0.68 | 204 |
| hybrid_exemplar − developmental | -0.0016 | [-0.0045, +0.0012] | 0.46 | 204 |
| hybrid_exemplar − memory_only | -0.1258 | [-0.1467, -0.1034] | 0.79 | 204 |
| hybrid_exemplar − plain_rl | -0.1779 | [-0.2106, -0.1462] | 0.76 | 204 |
| hybrid_fixed_gate − ref_instance_based | -0.1247 | [-0.1528, -0.0986] | 0.68 | 204 |
| hybrid_fixed_gate − developmental | +0.0003 | [-0.0022, +0.0030] | 0.46 | 204 |
| hybrid_fixed_gate − memory_only | -0.1238 | [-0.1456, -0.1033] | 0.79 | 204 |
| hybrid_fixed_gate − plain_rl | -0.1759 | [-0.2098, -0.1443] | 0.76 | 204 |
| hybrid_exemplar_only − ref_instance_based | -0.1133 | [-0.1399, -0.0875] | 0.66 | 204 |
| hybrid_exemplar_only − developmental | +0.0117 | [+0.0071, +0.0161] | 0.34 | 204 |
| hybrid_exemplar_only − memory_only | -0.1124 | [-0.1339, -0.0917] | 0.76 | 204 |
| hybrid_exemplar_only − plain_rl | -0.1645 | [-0.1969, -0.1328] | 0.70 | 204 |

Terciles of participant predictability (ref_instance_based top-1 on dev): mean NLL per model

| tercile | n | hybrid_exemplar | hybrid_fixed_gate | hybrid_exemplar_only | developmental | plain_rl | memory_only | ref_instance_based | ref_delta_rule |
|---|---|---|---|---|---|---|---|---|---|
| least_predictable | 68 | 1.235 | 1.237 | 1.249 | 1.232 | 1.372 | 1.358 | 1.365 | 1.390 |
| middle | 68 | 1.228 | 1.228 | 1.234 | 1.229 | 1.342 | 1.340 | 1.305 | 1.317 |
| most_predictable | 68 | 0.992 | 0.995 | 1.012 | 0.999 | 1.274 | 1.133 | 1.164 | 1.183 |
