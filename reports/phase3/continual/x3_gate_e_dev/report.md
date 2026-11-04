# x3_gate_e_dev — Gate E synthetic switching suite — F4 (meta-learned GRU, frozen at test) vs online fine-tuning, EWC and replay

> **EXPLORATORY (development seeds of the synthetic Gate E suite).** Synthetic data; no human data; a synthetic result does not establish human validity. Confirmatory seeds 3000–3049 were not generated.

* Lane research/continual-transfer; definition hash `aab8dc4c9e89799a`; seed 20261108; threads 1
* Seeds: meta-train [0, 120], selection [120, 160], report [160, 200] (half-open ranges); 5 rollouts per report seed
* Hypothesis: (Exploratory.) A recurrent learner meta-trained on development switching sequences shows larger savings on return to a previously seen context than online fine-tuning, EWC and experience replay of a small value network, without adapting more than 10 % slower after a switch.
* Metrics: expected reward of the chosen option. adaptation = mean over switches of (10 trials after the switch) / oracle (0.8); savings = first 10 trials of a return − first 10 trials of that context's first block; forgetting = first 10 of a return − last 10 of that context's previous block (negative = loss).

| learner | adaptation [95% CI] | savings [95% CI] | forgetting [95% CI] | reward / oracle [95% CI] | params | configs | select CPU s | report CPU s |
|---|---|---|---|---|---|---|---|---|
| oracle (privileged) | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | — | — | — | 0.03 |
| f4 | 0.637 [0.621, 0.653] | -0.024 [-0.050, 0.002] | -0.236 [-0.251, -0.220] | 0.829 [0.817, 0.839] | 31589 | 2 | 1251.0 | 0.19 |
| bayes_filter | 0.672 [0.653, 0.692] | 0.047 [0.024, 0.069] | -0.151 [-0.172, -0.131] | 0.814 [0.799, 0.829] | — | 4 | 30.7 | 25.18 |
| delta_rule | 0.595 [0.582, 0.607] | -0.041 [-0.065, -0.018] | -0.211 [-0.227, -0.196] | 0.757 [0.749, 0.766] | — | 12 | 4.5 | 0.86 |
| ft | 0.636 [0.623, 0.647] | -0.003 [-0.027, 0.020] | -0.142 [-0.155, -0.130] | 0.740 [0.731, 0.748] | 2724 | 18 | 44.5 | 5.82 |
| replay | 0.616 [0.602, 0.629] | 0.033 [0.013, 0.052] | -0.106 [-0.119, -0.093] | 0.708 [0.700, 0.716] | 2724 | 24 | 119.3 | 15.31 |
| ewc | 0.605 [0.594, 0.615] | -0.003 [-0.018, 0.015] | -0.116 [-0.125, -0.107] | 0.687 [0.680, 0.695] | 2724 | 24 | 74.5 | 7.39 |
| ewc_oracle_boundaries (privileged) | 0.603 [0.591, 0.615] | -0.002 [-0.021, 0.015] | -0.111 [-0.120, -0.102] | 0.685 [0.677, 0.694] | 2724 | 24 | 74.5 | 7.33 |
| stationary_ts | 0.550 [0.534, 0.567] | 0.048 [0.033, 0.063] | -0.085 [-0.098, -0.073] | 0.622 [0.611, 0.634] | — | — | — | 0.76 |
| random | 0.537 [0.527, 0.546] | 0.000 [-0.008, 0.008] | 0.002 [-0.003, 0.008] | 0.535 [0.526, 0.544] | — | — | — | 0.2 |

## Paired contrasts over report seeds (first − second; positive favours first)

| first | second | metric | estimate | 95% CI | p (max boot, t; H1 first > second) | n |
|---|---|---|---|---|---|---|
| ft | ewc | adaptation | +0.0306 | [+0.0209, +0.0398] | 0.0005 | 40 |
| ft | ewc | savings | -0.0005 | [-0.0199, +0.0162] | 0.5191 | 40 |
| ft | ewc | forgetting | -0.0256 | [-0.0370, -0.0146] | 1.0000 | 40 |
| ft | ewc | reward_fraction | +0.0525 | [+0.0449, +0.0601] | 0.0005 | 40 |
| ft | replay | adaptation | +0.0200 | [+0.0094, +0.0303] | 0.0005 | 40 |
| ft | replay | savings | -0.0364 | [-0.0560, -0.0190] | 1.0000 | 40 |
| ft | replay | forgetting | -0.0361 | [-0.0468, -0.0255] | 1.0000 | 40 |
| ft | replay | reward_fraction | +0.0318 | [+0.0255, +0.0383] | 0.0005 | 40 |
| ft | bayes_filter | adaptation | -0.0367 | [-0.0544, -0.0193] | 1.0000 | 40 |
| ft | bayes_filter | savings | -0.0501 | [-0.0787, -0.0191] | 0.9990 | 40 |
| ft | bayes_filter | forgetting | +0.0096 | [-0.0098, +0.0283] | 0.1740 | 40 |
| ft | bayes_filter | reward_fraction | -0.0742 | [-0.0890, -0.0602] | 1.0000 | 40 |
| ft | delta_rule | adaptation | +0.0408 | [+0.0288, +0.0522] | 0.0005 | 40 |
| ft | delta_rule | savings | +0.0380 | [+0.0139, +0.0630] | 0.0018 | 40 |
| ft | delta_rule | forgetting | +0.0696 | [+0.0543, +0.0847] | 0.0005 | 40 |
| ft | delta_rule | reward_fraction | -0.0172 | [-0.0253, -0.0093] | 1.0000 | 40 |
| ewc | replay | adaptation | -0.0107 | [-0.0208, -0.0005] | 0.9800 | 40 |
| ewc | replay | savings | -0.0360 | [-0.0525, -0.0193] | 1.0000 | 40 |
| ewc | replay | forgetting | -0.0105 | [-0.0219, +0.0015] | 0.9530 | 40 |
| ewc | replay | reward_fraction | -0.0207 | [-0.0275, -0.0137] | 1.0000 | 40 |
| ewc | bayes_filter | adaptation | -0.0673 | [-0.0859, -0.0488] | 1.0000 | 40 |
| ewc | bayes_filter | savings | -0.0496 | [-0.0751, -0.0244] | 1.0000 | 40 |
| ewc | bayes_filter | forgetting | +0.0352 | [+0.0157, +0.0558] | 0.0011 | 40 |
| ewc | bayes_filter | reward_fraction | -0.1267 | [-0.1416, -0.1124] | 1.0000 | 40 |
| ewc | delta_rule | adaptation | +0.0102 | [-0.0004, +0.0205] | 0.0336 | 40 |
| ewc | delta_rule | savings | +0.0385 | [+0.0185, +0.0604] | 0.0005 | 40 |
| ewc | delta_rule | forgetting | +0.0952 | [+0.0809, +0.1102] | 0.0005 | 40 |
| ewc | delta_rule | reward_fraction | -0.0697 | [-0.0788, -0.0611] | 1.0000 | 40 |
| replay | bayes_filter | adaptation | -0.0567 | [-0.0717, -0.0410] | 1.0000 | 40 |
| replay | bayes_filter | savings | -0.0136 | [-0.0375, +0.0096] | 0.8665 | 40 |
| replay | bayes_filter | forgetting | +0.0457 | [+0.0269, +0.0655] | 0.0005 | 40 |
| replay | bayes_filter | reward_fraction | -0.1061 | [-0.1188, -0.0929] | 1.0000 | 40 |
| replay | delta_rule | adaptation | +0.0208 | [+0.0112, +0.0304] | 0.0005 | 40 |
| replay | delta_rule | savings | +0.0744 | [+0.0559, +0.0950] | 0.0005 | 40 |
| replay | delta_rule | forgetting | +0.1057 | [+0.0927, +0.1189] | 0.0005 | 40 |
| replay | delta_rule | reward_fraction | -0.0490 | [-0.0584, -0.0403] | 1.0000 | 40 |
| f4 | ft | adaptation | +0.0018 | [-0.0148, +0.0194] | 0.4355 | 40 |
| f4 | ft | savings | -0.0210 | [-0.0490, +0.0073] | 0.9330 | 40 |
| f4 | ft | forgetting | -0.0941 | [-0.1144, -0.0724] | 1.0000 | 40 |
| f4 | ft | reward_fraction | +0.0891 | [+0.0781, +0.1008] | 0.0005 | 40 |
| f4 | ewc | adaptation | +0.0324 | [+0.0143, +0.0505] | 0.0009 | 40 |
| f4 | ewc | savings | -0.0214 | [-0.0468, +0.0030] | 0.9595 | 40 |
| f4 | ewc | forgetting | -0.1197 | [-0.1391, -0.1002] | 1.0000 | 40 |
| f4 | ewc | reward_fraction | +0.1416 | [+0.1298, +0.1530] | 0.0005 | 40 |
| f4 | replay | adaptation | +0.0218 | [+0.0055, +0.0383] | 0.0070 | 40 |
| f4 | replay | savings | -0.0574 | [-0.0817, -0.0333] | 1.0000 | 40 |
| f4 | replay | forgetting | -0.1302 | [-0.1466, -0.1136] | 1.0000 | 40 |
| f4 | replay | reward_fraction | +0.1210 | [+0.1104, +0.1318] | 0.0005 | 40 |
| f4 | bayes_filter | adaptation | -0.0349 | [-0.0524, -0.0150] | 1.0000 | 40 |
| f4 | bayes_filter | savings | -0.0710 | [-0.0961, -0.0453] | 1.0000 | 40 |
| f4 | bayes_filter | forgetting | -0.0845 | [-0.1069, -0.0634] | 1.0000 | 40 |
| f4 | bayes_filter | reward_fraction | +0.0149 | [+0.0025, +0.0277] | 0.0108 | 40 |
| f4 | delta_rule | adaptation | +0.0426 | [+0.0282, +0.0571] | 0.0005 | 40 |
| f4 | delta_rule | savings | +0.0170 | [-0.0041, +0.0388] | 0.0659 | 40 |
| f4 | delta_rule | forgetting | -0.0245 | [-0.0421, -0.0061] | 0.9945 | 40 |
| f4 | delta_rule | reward_fraction | +0.0719 | [+0.0610, +0.0830] | 0.0005 | 40 |
| bayes_filter | ft | adaptation | +0.0367 | [+0.0188, +0.0547] | 0.0005 | 40 |
| bayes_filter | ft | savings | +0.0501 | [+0.0190, +0.0797] | 0.0013 | 40 |
| bayes_filter | ft | forgetting | -0.0096 | [-0.0288, +0.0096] | 0.8330 | 40 |
| bayes_filter | ft | reward_fraction | +0.0742 | [+0.0601, +0.0875] | 0.0005 | 40 |
| bayes_filter | ewc | adaptation | +0.0673 | [+0.0485, +0.0853] | 0.0005 | 40 |
| bayes_filter | ewc | savings | +0.0496 | [+0.0235, +0.0753] | 0.0005 | 40 |
| bayes_filter | ewc | forgetting | -0.0352 | [-0.0567, -0.0152] | 0.9990 | 40 |
| bayes_filter | ewc | reward_fraction | +0.1267 | [+0.1118, +0.1424] | 0.0005 | 40 |
| bayes_filter | replay | adaptation | +0.0567 | [+0.0412, +0.0721] | 0.0005 | 40 |
| bayes_filter | replay | savings | +0.0136 | [-0.0098, +0.0385] | 0.1370 | 40 |
| bayes_filter | replay | forgetting | -0.0457 | [-0.0661, -0.0266] | 1.0000 | 40 |
| bayes_filter | replay | reward_fraction | +0.1061 | [+0.0935, +0.1188] | 0.0005 | 40 |
| bayes_filter | delta_rule | adaptation | +0.0775 | [+0.0586, +0.0963] | 0.0005 | 40 |
| bayes_filter | delta_rule | savings | +0.0881 | [+0.0623, +0.1133] | 0.0005 | 40 |
| bayes_filter | delta_rule | forgetting | +0.0600 | [+0.0364, +0.0814] | 0.0005 | 40 |
| bayes_filter | delta_rule | reward_fraction | +0.0570 | [+0.0430, +0.0709] | 0.0005 | 40 |

## Development analogue of the Gate E rule (F4 vs ft, ewc, replay)

| comparator | savings difference | 95% CI | Holm p | savings condition |
|---|---|---|---|---|
| ft | -0.0210 | [-0.0485, +0.0064] | 1.0000 | False |
| ewc | -0.0214 | [-0.0466, +0.0030] | 1.0000 | False |
| replay | -0.0574 | [-0.0833, -0.0322] | 1.0000 | False |

Adaptation: F4 0.637 vs best comparator `ft` 0.636; ratio 1.003 (needs ≥ 0.9): True. **Development analogue passes: False** (exploratory; cannot pass Gate E).

Pretrained initialisation of the comparators: {"n_transitions": 36000, "mse_by_epoch": [0.22696, 0.22267, 0.22204, 0.22115, 0.21928, 0.22093, 0.21985, 0.22064, 0.219, 0.2196, 0.21839, 0.21915, 0.2189, 0.22118, 0.21902, 0.21799, 0.21843, 0.21966, 0.2183, 0.21815], "cpu_s": 7.25}

