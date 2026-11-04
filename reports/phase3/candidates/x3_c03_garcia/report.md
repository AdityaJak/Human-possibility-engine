# x3_c03_garcia — Design-team candidates (F3 episodic arbiter, F2 world model + policy, F1 individual latent) vs same-size pop_gru, garcia

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: research/adaptation; preregistration hash `07201d9d9d2f6a6e`; fit dev / select tune / report tune; seed 20261108; threads 2
* Hypothesis: (Exploratory.) Explicit individual adaptation (F3 episodic memory + learned gate, F2 individual policy posterior, F1 online latent) improves tune NLL over a same-size pop_gru twin and over the same population network with its recurrent state running.

## garcia (counted); best comparator on tune: `pop_gru`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f3_arbiter | candidate | 0.4348 [0.4032, 0.4680] | 68 | 0.775 | 0.048 | 0.535 / 0.493 / 0.392 / 0.319 | 2290 | 0.0 | hidden, dropout, bandwidth | 79.0 | ok |
| f3_gruepi | reference | 0.4367 [0.4026, 0.4724] | 68 | 0.778 | 0.064 | 0.535 / 0.499 / 0.388 / 0.324 | 4748 | 0.02 | hidden, dropout | 97.0 | ok |
| f3_fixedshare | reference | 0.4419 [0.4104, 0.4745] | 68 | 0.771 | 0.052 | 0.540 / 0.493 / 0.401 / 0.333 | 2284 | 0.0 | hidden, dropout | 0.1 | ok |
| f1_latent | candidate | 0.4426 [0.4065, 0.4774] | 68 | 0.769 | 0.059 | 0.544 / 0.502 / 0.399 / 0.325 | 2564 | 0.01 | hidden, dropout | 45.7 | ok |
| f1_bolt | reference | 0.4426 [0.4084, 0.4790] | 68 | 0.769 | 0.060 | 0.544 / 0.502 / 0.399 / 0.325 | 2564 | 0.01 | hidden, dropout | 0.0 | ok |
| pop_gru | comparator | 0.4438 [0.4103, 0.4788] | 68 | 0.779 | 0.064 | 0.545 / 0.509 / 0.400 / 0.321 | 8004 | 0.02 | hidden, dropout | 89.6 | ok |
| f3_fixedmix | reference | 0.4453 [0.4095, 0.4806] | 68 | 0.768 | 0.059 | 0.544 / 0.504 / 0.402 / 0.332 | 2285 | 0.01 | hidden, dropout | 9.0 | ok |
| f1_bolt[frozen] | reference | 0.4455 [0.4108, 0.4810] | 68 | 0.768 | 0.059 | 0.544 / 0.504 / 0.402 / 0.331 | 2564 | 0.01 | hidden, dropout | 0.0 | ok |
| f1_latent[frozen] | candidate | 0.4455 [0.4116, 0.4812] | 68 | 0.768 | 0.059 | 0.544 / 0.504 / 0.402 / 0.331 | 2564 | 0.01 | hidden, dropout | 45.7 | ok |
| f3_fixedmix[frozen] | reference | 0.4455 [0.4119, 0.4794] | 68 | 0.768 | 0.059 | 0.544 / 0.504 / 0.402 / 0.331 | 2285 | 0.01 | hidden, dropout | 9.0 | ok |
| f3_arbiter[frozen] | candidate | 0.4457 [0.4120, 0.4808] | 68 | 0.768 | 0.060 | 0.545 / 0.506 / 0.402 / 0.330 | 2290 | 0.0 | hidden, dropout, bandwidth | 79.0 | ok |
| f3_fixedshare[frozen] | reference | 0.4457 [0.4110, 0.4812] | 68 | 0.768 | 0.060 | 0.545 / 0.506 / 0.402 / 0.330 | 2284 | 0.0 | hidden, dropout | 0.1 | ok |
| f1_latent[mismatched] | candidate | 0.4462 [0.4107, 0.4818] | 68 | 0.766 | 0.061 | 0.545 / 0.505 / 0.402 / 0.333 | 2564 | 0.01 | hidden, dropout | 45.7 | ok |
| f3_arbiter[mismatched] | candidate | 0.4525 [0.4207, 0.4859] | 68 | 0.773 | 0.073 | 0.547 / 0.514 / 0.409 / 0.340 | 2290 | 0.0 | hidden, dropout, bandwidth | 79.0 | ok |
| f1_latent[twin] | candidate | 0.4539 [0.4188, 0.4913] | 68 | 0.767 | 0.064 | 0.543 / 0.513 / 0.407 / 0.353 | 2284 | 0.01 | hidden, dropout | 45.7 | ok |
| f3_arbiter[twin] | candidate | 0.4541 [0.4217, 0.4901] | 68 | 0.766 | 0.065 | 0.548 / 0.515 / 0.414 / 0.339 | 2284 | 0.0 | hidden, dropout, bandwidth | 79.0 | ok |
| f2_world_policy[twin] | candidate | 0.4608 [0.4235, 0.4989] | 68 | 0.772 | 0.068 | 0.546 / 0.512 / 0.421 / 0.365 | 8004 | 0.01 | hidden, dropout, prior_scale | 71.4 | ok |
| f2_delta | reference | 0.5732 [0.5476, 0.5989] | 68 | 0.649 | 0.064 | 0.541 / 0.503 / 0.551 / 0.698 | 21 | 0.02 | delta_alpha | 1.7 | ok |
| f2_world_policy | candidate | 0.5793 [0.5535, 0.6042] | 68 | 0.649 | 0.066 | 0.542 / 0.503 / 0.562 / 0.709 | 8156 | 0.01 | hidden, dropout, prior_scale | 71.4 | ok |
| f2_delta[frozen] | reference | 0.5890 [0.5638, 0.6160] | 68 | 0.651 | 0.097 | 0.557 / 0.522 / 0.573 / 0.704 | 21 | 0.02 | delta_alpha | 1.7 | ok |
| f2_world_policy[frozen] | candidate | 0.5920 [0.5657, 0.6199] | 68 | 0.651 | 0.099 | 0.556 / 0.519 / 0.578 / 0.714 | 8156 | 0.01 | hidden, dropout, prior_scale | 71.4 | ok |
| f2_world_policy[mismatched] | candidate | 0.6218 [0.5954, 0.6465] | 68 | 0.645 | 0.110 | 0.588 / 0.583 / 0.605 / 0.712 | 8156 | 0.01 | hidden, dropout, prior_scale | 71.4 | ok |

| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | sd | n |
|---|---|---|---|---|---|---|
| f1_latent − pop_gru | vs_best_comparator | -0.0012 | [-0.0079, +0.0056] | 0.3775 | 0.0296 | 68 |
| f1_latent − f1_latent[twin] | vs_twin | -0.0113 | [-0.0190, -0.0050] | 0.0005 | 0.0298 | 68 |
| f1_latent − f1_latent[frozen] | vs_frozen | -0.0029 | [-0.0048, -0.0015] | 0.0005 | 0.0069 | 68 |
| f1_latent − f1_latent[mismatched] | vs_mismatched | -0.0036 | [-0.0059, -0.0019] | 0.0005 | 0.0086 | 68 |
| f2_world_policy − pop_gru | vs_best_comparator | +0.1355 | [+0.1126, +0.1570] | 1.0000 | 0.0945 | 68 |
| f2_world_policy − f2_world_policy[twin] | vs_twin | +0.1185 | [+0.0931, +0.1437] | 1.0000 | 0.1113 | 68 |
| f2_world_policy − f2_world_policy[frozen] | vs_frozen | -0.0128 | [-0.0171, -0.0083] | 0.0005 | 0.0182 | 68 |
| f2_world_policy − f2_world_policy[mismatched] | vs_mismatched | -0.0425 | [-0.0554, -0.0306] | 0.0005 | 0.0537 | 68 |
| f3_arbiter − pop_gru | vs_best_comparator | -0.0090 | [-0.0154, -0.0026] | 0.0030 | 0.0275 | 68 |
| f3_arbiter − f3_arbiter[twin] | vs_twin | -0.0193 | [-0.0265, -0.0123] | 0.0005 | 0.0304 | 68 |
| f3_arbiter − f3_arbiter[frozen] | vs_frozen | -0.0109 | [-0.0164, -0.0060] | 0.0005 | 0.0232 | 68 |
| f3_arbiter − f3_arbiter[mismatched] | vs_mismatched | -0.0177 | [-0.0228, -0.0124] | 0.0005 | 0.0222 | 68 |

## Candidate vs best comparator across counted families (Holm)

| candidate | family | estimate | 95% CI | Holm p | meets 0.02 margin |
|---|---|---|---|---|---|
| f1_latent | garcia | -0.0012 | [-0.0079, +0.0056] | 0.3775 | False |
| f2_world_policy | garcia | +0.1355 | [+0.1126, +0.1570] | 1.0000 | False |
| f3_arbiter | garcia | -0.0090 | [-0.0154, -0.0026] | 0.0030 | False |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
