# Social mechanism v2 — dev-partition results (EXPLORATORY)

Seeds [1000, 1001, 1002, 1003, 1004]; 100 agents × 15 years; policy `developmental_social`; CPU 0.3 h.

| arm | n | accuracy 10–60 | youth 10–20 | mean friends | socialise share | demos/agent-yr | repeat share | instructional | info gain | trust honest | trust misinf. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_connected | 5 | 0.807 | 0.817 | 0.055 | 0.188 | — | — | — | — | — | — |
| v1_isolated | 5 | 0.805 | 0.810 | 0.000 | 0.198 | — | — | — | — | — | — |
| v2_connected | 5 | 0.810 | 0.816 | 1.459 | 0.210 | 10.230 | 0.583 | 0.528 | 0.152 | 0.603 | — |
| v2_isolated | 5 | 0.813 | 0.820 | 0.000 | 0.212 | 17.235 | 0.957 | 0.512 | 0.129 | nan | — |
| v2_reward_only | 5 | 0.789 | 0.806 | 0.027 | 0.201 | 0.000 | 0.588 | 0.000 | 0.000 | 0.377 | — |
| v2_no_learn_signal | 5 | 0.809 | 0.812 | 1.292 | 0.186 | 9.534 | 0.584 | 0.543 | 0.142 | 0.604 | — |
| v1_misinfo | 5 | 0.798 | 0.806 | 0.041 | 0.190 | — | — | — | — | — | — |
| v2_misinfo | 5 | 0.810 | 0.817 | 1.279 | 0.188 | 10.334 | 0.581 | 0.535 | 0.141 | 0.600 | 0.404 |

| contrast | treatment − control | metric | estimate | 95% CI (paired, seeds) |
|---|---|---|---|---|
| connection_effect_v2 | v2_connected − v2_isolated | accuracy_10_60 | -0.0030 | [-0.0074, +0.0022] (n=5) |
| connection_effect_v1 | v1_connected − v1_isolated | accuracy_10_60 | +0.0026 | [-0.0017, +0.0070] (n=5) |
| information_effect_v2 | v2_connected − v2_reward_only | accuracy_10_60 | +0.0209 | [+0.0146, +0.0276] (n=5) |
| v2_vs_v1_connected | v2_connected − v1_connected | accuracy_10_60 | +0.0029 | [-0.0015, +0.0073] (n=5) |
| learn_signal_effect | v2_connected − v2_no_learn_signal | accuracy_10_60 | +0.0013 | [-0.0050, +0.0058] (n=5) |
| youth_connection_effect_v2 | v2_connected − v2_isolated | youth_accuracy_10_20 | -0.0043 | [-0.0110, +0.0020] (n=5) |
| youth_information_effect_v2 | v2_connected − v2_reward_only | youth_accuracy_10_20 | +0.0092 | [-0.0049, +0.0288] (n=5) |
| misinformation_harm_v2 | v2_connected − v2_misinfo | accuracy_10_60 | -0.0001 | [-0.0035, +0.0026] (n=5) |
| misinformation_harm_v1 | v1_connected − v1_misinfo | accuracy_10_60 | +0.0094 | [+0.0067, +0.0120] (n=5) |
| friends_v2_vs_v1 | v2_connected − v1_connected | mean_friends | +1.4035 | [+1.2815, +1.5674] (n=5) |
| friends_v2_vs_reward_only | v2_connected − v2_reward_only | mean_friends | +1.4312 | [+1.3136, +1.5871] (n=5) |
| socialize_share_v2_vs_v1 | v2_connected − v1_connected | socialize_share | +0.0212 | [-0.0021, +0.0504] (n=5) |
| socialize_share_learn_signal | v2_connected − v2_no_learn_signal | socialize_share | +0.0233 | [+0.0057, +0.0407] (n=5) |
