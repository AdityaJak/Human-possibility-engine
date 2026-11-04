# Social mechanism v2 — dev-partition results (EXPLORATORY)

Seeds [1000, 1001, 1002, 1003, 1004]; 100 agents × 15 years; policy `hybrid_exemplar`; CPU 0.448 h.

| arm | n | accuracy 10–60 | youth 10–20 | mean friends | socialise share | demos/agent-yr | repeat share | instructional | info gain | trust honest | trust misinf. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_connected | 5 | 0.801 | 0.810 | 0.077 | 0.190 | — | — | — | — | — | — |
| v2_connected | 5 | 0.806 | 0.819 | 1.681 | 0.192 | 10.147 | 0.584 | 0.566 | 0.057 | 0.628 | — |
| v2_isolated | 5 | 0.806 | 0.807 | 0.000 | 0.234 | 16.869 | 0.956 | 0.522 | 0.056 | nan | — |
| v2_reward_only | 5 | 0.788 | 0.808 | 0.076 | 0.222 | 0.000 | 0.579 | 0.000 | 0.000 | 0.386 | — |

| contrast | treatment − control | metric | estimate | 95% CI (paired, seeds) |
|---|---|---|---|---|
| connection_effect_v2 | v2_connected − v2_isolated | accuracy_10_60 | +0.0000 | [-0.0061, +0.0054] (n=5) |
| information_effect_v2 | v2_connected − v2_reward_only | accuracy_10_60 | +0.0188 | [+0.0145, +0.0249] (n=5) |
| v2_vs_v1_connected | v2_connected − v1_connected | accuracy_10_60 | +0.0054 | [-0.0027, +0.0113] (n=5) |
| youth_connection_effect_v2 | v2_connected − v2_isolated | youth_accuracy_10_20 | +0.0117 | [+0.0056, +0.0174] (n=5) |
| youth_information_effect_v2 | v2_connected − v2_reward_only | youth_accuracy_10_20 | +0.0108 | [+0.0075, +0.0139] (n=5) |
| friends_v2_vs_v1 | v2_connected − v1_connected | mean_friends | +1.6046 | [+1.4785, +1.7287] (n=5) |
| friends_v2_vs_reward_only | v2_connected − v2_reward_only | mean_friends | +1.6051 | [+1.3992, +1.7703] (n=5) |
| socialize_share_v2_vs_v1 | v2_connected − v1_connected | socialize_share | +0.0023 | [-0.0273, +0.0307] (n=5) |
