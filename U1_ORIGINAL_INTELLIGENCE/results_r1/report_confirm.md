### Normalized expected return (confirm; universes [4, 5, 6, 7, 8])

| agent | iid | cue_reversed | recombined | novel_role | role_switch |
|---|---|---|---|---|---|
| `ira` | 0.905 | 0.816 | 0.881 | 0.747 | 0.712 |
| `ira_beta1` | 0.848 | 0.738 | 0.834 | 0.656 | 0.653 |
| `ira_noactive` | 0.912 | 0.822 | 0.881 | 0.806 | 0.717 |
| `ira_closed` | 0.899 | 0.816 | 0.887 | 0.742 | 0.706 |
| `ira_nocuetrack` | 0.900 | 0.696 | 0.873 | 0.710 | 0.701 |
| `ira_nochange` | 0.930 | 0.831 | 0.903 | 0.765 | 0.492 |
| `ira_nolibrary` | 0.770 | 0.775 | 0.791 | 0.780 | 0.423 |
| `gru_narrow` | 0.936 | 0.733 | 0.887 | 0.791 | 0.498 |
| `entity_bayes` | 0.553 | 0.560 | 0.566 | 0.573 | 0.360 |
| `ucb` | 0.444 | 0.445 | 0.467 | 0.458 | 0.244 |
| `random` | 0.207 | 0.214 | 0.222 | 0.221 | 0.236 |

### H1: IRA − `gru_narrow` (universe-level; Holm over the 4 shift conditions)

| condition | estimate | per universe | universes > 0 | t | p | Holm p | passes | world-level 95% CI |
|---|---|---|---|---|---|---|---|---|
| iid | -0.0305 | -0.050, -0.008, -0.043, +0.004, -0.054 | 1 | -2.588 | 0.9696 | — | non-inferiority ≥ −0.05 | [-0.0470, -0.0141] |
| cue_reversed | +0.0836 | +0.179, -0.059, -0.033, -0.040, +0.372 | 2 | 0.994 | 0.1882 | 0.5647 | no | [+0.0574, +0.1101] |
| recombined | -0.0059 | -0.033, -0.024, -0.051, -0.056, +0.134 | 1 | -0.165 | 0.5617 | 1.0000 | no | [-0.0249, +0.0137] |
| novel_role | -0.0444 | +0.034, -0.186, +0.012, -0.028, -0.054 | 2 | -1.149 | 0.8427 | 1.0000 | no | [-0.0700, -0.0188] |
| role_switch | +0.2138 | +0.289, +0.225, +0.147, +0.184, +0.224 | 5 | 9.036 | 0.0004 | 0.0017 | **yes** | [+0.2008, +0.2266] |

Shift conditions passed: **1 of 4**; iid non-inferior: **True**; H1 against `gru_narrow`: **False**

### H1: IRA − `gru_broad` (universe-level; Holm over the 4 shift conditions)

not run (stage 2 runs only if H1 holds against gru_narrow)

### H1: IRA − `rel_broad` (universe-level; Holm over the 4 shift conditions)

not run (stage 2 runs only if H1 holds against gru_narrow)

**H1 supported: False**

### Mechanisms (IRA − ablation; Holm over M1–M5)

| mechanism | ablation | condition(s) | margin | estimate | per universe | Holm p | passes |
|---|---|---|---|---|---|---|---|
| M1 cue-validity tracking | `ira_nocuetrack` | cue_reversed | 0.05 | +0.1202 | +0.031, +0.084, +0.031, +0.004, +0.450 | 0.3353 | no |
| M2 open concept set | `ira_closed` | novel_role | 0.05 | +0.0051 | +0.012, +0.035, -0.005, -0.002, -0.014 | 0.5792 | no |
| M3 re-identification | `ira_nochange` | role_switch | 0.05 | +0.2207 | +0.304, +0.191, +0.156, +0.207, +0.245 | 0.0019 | **yes** |
| M4 active identification (information gain) | `ira_noactive` | iid, cue_reversed, recombined, novel_role, role_switch | 0.03 | -0.0156 | -0.030, -0.040, -0.004, -0.002, -0.001 | 0.9351 | no |
| M5 cross-world library | `ira_nolibrary` | iid | 0.03 | +0.1354 | +0.109, +0.129, +0.126, +0.151, +0.162 | 0.0003 | **yes** |

**Accumulation (A):** accumulating − frozen (last 25 of 50 unseen-role worlds) = +0.0059, per universe -0.005, -0.012, +0.025, +0.051, -0.029, p = 0.3487; passes: **False** (margin 0.05).

### Diagnostics

* ira post-switch norm: 0.5525
* ira_nochange post-switch norm: 0.1043
* gru_narrow post-switch norm: 0.102
* ira iid probes per world: 0.16
* ira cue_reversed probes per world: 0.4
* ira recombined probes per world: 0.28
* ira novel_role probes per world: 0.94
* ira role_switch probes per world: 0.15
* gru_narrow iid probes per world: 0.0
* gru_narrow cue_reversed probes per world: 0.0
* gru_narrow recombined probes per world: 0.0
* gru_narrow novel_role probes per world: 0.0
* gru_narrow role_switch probes per world: 0.0
