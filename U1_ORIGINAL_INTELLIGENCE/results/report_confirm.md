### Normalized expected return (confirm; universes [1, 2, 3]; n per cell shown in the first row)

| agent | iid | cue_reversed | recombined | novel_role | role_switch |
|---|---|---|---|---|---|
| `ira` | 0.890 | 0.856 | 0.894 | 0.837 | 0.673 |
| `metarl` | 0.937 | 0.870 | 0.940 | 0.865 | 0.500 |
| `metarl_s0` | 0.934 | 0.866 | 0.941 | 0.866 | 0.504 |
| `metarl_s1` | 0.941 | 0.874 | 0.939 | 0.864 | 0.497 |
| `ira_noactive` | 0.631 | 0.588 | 0.613 | 0.579 | 0.473 |
| `ira_closed` | 0.888 | 0.862 | 0.893 | 0.837 | 0.677 |
| `ira_nocuetrack` | 0.873 | 0.837 | 0.877 | 0.811 | 0.641 |
| `ira_nochange` | 0.906 | 0.874 | 0.913 | 0.857 | 0.494 |
| `ira_nolibrary` | 0.819 | 0.809 | 0.814 | 0.796 | 0.447 |
| `entity_bayes` | 0.529 | 0.523 | 0.535 | 0.523 | 0.345 |
| `ucb` | 0.475 | 0.460 | 0.483 | 0.456 | 0.252 |
| `random` | 0.229 | 0.226 | 0.229 | 0.221 | 0.240 |

(n = 600 worlds per cell.)

### Primary contrasts: IRA − meta-RL (mean of two seeds), paired over worlds

| condition | estimate | 95% CI | per universe | p (one-sided) | Holm p | passes (≥ 0.10, Holm < 0.05, all universes > 0) |
|---|---|---|---|---|---|---|
| iid | -0.0472 | [-0.0602, -0.0350] | -0.042, -0.067, -0.033 | 1.0000 | — | (non-inferiority: ≥ −0.05) |
| cue_reversed | -0.0140 | [-0.0296, +0.0015] | +0.039, -0.046, -0.035 | 0.9632 | 1.0000 | no |
| recombined | -0.0460 | [-0.0620, -0.0304] | -0.035, -0.057 | 1.0000 | 1.0000 | no |
| novel_role | -0.0283 | [-0.0430, -0.0131] | -0.072, -0.009, -0.003 | 0.9999 | 1.0000 | no |
| role_switch | +0.1734 | [+0.1600, +0.1862] | +0.168, +0.202, +0.151 | <1e-4 | <1e-4 | **yes** |

Shift conditions passed: **1 of 4**; IRA non-inferior in-distribution: **True**; **H1 supported: False**.

### Mechanism attributions (IRA − ablation; Holm over M1–M6)

| mechanism | comparator | condition(s) | margin | estimate | 95% CI | per universe | Holm p | passes |
|---|---|---|---|---|---|---|---|---|
| M1 cue-validity tracking | `ira_nocuetrack` | cue_reversed | 0.05 | +0.0188 | [+0.0049, +0.0324] | -0.025, +0.063, +0.018 | 0.0080 | no |
| M2 open concept set | `ira_closed` | novel_role | 0.05 | +0.0001 | [-0.0053, +0.0057] | -0.000, -0.001, +0.002 | 0.5018 | no |
| M3 re-identification | `ira_nochange` | role_switch | 0.05 | +0.1798 | [+0.1686, +0.1907] | +0.187, +0.186, +0.167 | <1e-4 | **yes** |
| M4 active identification | `ira_noactive` | iid, cue_reversed, recombined, novel_role, role_switch | 0.03 | +0.2605 | [+0.2520, +0.2688] | +0.263, +0.258 | <1e-4 | **yes** |
| M5 cross-world library | `ira_nolibrary` | iid, recombined | 0.03 | +0.0827 | [+0.0681, +0.0970] | +0.092, +0.074 | <1e-4 | **yes** |
| M6 concepts vs. none | `entity_bayes` | iid | 0.1 | +0.3609 | [+0.3474, +0.3744] | +0.371, +0.338, +0.373 | <1e-4 | **yes** |

### Accumulation without retraining (last 25 of 50 novel-role worlds per universe)

accumulating − frozen = **+0.0271** [+0.0057, +0.0504], per universe +0.021, +0.023, +0.038, p = 0.0090, n = 75; **passes: False** (margin 0.05).

### Diagnostics

* ira post-switch norm: 0.5043
* metarl post-switch norm: 0.1117
* ira_nochange post-switch norm: 0.1413
* ira iid steps to first reaction: 1.57
* ira novel_role steps to first reaction: 2.34
* metarl iid steps to first reaction: 1.07
* metarl novel_role steps to first reaction: 2.14
* entity_bayes iid steps to first reaction: 3.02
* entity_bayes novel_role steps to first reaction: 3.17
* ucb iid steps to first reaction: 3.13
* ucb novel_role steps to first reaction: 3.22
