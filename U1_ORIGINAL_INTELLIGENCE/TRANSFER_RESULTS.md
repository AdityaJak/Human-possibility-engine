# U1 — Transfer Results

Two confirmatory attempts were preregistered and run; both are reported in full.

* **Attempt 1** tested the agent as frozen at commit 6776cfc. The independent critique later showed that this agent
  never used its probe actions, i.e. it did not run the experiments the principle is about.
* **Attempt 2 (revision R1)** fixed that and the other critique findings, and added stronger comparators.

All numbers below are copied from `results/summary_confirm.json` (attempt 1) and `results_r1/summary_confirm.json`
(attempt 2), produced by `u1.stats`. Metric: normalized expected return = (true reaction probability of the chosen
pairs, summed over steps) / (the oracle's).

## 1. Attempt 1 (frozen code; confirmatory universes 1–3, 200 worlds per condition per universe)

**Deviation.** Universe 2 cannot generate a `recombined` world: under its laws, 1–2 reacting pairs among 4–5 distinct
roles is impossible, and the generator lacked a feasibility check. The confirmatory chain stopped there. The frozen
code was then run on universe 2's other four conditions. `recombined` is therefore pooled over universes 1 and 3
(n = 400), and contrasts that average over all five conditions (M4) use universes 1 and 3 only. No code was changed
for these runs (each invocation re-checked the freeze hash).

### 1.1 All agents (n = 600 worlds per cell; `recombined` n = 400)

| agent | iid | cue_reversed | recombined | novel_role | role_switch |
|---|---|---|---|---|---|
| `ira` | 0.890 | 0.856 | 0.894 | 0.837 | 0.673 |
| `metarl` (mean of 2 seeds) | **0.937** | **0.870** | **0.940** | **0.865** | 0.500 |
| `ira_noactive` (Thompson) | 0.631 | 0.588 | 0.613 | 0.579 | 0.473 |
| `ira_closed` | 0.888 | 0.862 | 0.893 | 0.837 | **0.677** |
| `ira_nocuetrack` | 0.873 | 0.837 | 0.877 | 0.811 | 0.641 |
| `ira_nochange` | 0.906 | 0.874 | 0.913 | 0.857 | 0.494 |
| `ira_nolibrary` | 0.819 | 0.809 | 0.814 | 0.796 | 0.447 |
| `entity_bayes` | 0.529 | 0.523 | 0.535 | 0.523 | 0.345 |
| `ucb` | 0.475 | 0.460 | 0.483 | 0.456 | 0.252 |
| `random` | 0.229 | 0.226 | 0.229 | 0.221 | 0.240 |

Per universe:

| agent | universe | iid | cue_reversed | recombined | novel_role | role_switch |
|---|---|---|---|---|---|---|
| `ira` | 1 | 0.898 | 0.851 | 0.907 | 0.845 | 0.662 |
| `ira` | 2 | 0.860 | 0.834 | — | 0.848 | 0.688 |
| `ira` | 3 | 0.913 | 0.883 | 0.881 | 0.817 | 0.671 |
| `metarl_s0` / `s1` | 1 | 0.942 / 0.937 | 0.812 / 0.811 | 0.942 / 0.941 | 0.919 / 0.915 | 0.493 / 0.496 |
| `metarl_s0` / `s1` | 2 | 0.918 / 0.936 | 0.872 / 0.890 | — | 0.856 / 0.859 | 0.492 / 0.480 |
| `metarl_s0` / `s1` | 3 | 0.942 / 0.949 | 0.913 / 0.922 | 0.940 / 0.937 | 0.823 / 0.818 | 0.526 / 0.514 |

### 1.2 Primary hypothesis H1 (IRA − meta-RL, paired over worlds)

| condition | estimate | 95% CI | per universe | Holm p | passes (≥ 0.10, Holm < 0.05, every universe > 0) |
|---|---|---|---|---|---|
| iid | −0.0472 | [−0.0602, −0.0350] | −0.042, −0.067, −0.033 | — | non-inferiority (≥ −0.05): met, barely |
| cue_reversed | −0.0140 | [−0.0296, +0.0015] | +0.039, −0.046, −0.035 | 1.0 | no |
| recombined | −0.0460 | [−0.0620, −0.0304] | −0.035, —, −0.057 | 1.0 | no |
| novel_role | −0.0283 | [−0.0430, −0.0131] | −0.072, −0.009, −0.003 | 1.0 | no |
| role_switch | **+0.1734** | [+0.1600, +0.1862] | +0.168, +0.202, +0.151 | < 1e-4 | **yes** |

**H1: not supported** (1 of 4 shift conditions). The meta-trained GRU had seen none of the shifts in training, yet it
matches or beats IRA under cue reversal, recombination and an unseen concept, and in-distribution.

### 1.3 Mechanisms (IRA − ablation; Holm over M1–M6)

| | ablation | condition(s) | estimate | Holm p | passes |
|---|---|---|---|---|---|
| M1 cue-validity tracking | `ira_nocuetrack` | cue_reversed | +0.019 | 0.008 | no (margin 0.05) |
| M2 open concept set | `ira_closed` | novel_role | +0.000 | 0.50 | no |
| M3 re-identification | `ira_nochange` | role_switch | **+0.180** | < 1e-4 | yes |
| M4 "active identification" | `ira_noactive` | all five | +0.261 | < 1e-4 | yes — **but see 1.4** |
| M5 cross-world library | `ira_nolibrary` | iid + recombined | +0.083 | < 1e-4 | yes |
| M6 concepts vs. none | `entity_bayes` | iid | +0.361 | < 1e-4 | yes |

Accumulation (A): accumulating − frozen on the last 25 of 50 unseen-role worlds = +0.027 [+0.006, +0.050]
(per universe +0.021, +0.023, +0.038), p = 0.009. **Fails** the 0.05 margin. The number of concepts never changed
(K 5→5, 8→8, 7→7): no new concept was ever created.

Diagnostics:
* post-switch normalized return: IRA 0.504, meta-RL 0.112, `ira_nochange` 0.141;
* steps to the first reaction (iid): IRA 1.57, meta-RL 1.07;
* library agreement with the true roles (adjusted Rand index): 0.268, 0.302, 0.260;
* IRA probes: **0 in 36,000 confirmatory steps**.

### 1.4 What attempt 1 does and does not show

* **The only advantage, under a mid-episode role switch, comes from a hand-written rule.** IRA's surprise-triggered
  reset (τ = 4 nats) produces it, and removing the rule erases it (M3). The rule was written by designers who knew
  that this shift would be tested (critique F3).
* **The proposed principle was not exercised.**
  * IRA never probed, so identification by "self-designed discriminating experiments" did not happen (critique F1).
    M4 compared Thompson sampling with greedy selection (F4); it did not test active identification.
  * Concepts were never created for the unseen role (M2 = 0.000; K constant).
  * The library only partly recovers the roles (ARI ≈ 0.27).
* **The generic meta-learner transfers well.** It was trained without any shift, yet it lost little under cue
  reversal (0.937 → 0.870), lost nothing under recombination (0.940), and handled the unseen role (0.865) better
  than IRA. The critique's expectation that it "learned incidental correlations" is not borne out, except under the
  role switch.

## 2. Attempt 2 — revision R1 (confirmatory universes 4–8, 100 worlds per condition per universe)

_Filled in from `results_r1/summary_confirm.json` after the R1 run._
