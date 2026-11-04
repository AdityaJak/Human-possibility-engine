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

**What R1 changed** (`EXPERIMENTS_AND_BASELINES.md` §R1, frozen at 2edb4e7):
* information gain over the concept assignment **and** the laws, so probes now have value;
* an unbiased prior for unobserved concept pairs;
* β = 0 as the "no active identification" ablation;
* feasible universes;
* universe-level statistics;
* a staged design. The broad-trained GRU and the relational comparator run only if H1 first holds against the
  narrow GRU.

β = 0.5 was selected on dev `iid`.

**Deviation R1-X1 (analysis code).**
* What happened: the stage-1 summary refused to run. The frozen check "was this stage-2 comparator run?" applied
  `any()` to generator objects, which are always truthy, so the skip was never recognised.
* The loud stage gate added after the critic's review stopped the run instead of deciding. No result was printed or
  read.
* The line was fixed (`u1/stats.py`, commented) and the preregistered analysis then run unchanged. Because the code
  changed after the freeze, the freeze hash no longer matches `stats.py`. No evaluation code changed.

### 2.1 All agents (universe means of 100 worlds; 500 worlds per cell)

| agent | iid | cue_reversed | recombined | novel_role | role_switch |
|---|---|---|---|---|---|
| `ira` (β = 0.5) | 0.905 | **0.816** | 0.881 | 0.747 | **0.712** |
| `gru_narrow` (20k updates, 2.56 M episodes) | **0.936** | 0.733 | **0.887** | **0.791** | 0.498 |
| `ira_noactive` (β = 0) | 0.912 | 0.822 | 0.881 | 0.806 | 0.717 |
| `ira_beta1` (β = 1, descriptive) | 0.848 | 0.738 | 0.834 | 0.656 | 0.653 |
| `ira_closed` | 0.899 | 0.816 | 0.887 | 0.742 | 0.706 |
| `ira_nocuetrack` | 0.900 | 0.696 | 0.873 | 0.710 | 0.701 |
| `ira_nochange` | 0.930 | 0.831 | 0.903 | 0.765 | 0.492 |
| `ira_nolibrary` | 0.770 | 0.775 | 0.791 | 0.780 | 0.423 |
| `entity_bayes` | 0.553 | 0.560 | 0.566 | 0.573 | 0.360 |
| `ucb` | 0.444 | 0.445 | 0.467 | 0.458 | 0.244 |
| `random` | 0.207 | 0.214 | 0.222 | 0.221 | 0.236 |

### 2.2 H1: IRA − `gru_narrow` (unit = universe; one-sided t over 5 universe means; Holm over 4 shifts)

| condition | estimate | per universe (u4–u8) | > 0 | Holm p | passes |
|---|---|---|---|---|---|
| iid | −0.031 | −0.050, −0.008, −0.043, +0.004, −0.054 | 1 | — | non-inferior (≥ −0.05): yes |
| cue_reversed | +0.084 | +0.179, −0.059, −0.033, −0.040, +0.372 | 2 | 0.56 | no |
| recombined | −0.006 | −0.033, −0.024, −0.051, −0.056, +0.134 | 1 | 1.0 | no |
| novel_role | −0.044 | +0.034, −0.186, +0.012, −0.028, −0.054 | 2 | 1.0 | no |
| role_switch | **+0.214** | +0.289, +0.225, +0.147, +0.184, +0.224 | 5 | 0.002 | **yes** |

**H1 is not supported (1 of 4 shift conditions).** By the preregistered staging, H1 requires all three comparators,
so stage 2 (`gru_broad`, `rel_broad`) was not run. On dev, `rel_broad` exceeded IRA in four of five conditions,
including `role_switch` (R1-D).

### 2.3 Mechanisms (Holm over M1–M5)

| | ablation | condition(s) | estimate | per universe | Holm p | passes |
|---|---|---|---|---|---|---|
| M1 cue-validity tracking | `ira_nocuetrack` | cue_reversed | +0.120 | +0.031, +0.084, +0.031, +0.004, +0.450 | 0.34 | no |
| M2 open concept set | `ira_closed` | novel_role | +0.005 | +0.012, +0.035, −0.005, −0.002, −0.014 | 0.58 | no |
| M3 re-identification | `ira_nochange` | role_switch | **+0.221** | +0.304, +0.191, +0.156, +0.207, +0.245 | 0.002 | **yes** |
| M4 information-gain action selection | `ira_noactive` (β = 0) | all five | **−0.016** | −0.030, −0.040, −0.004, −0.002, −0.001 | 0.94 | no |
| M5 cross-world library | `ira_nolibrary` | iid | **+0.135** | +0.109, +0.129, +0.126, +0.151, +0.162 | < 0.001 | **yes** |

**Accumulation (A):** accumulating − frozen = +0.006 (per universe −0.005, −0.012, +0.025, +0.051, −0.029),
p = 0.35. **Fails.**

**Descriptive** (`results_r1/diagnostics_concepts.json`):

| diagnostic | value |
|---|---|
| IRA probes per world | 0.16 (iid), 0.40 (cue_reversed), 0.28 (recombined), 0.94 (novel_role), 0.15 (role_switch) |
| unseen-role worlds in which any entity got a new concept | 2%, 0%, 0%, 0%, 0% (u4–u8) |
| library vs. true roles (adjusted Rand index) | 0.28, 0.43, 0.47, 0.24, 0.53 |
| post-switch normalized return | IRA 0.553, `ira_nochange` 0.104, `gru_narrow` 0.102 |

### 2.4 What attempt 2 shows

* **Fixing the information gain changed behaviour but did not help.**
  * IRA now probes, most under novelty (0.94 per world on `novel_role`).
  * Yet removing information-gain-driven action selection entirely does not hurt; it helps slightly (M4 −0.016;
    `ira_noactive` beats `ira` on `novel_role`, 0.806 vs 0.747).
  * More information seeking is worse still (`ira_beta1`, every condition).
  * Within a 20-step episode, greedy exploitation of the posterior beats paying for experiments in this
    environment family.
* **Concepts are not formed for novelty.**
  * The open concept set is inert (M2), and a new concept was opened in 1 of 500 unseen-role worlds.
  * The learned library makes the unseen role *harder*: `ira` 0.747 vs `ira_nolibrary` 0.780. Well-learned concepts
    absorb the novel entity, so knowledge entrenches.
* **The cross-world library is the one learned component that clearly helps** (M5 +0.135 in-distribution). This is
  ordinary hierarchical-Bayesian transfer, not a new mechanism.
* **The only robust advantage over the meta-trained agent is the role switch.**
  * It is fully explained by the designers' surprise-reset rule (M3).
  * Meta-learners trained with mid-episode switches matched or exceeded IRA there on dev (`gru_broad` 0.673,
    `rel_broad` 0.770 vs IRA 0.682). This is not a held-out comparison: the dev models were trained before t = 10
    was excluded from their training switch times.
* **Cue reversal hurts the narrow GRU strongly in two universes** (u4 and u8; IRA − GRU +0.18 and +0.37), while
  IRA's cue tracking protects it there (M1 +0.45 in u8). The effect is not consistent across universes (2 of 5),
  so it does not pass. This is the closest thing to the predicted "incidental correlation" failure, and it is
  universe-dependent: it depends on how the reversed codes map onto other roles' codes.

## 3. Both attempts together

| claim | attempt 1 | attempt 2 |
|---|---|---|
| H1: IRA beats a meta-trained recurrent agent under ≥ 3 of 4 shifts | not supported (1/4) | not supported (1/4) |
| active identification explains performance | not tested (agent never probed) | **no** (M4 −0.016) |
| new concepts opened for the unseen concept | no (M2 0.000) | no (M2 +0.005; 1/500 worlds) |
| cue-validity tracking separates general from incidental | no (M1 +0.019) | inconsistent (M1 +0.120, 2 universes carry it) |
| re-identification after change | yes (M3 +0.180), hand-written rule | yes (M3 +0.221), hand-written rule |
| cross-world library | yes (M5 +0.083) | yes (M5 +0.135) |
| accumulation without retraining | no (+0.027) | no (+0.006) |
