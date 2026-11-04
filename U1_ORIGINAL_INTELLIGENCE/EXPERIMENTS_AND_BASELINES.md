# U1 — Experiments and Baselines (preregistration)

_Written 2026-11-04 **before** any confirmatory universe (1–3) was generated, before any comparator was evaluated
on a shift condition, and before the thresholds in §4 were seen against any meta-RL result. Development history
that preceded this document is disclosed in §7. The freeze (`results/FREEZE.json`) hashes this file and every
module in `U1_IMPLEMENTATION/u1/`; the confirmatory runner refuses to run if either changed._

## 1. Environment (version 2) — `U1_IMPLEMENTATION/u1/env.py`

A **universe** (seeded) fixes hidden laws over six roles: a 3-bit probe signature per role (distinct; each bit is
read correctly with probability 0.9), a 3-bit appearance code per known role (distinct), and a symmetric reaction
law between roles (3–5 reacting role pairs among the five known roles, every known role reacting with at least one;
role 5 — the unseen role — reacts with 1–2 known roles and not with itself).

A **world** is one episode of $T=20$ steps with five entities with hidden roles. Observed: appearance
$x_e\in\mathbb R^4$ (dims 0–2 = $\pm2$ by the role's code, each sign flipped with probability $1-\rho$, plus noise;
dim 3 pure noise). Actions: probe(e) (5; outcome = 3-bit reading, reward 0) and pair(i, j) (10; reaction with
probability 0.9 if the roles' law says they react, else 0.05; reward = reaction). Every world has 1–2 reacting pairs.
Outcome noise uses common random numbers, so paired agents face identical noise.

| condition | what changes relative to training | roles in the world | ρ |
|---|---|---|---|
| `train` / `iid` | nothing (training distribution; different seeds) | 2–3 distinct known roles | 0.9 |
| `cue_reversed` | perceptual correlation reversed | 2–3 distinct known | **0.1** |
| `recombined` | familiar concepts in unfamiliar combinations | **4–5 distinct** known | 0.9 |
| `novel_role` | an unseen concept carries all the reward | 1–2 entities of role 5 + known; every reacting pair involves role 5 | 0.9 (role 5's appearance random) |
| `role_switch` | the world changes mid-episode | 2–3 distinct known; at $t=10$ one entity of a reacting pair changes role so that every pair that reacted stops and 1–2 others react | 0.9 |

Metric: **normalized expected return** = $\sum_t p(a_t)$ / $\sum_t \max_a p(a)$, where $p$ is the true reaction
probability of the chosen pair at that step (0 for probes). Secondary: steps to first reaction; post-switch
normalized return (role_switch).

## 2. Agents (same information for all: appearances, own actions, outcomes, rewards)

| agent | knowledge across worlds | trained on | notes |
|---|---|---|---|
| `ira` (proposed) | concept library (counts) | 300 `train` worlds it plays itself (seeds 100000–100299) | `MATHEMATICAL_FORMULATION.md` §2 + the re-clustering step (§7, D-2) |
| `ira_noactive` | same library | — | Thompson sampling of the assignment, no information-gain term |
| `ira_closed` | same library | — | no fresh labels at evaluation |
| `ira_nocuetrack` | same library | — | cue reliability $w$ fixed at 1 |
| `ira_nochange` | same library | — | no contradiction-triggered re-identification |
| `ira_nolibrary` | none (empty library) | — | within-world inference only |
| **`metarl`** (primary comparator) | GRU weights | 20,000 A2C updates × 128 fresh `train` worlds = **2.56 M episodes** per seed | RL²-style; mean over **two independently trained seeds** (`metarl_s0`, `metarl_s1`); weights frozen at evaluation |
| `entity_bayes` | pooled reaction rate | — | Beta-Bernoulli per pair, Thompson sampling, no concepts |
| `ucb` | none | — | UCB1 over pairs |
| `random` | none | — | uniform over pairs |

The library of `ira` is learned once per universe and frozen for evaluation (all ablations use the same library).
The meta-RL agent receives roughly 8,500× more training episodes than IRA and more training compute (≈ 15–30 CPU-min
vs ≈ 2–3 CPU-min per universe); IRA uses more computation per decision (exact enumeration, ≈ 0.2–0.4 s per world vs
≈ 0.01 s). Both are reported.

## 3. Development (universe 0 only; seeds 0–199) — selection rules fixed here

* IRA: $\beta\in\{0.5,1,2\}$ selected by mean normalized return on **`iid` only**, each with its own library trained
  at that $\beta$. Everything else fixed a priori: $\tau=4$, $\alpha=1$, $K_{top}=7$, $M=2$, prune 14 nats,
  re-cluster every 50 training worlds (40 Gibbs sweeps + 5 greedy), 300 training worlds.
* Meta-RL: (hidden 128, lr 1e-3) vs (hidden 256, lr 5e-4), and greedy vs sampled actions, selected on **`iid` only**.
  Adequacy check: if the selected meta-RL scores < 0.85 × IRA's dev `iid` score, train 3× longer once.
* The selected settings are written to `results/settings.json` before the freeze.

**Selected (dev `iid`, 200 worlds; `results/dev/select_ira.json`, `results/dev/select_metarl.json`):**

| | candidates (dev `iid` normalized return) | selected |
|---|---|---|
| IRA β | 0.5 → 0.887; **1.0 → 0.899**; 2.0 → 0.834 | β = 1.0 |
| meta-RL | **h128/lr 1e-3 greedy 0.931**; h128 sampled 0.925; h256/lr 5e-4 greedy 0.925; h256 sampled 0.918 | hidden 128, lr 1e-3, 20,000 updates, greedy actions |

Adequacy check passed: 0.931 ≥ 0.85 × 0.899. **Note: the selected meta-RL agent is better than IRA in-distribution
on dev (0.931 vs 0.899).**

## 4. Hypotheses and decision thresholds (confirmatory universes 1, 2, 3; seeds 10000–10199 per condition)

**H1 (primary).** For each shift condition $c$ ∈ {cue_reversed, recombined, novel_role, role_switch}, the paired
difference $\delta_c$ = IRA − metaRL in normalized return over 600 worlds (3 universes × 200) satisfies
(i) $\delta_c \ge 0.10$, (ii) Holm-adjusted (over the 4 conditions) one-sided $p<0.05$ with
$p=\max(p_{\text{bootstrap}},p_{t})$, and (iii) $\delta_c>0$ in each of the three universes.
**H1 is supported** iff ≥ 3 of 4 conditions pass **and** IRA is non-inferior in-distribution
($\delta_{\text{iid}}\ge-0.05$).

**Mechanism attributions** (Holm over M1–M6; each passes iff estimate ≥ margin and Holm $p<0.05$):

| | comparison (IRA − ablation) | condition(s) | margin |
|---|---|---|---|
| M1 | `ira_nocuetrack` | cue_reversed | 0.05 |
| M2 | `ira_closed` | novel_role | 0.05 |
| M3 | `ira_nochange` | role_switch | 0.05 |
| M4 | `ira_noactive` | mean over all five | 0.03 |
| M5 | `ira_nolibrary` | mean over iid and recombined | 0.03 |
| M6 | `entity_bayes` | iid | 0.10 |

**A (accumulation without retraining, secondary).** Per confirmatory universe, a sequence of 50 `novel_role`
worlds (seeds 20000–20049): IRA consolidating additively after each world vs. the same IRA with the library frozen,
on identical worlds. Passes iff the paired difference over the last 25 worlds of each sequence (75 pairs) is
≥ 0.05 with one-sided $p<0.05$.

## 5. Decision rule (fixed now)

* **H1 supported and at least one of M1–M3 passes** → *advance*: a reproducible empirical advantage of this
  combination over a strongly meta-trained recurrent agent under controlled shifts in this environment family
  (novelty level: new empirical finding about a combination; **not** a new principle).
* **H1 supported but none of M1–M3 passes** → the advantage is not explained by the shift-specific mechanisms;
  report as an unexplained advantage, no mechanism claim.
* **H1 not supported** → *reject* the hypothesis that this combination provides an advantage the meta-trained
  agent cannot match under these shifts; report which conditions, if any, favour either agent.
* A is reported separately; failing A rejects the "accumulates without retraining" part of the claim.

## 6. What is reported regardless of outcome

All agents × all conditions (means, paired contrasts with CIs, per-universe estimates), compute and parameter counts,
the learned libraries and their agreement with the true roles (adjusted Rand index; diagnostic only), training
curves, every dev run, and every abandoned design.

## 7. Development history disclosed before the freeze

* **D-1 (environment v1 rejected).** In v1 an IRA with an *empty* library already retained 0.92 of the oracle
  in-distribution, the same as with a perfect library (`results/dev/env_v1_findings.md`). v2 was designed so that
  knowledge can matter (sparse laws, informative probes, informative appearance code).
* **D-2 (additive consolidation collapses).** Plain MAP consolidation after each world collapsed the library into
  2–3 mixed concepts (adjusted Rand index ≈ 0) in v1 and v2. Offline re-clustering of the stored training worlds
  (collapsed Gibbs sampling under the agent's own model; `recluster.py`) was added. **This is a change to the
  proposed mechanism:** the library is not purely additive during training — it is periodically re-estimated from
  stored experience (closer to the hierarchical Bayesian model of Kemp et al. than to the additive design). The
  accumulation test (A) keeps the additive path.
* **D-3 (IRA evaluated on shift conditions during development).** While diagnosing D-1/D-2, IRA (and UCB,
  entity_bayes) were run on all five conditions in universe 0 (dev seeds). Observed IRA normalized returns with a
  learned library, 40 worlds each: iid 0.94, cue_reversed 0.87, recombined 0.92, novel_role 0.86, role_switch 0.71.
  An oracle library (true roles) scored *lower* than a learned or empty library on novel_role (0.60) and
  cue_reversed (0.73) in one diagnostic, i.e. strong knowledge made novelty harder to detect. No design choice was
  tuned on these numbers; the CRP concentration and thresholds were left at their a-priori values. The meta-RL
  agent was **not** evaluated on any shift condition before this document.
* **D-5 (full dev evaluation of IRA, its ablations and the non-neural baselines before the freeze).** Universe 0,
  dev seeds 0–199, selected β = 1.0 library (`results/u0/dev/`, log `results/dev/logs/dev_eval_u0.log`):

  | agent | iid | cue_reversed | recombined | novel_role | role_switch |
  |---|---|---|---|---|---|
  | ira | 0.899 | 0.854 | 0.863 | 0.839 | 0.681 |
  | ira_noactive | 0.618 | 0.581 | 0.576 | 0.616 | 0.476 |
  | ira_closed | 0.897 | 0.850 | 0.865 | 0.832 | 0.679 |
  | ira_nocuetrack | 0.883 | 0.849 | 0.851 | 0.833 | 0.659 |
  | ira_nochange | 0.908 | 0.868 | 0.879 | 0.868 | 0.510 |
  | ira_nolibrary | 0.804 | 0.831 | 0.790 | 0.805 | 0.443 |
  | entity_bayes | 0.524 | 0.537 | 0.514 | 0.535 | 0.357 |
  | ucb | 0.465 | 0.478 | 0.432 | 0.466 | 0.256 |
  | random | 0.227 | 0.224 | 0.209 | 0.229 | 0.264 |

  On dev, M1 (cue tracking) and M2 (open concept set) look **inert** (differences ≤ 0.007) and are expected to
  fail; M3, M4, M5, M6 look large. These observations did **not** change any threshold, margin or mechanism; they
  are recorded so that no later reader mistakes a predicted failure for a surprise or a predicted pass for a
  discovery. The meta-RL agent was still not evaluated on any shift condition before the freeze.
* **D-4 (speed).** Prediction skips assignments more than 14 nats below the best (each < 1e-6 of the mass);
  measured differences from unpruned inference on 40 dev worlds per condition were ≤ 0.02.
