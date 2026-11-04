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

---

# R1 — Revision 1 (attempt 2): preregistration

_Written 2026-11-04, after the independent critique (`INDEPENDENT_CRITIQUE.md`, commit c5922ab) and finalised
**after attempt 1's confirmatory results were known** (commit dfe2071: H1 not supported). It was written **before**
any R1 confirmatory universe (4–8) was generated. The R1 margins, thresholds and decision structure are those of
attempt 1, carried over unchanged; only the unit of inference (universe-level, F6) and the added comparators are
new. The
sections above (attempt 1) are kept unchanged as the historical record; attempt 1's results are reported in full in
`TRANSFER_RESULTS.md` whatever they are. The R1 freeze (`results_r1/FREEZE.json`) hashes this whole file and every
module in `U1_IMPLEMENTATION/u1/`._

## R1.0 Why a revision

1. **The frozen attempt-1 agent did not implement the principle (critique F1, verified).** Its information gain was
   computed about the concept assignment only, using the posterior-mean law. With no probe counts in the library,
   every assignment predicts 0.5 per probe bit, so the information value of a probe was exactly zero. IRA never
   probed: 0 probes in 56,000 dev and confirmatory steps, and 0 probe counts in all four libraries. Identification
   ran only through pair outcomes, driven by an optimistic prior.
2. **That prior was biased (F11, verified).** The prior reaction rate pooled the agent's own outcomes, which are
   dominated by exploitation: 0.77–0.80 against a true rate near 0.18.
3. **The "no active identification" ablation tested something else (F4).** It compared Thompson sampling against
   greedy action selection.
4. **The attempt-1 generator was invalid for one condition.** Universe 2 cannot produce a `recombined` world
   (1–2 reacting pairs among 4–5 distinct roles is impossible under its laws). The confirmatory run stopped there;
   the frozen code was then run on universe 2's other conditions.
5. **The comparator was narrow (F5), the shifts' robustness was supplied by design (F3), and the statistics treated
   worlds as independent although each universe is one replication (F6).**

## R1.1 Changes (each traced to a finding)

| change | finding | where |
|---|---|---|
| Information gain over the assignment **and** the laws: $\mathrm{IG}_a = H(\hat p_a) - \sum_z P(z)\,\mathbb E_{\theta\mid z,\mathcal D}H(o\mid\theta)$ (closed form under Beta posteriors) | F1 | `ira.py` `_expected_hb`, `act` |
| Prior for an unobserved concept pair = mean posterior rate over the library's concept pairs (each pair once); re-clustering uses a symmetric Beta(½,½) pair prior | F11 | `ira.py` `Library.rbar`, `recluster.py` |
| `ira_noactive` = the same greedy agent with β = 0; attempt 1's variant kept as `ira_thompson` (descriptive) | F4 | `experiment.py` |
| Universe generator rejects universes in which any condition is impossible; environment v3 (8 roles: 0–4 known, 5 unseen, 6–7 broad-training-only; all signatures and codes distinct) | attempt-1 defect | `env.py` `feasible` |
| **Broad training distribution** `train_broad` for meta-RL comparators. It contains the *kinds* of variation IRA's priors admit: cue validity ρ ~ U(0.3, 1); a role switch in 25% of worlds at t ∈ {5–8, 12–15}; 1–2 entities of a training-only extra role (6 or 7) in 25% of worlds. The tested values stay held out: ρ = 0.1, the unseen role 5, 4–5 distinct roles, and a switch at t = 10. The switch *construction* is the same as in the test, so `role_switch` is only a near-held-out shift for the broad comparators (review R1-d) | F3 | `env.py`, `metarl.py` |
| **Relational meta-RL comparator** (`rel`): permutation-equivariant per-entity GRU, self-attention, symmetric pair head. The attention output feeds the heads but not the recurrent state, so each entity's memory holds only its own interactions (review; disclosed) | F5 | `metarl.py` `RelNet` |
| **Universe as the unit of inference**: 5 confirmatory universes (4–8) × 100 worlds per condition; one-sided t-test over universe means (df = 4); missing cells are an error | F6, F13 | `stats.py` |
| Probe counts, new-concept use (MAP concept is new) and library-vs-role agreement reported for every IRA run | F2 | `play.py`, `experiment.py` |
| Tests: closed-form expected entropy, IRA probes when identity is uncertain, re-clustering on records with pair outcomes, relational equivariance, strict statistics | F14 | `tests/test_u1.py` |

Not changed (accepted limitations; F3, F7): IRA's generative model is specified by the designers (latent types,
Bernoulli laws, the action→entity binding). The cue-reliability prior stays uniform on {0, ½, 1}, the surprise
threshold stays τ = 4, and the concept concentration stays α = 1. The environment family is the one co-designed
with IRA; `role_switch` stops every reacting pair (a v2 change that attempt 1 did not disclose: in v1 only the
chosen pair stopped).

## R1.2 Agents

| agent | trained on | budget |
|---|---|---|
| `ira` + `ira_noactive` (β = 0), `ira_closed`, `ira_nocuetrack`, `ira_nochange`, `ira_nolibrary` (`ira_thompson` dev only) | 300 `train` worlds (seeds 100000–100299); library re-clustered every 50 worlds; frozen for evaluation | ≈ 1–3 CPU-min training |
| `gru_narrow` | flat GRU (hidden 128, lr 1e-3), narrow `train` distribution | 20,000 A2C updates × 128 worlds = 2.56 M episodes (as in attempt 1) |
| `gru_broad` | the same, on `train_broad` | 2.56 M episodes |
| `rel_broad` | relational agent (hidden 64, lr 1e-3), on `train_broad` | 2.56 M episodes |
| `ira_beta1` (descriptive) | `ira` with β = 1 at evaluation (same library) | — |
| `entity_bayes`, `ucb`, `random` | — | — |

All meta-RL agents act greedily (attempt 1's selection) and use one training seed per universe.

**Staged execution** (decided after attempt 1's results were known; see R1.0). Stage 1 runs IRA (all variants), `gru_narrow`, the baselines and A on universes 4–8. Stage 2
(`gru_broad`, `rel_broad`) runs only if H1 holds against `gru_narrow`. H1 requires all three comparators
(intersection–union), so if it fails against `gru_narrow` it is not supported whatever stage 2 would show. Skipping
stage 2 then saves ≈ 4 CPU-hours and changes no conclusion. Hyperparameters were
not searched again; `rel` had a single configuration (time limit; disclosed).

## R1.3 Development (universe 0 of environment v3; seeds 0–199)

* IRA β ∈ {0.5, 1, 2} selected on **`iid` only**, each with its own library.
* Each meta-RL comparator is trained on universe 0. Adequacy check: dev `iid` score ≥ 0.85 × IRA's. A comparator
  that fails is trained once more with 2× the updates.
* Before the freeze, IRA and the three comparators are evaluated on all five conditions in universe 0 (200 dev
  worlds each), and the IRA ablations on 50 dev worlds each (a bug check). Everything is reported in R1-D. The
  thresholds below are fixed **now**, before any R1 comparator result exists.
* Compute: IRA costs ≈ 0.6–0.7 CPU-s per world in R1, against ≈ 0.01 for the comparators. `ira_thompson` is
  therefore run only on development worlds.

## R1.4 Hypotheses (confirmatory universes 4–8; seeds 10000–10099 per condition)

For a contrast X − Y, let $d_u$ be the mean paired difference in universe $u$. The estimate is $\bar d$ over the
five universes; $p$ comes from a one-sided t-test on $\{d_u\}$ (df = 4).

**H1 (primary).** For each comparator C ∈ {`gru_narrow`, `gru_broad`, `rel_broad`} and each shift condition c, the
contrast IRA − C passes iff:
* $\bar d \ge 0.10$;
* the Holm-adjusted p (over the 4 shift conditions, within C) is < 0.05;
* $d_u > 0$ in ≥ 4 of 5 universes.

**H1 holds against C** iff ≥ 3 of 4 conditions pass and IRA is non-inferior on `iid` ($\bar d \ge -0.05$).
**H1 is supported** iff it holds against **all three** comparators (an intersection–union test, so there is no
correction across comparators).

**Mechanisms** (IRA − ablation; Holm over M1–M5; each passes iff $\bar d \ge$ margin and Holm p < 0.05):

| | ablation | condition(s) | margin |
|---|---|---|---|
| M1 | `ira_nocuetrack` | cue_reversed | 0.05 |
| M2 | `ira_closed` | novel_role | 0.05 |
| M3 | `ira_nochange` | role_switch | 0.05 |
| M4 | `ira_noactive` (β = 0) | mean over all five | 0.03 |
| M5 | `ira_nolibrary` | iid | 0.03 |

**A (secondary).** Accumulation, as in attempt 1 (50 sequential `novel_role` worlds per universe, seeds
20000–20049; additive consolidation vs. the same library frozen; last 25 worlds), tested at the universe level with
margin 0.05.

**Descriptive (preregistered, no threshold):**
* probes per world;
* the fraction of `novel_role` worlds in which IRA's MAP assignment uses a new concept;
* adjusted Rand index of each training library against the true roles;
* `entity_bayes`, `ucb`, `random`.

## R1.5 Decision rule

* **H1 supported and M4 passes and at least one of M1–M3 passes** → *advance, narrowly*: in this co-designed
  environment family, the agent beats generic meta-learners, including a relational meta-learner trained on the
  same kinds of variation, under held-out shift values, and active identification explains part of it. Novelty
  level: a new empirical finding about a combination (see `PRIOR_ART_AND_NOVELTY.md` for the precedents —
  Alchemy, EMPA — that bound it). Not a new principle.
* **H1 supported but M4 fails** → a structured-prior advantage that is not attributable to the proposed active
  identification; no mechanism claim.
* **H1 holds against `gru_narrow` but not against `gru_broad` or `rel_broad`** → a learner trained on near-test task
  structure (or with relational structure) matches IRA. The principle-level claim is then rejected: it is not shown
  that IRA's advantage is "only priors", but it is shown that the advantage is not unique to IRA's mechanism.
* **H1 fails against `gru_narrow`** → *reject*.
* A failing → reject "accumulates new concepts without retraining".

## R1-D — Development disclosure before the R1 freeze

* **Review of R1 by the critic** (`INDEPENDENT_CRITIQUE.md`, section "Review of revision R1"). It found:
  * R1-a, a blocker: `ira_noactive` crashed because β was passed twice. The fix to `ira_cfg` had silently failed to
    apply, and the dev bug check had crashed on it too.
  * The dev protocol had not yet been run on the committed code.
  * β = 0.5 hardly probes.
  * `role_switch` was not held out for the broad comparators.
  * An undisclosed budget cut for `gru_narrow` (15k vs 20k updates).
  All are fixed or disclosed above. A test now builds every agent through the runner, and the stage-2 gate fails
  loudly if the stage-1 summary is missing.
* **Provenance of dev artefacts.** The β-selection libraries and the dev `gru_narrow`/`gru_broad`/`rel_broad` models
  were trained by a scratch copy of the R1 code. It differed from the committed code only in formatting, in the
  `ira_cfg` bug (which affects only `ira_noactive`), in 15,000 instead of 20,000 updates, and, for the broad models,
  in switch times drawn from [5, 15]. The dev models therefore serve only the adequacy check. All dev evaluations
  below were run with the committed code.
* **β selection** (dev `iid`, 200 worlds): 0.5 → 0.865, 1.0 → 0.857, 2.0 → 0.779; probes per world 0.04 / 1.42 /
  3.31. The preregistered rule picks β = 0.5. The margin over β = 1 (0.008) is below the paired standard error
  (≈ 0.014). No tie rule was preregistered, and none is introduced after seeing the numbers. Instead `ira_beta1` is
  reported descriptively.
* **Predictions written before the freeze** (from the dev numbers and the critic's runs):
  * M2 (open concept set) and M4 (information gain, β = 0.5 vs 0) are expected to **fail**. On dev, the selected
    agent barely probes and gave no new concept to any unseen-role entity.
  * An "advance" verdict is therefore nearly unreachable by construction.
  * If H1 fails against `gru_narrow`, as in attempt 1, R1 rejects the hypothesis for the agent that does implement
    information gain over laws.
* **R1 dev evaluation (universe 0, committed code).** 200 worlds per cell for IRA and the comparators; 50 for the
  ablations (bug check). Logs: `results_r1/dev/dev_eval_*.log`. The dev comparator models are scratch-trained, as
  described above.

  | agent | iid | cue_reversed | recombined | novel_role | role_switch |
  |---|---|---|---|---|---|
  | ira (β = 0.5) | 0.865 | 0.842 | 0.890 | 0.701 | 0.682 |
  | gru_narrow | 0.932 | 0.869 | 0.960 | 0.849 | 0.487 |
  | gru_broad | 0.822 | 0.818 | 0.867 | 0.830 | 0.673 |
  | rel_broad | 0.906 | 0.831 | 0.947 | 0.898 | 0.770 |
  | ira_noactive (β = 0; 50) | 0.866 | 0.842 | 0.884 | 0.737 | 0.699 |
  | ira_beta1 (50) | 0.813 | 0.710 | 0.804 | 0.669 | 0.645 |
  | ira_thompson (50) | 0.593 | 0.552 | 0.635 | 0.526 | 0.420 |
  | ira_closed (50) | 0.878 | 0.808 | 0.871 | 0.719 | 0.694 |
  | ira_nocuetrack (50) | 0.873 | 0.775 | 0.839 | 0.708 | 0.665 |
  | ira_nochange (50) | 0.878 | 0.858 | 0.905 | 0.732 | 0.478 |
  | ira_nolibrary (50) | 0.800 | 0.782 | 0.808 | 0.785 | 0.452 |
  | entity_bayes (50) | 0.627 | 0.593 | 0.592 | 0.596 | 0.369 |
  | ucb (50) | 0.480 | 0.455 | 0.501 | 0.494 | 0.260 |
  | random (50) | 0.230 | 0.259 | 0.207 | 0.222 | 0.260 |

  Adequacy (≥ 0.85 × IRA's dev `iid` 0.865 = 0.735) holds for every comparator.

  **Predictions written before the freeze:**
  * H1 is expected to **fail** against `gru_narrow`: on dev it is behind IRA only on `role_switch`. Stage 2 is then
    skipped.
  * Had stage 2 run, the dev numbers suggest that `rel_broad` would exceed IRA in 4 of 5 conditions, including
    `role_switch` (0.770 vs 0.682). On dev it trails IRA only on `cue_reversed` (0.831 vs 0.842).
  * On `novel_role`, IRA with its learned library does worse than without it (0.701 vs 0.785).
