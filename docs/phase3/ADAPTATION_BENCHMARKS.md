# Phase III — Adaptation benchmarks: the comparator pool (evaluation team, lane `science/validation-v3`)

_Written 2026-11-04. Exploratory: every result below is on the **tune** partition, which is also the selection
partition (selection-optimistic). No gate is evaluated here. Every number is copied from a committed result
directory named next to it._

## 1. Common protocol

* Data: `hpe.population.core.data.load_family` (partition lock before parsing); encoding v3 (`k_max = 4`).
* Fit on **dev** (internal early-stopping split = 20 % of dev participants, drawn by `nets.split_train_val`);
  select configuration and lapse λ ∈ {0, .01, .02, .05, .1, .2, .3} on **tune**; report on **tune**.
* Metric: per participant, mean NLL (nats) of free choices after lapse mixing; family score = mean over
  participants with a 95 % percentile bootstrap CI (2,000 resamples). Paired contrasts per participant.
* Caps (thesis §9): ≤ 24 configurations per model and family, ≤ 2700 CPU-s of fitting per model and family
  (`budget_s_per_model_family`); configurations past the budget are recorded as skipped. Torch: 2 threads; one
  experiment process at a time; numpy BLAS pinned to 1 thread (see §7).
* Causality: every selected model passes `core.conformance.check_causal` (adapt and frozen modes); the runner
  records violations and the pool aggregator excludes any model with one.
* Adaptive models are also scored in the `frozen` variant (no individual adaptation; recurrent state and any
  within-model learning still run) and the `mismatched` variant (individual parameters learned on another
  tune participant's whole sequence — a deterministic derangement — and held fixed).
* Seeds: every baseline experiment uses seed 20261108; per-configuration seeds come from the runner
  (`derive_seed(seed, family, model, config_index)`).
* No model calls a language model; no outcome is hard-coded; no family-specific outcome rule exists in any
  model (family-specific settings are limited to which context features identify a state/option for `hb_cog`
  and the adaptation interval `m`, both fixed in the YAMLs before fitting).

## 2. Pool members: definitions, maths, parameters and compute

Notation: `x_t` = encoding-v3 input at step t (known before choice t), `a_t` = choice, `S_t` = scored (free) steps
before t, `K` = available actions, `h_t` = GRU hidden state after reading `x_{0..t}`.

### 2.1 Population sequence models (no explicit per-person parameters; adaptation only in context)

| Model | Definition | Grid (configurations) | Owner |
|---|---|---|---|
| `pop_gru` | 1-layer GRU over v3, linear head, masked log-softmax; core trainer (Adam, batch 8, ≤ 150 epochs, plateau lr halving, patience 10, early stopping on the dev split), input dropout | hidden {16, 32, 64} × lr {0.003, 0.01} × dropout {0, 0.2} (12) | core |
| `pop_lstm` | same with an LSTM cell | same (12) | core |
| `pop_tf` | causal transformer: linear projection to d = 32, 2 pre-LN blocks of 4-head self-attention + 2d GELU MLP, final LN + linear head. Attention restricted to a sliding causal window `t − W + 1 … t` (band mask) with a learned relative-position bias per head for offsets 0…W−1 (no absolute positions). Same trainer | W {32, 128} × lr {0.001, 0.003} × dropout {0, 0.1} (8) | baselines (`baselines/tf.py`) |
| `ref_gru_p2` | the Phase II numpy GRU (`hpe.evaluation.human.gru`, manual backprop, truncated BPTT window 64, Adam, early stopping on an 80/20 split drawn with `default_rng(seed)` as in Phase II) refitted on Phase III dev, input rebuilt from v3 exactly as Phase II built it: features (12), previous action one-hot (K_family), previous chosen reward, previous correct action one-hot (= argmax of the previous outcomes when every option's outcome was shown; zero otherwise), previous-feedback flag | Phase II grid: hidden {8, 12} × lr {0.003, 0.01}, max_epochs 15, patience 4 (4) | baselines (`baselines/ref_gru_p2.py`) |

### 2.2 Conventional adapters on the population GRU backbone (`baselines/adapters.py`)

Backbone: trained inside `fit` exactly like `pop_gru` (`baselines/backbone.py` replicates
`PopRNNPredictor.fit`: same `setup_torch` key, module, trainer call and defaults). Its configuration and seed are
fixed per family to the **tune-selected `pop_gru` of X3-B01 and the seed the runner gave that configuration**, so
every adapter's `[frozen]` variant reproduces the selected `pop_gru` bit-for-bit (unit test
`test_adapter_frozen_variant_equals_pop_gru`). The backbone is trained once per family and process and cached;
its CPU cost is logged in each adapter's fit log (`backbone_cpu_s`) and counted once.

Online protocol shared by all adapters: individual parameters are refitted at update points `u = m, 2m, …`
using only scored steps `< u`, starting each time from the population (or meta-learned) parameters; rows
`u … u+m−1` are predicted with them while the recurrent state keeps running. Row t therefore depends only on
`x_{0..t}` and `a_{0..t−1}`.

| Model | Individual adaptation (maths) | Grid | Per-person parameters |
|---|---|---|---|
| `ft_head` | `W, b ← argmin` by `n` Adam steps (lr) of `mean_{s∈S_u} −log softmax(W h_s + b)[a_s] (+ l2/2‖(W,b) − (W_0,b_0)‖²)`, start `(W_0, b_0)` | m {10, 25} × n {5, 20} × lr {0.003, 0.01, 0.03} (12) | K·(H+1) |
| `ft_full` | the same on all GRU and head weights θ, BPTT over the prefix, gradient clip 1, penalty `l2/2 ‖θ − θ_pop‖²` | n {3, 10} × lr {0.001, 0.003} × l2 {0, 1}, m = 25 (restless4) (8) | all backbone weights |
| `bayes_head` | prior `vec(W̃) ~ N(vec(W̃_pop), σ² I)` on `W̃ = [W | b]`; MAP `W̃_u = argmin Σ_{s∈S_u} −log softmax(W̃ φ_s)[a_s] + ‖W̃ − W̃_pop‖²/(2σ²)`, φ = (h, 1), by damped Newton (warm-started from the previous update); `predictive=laplace`: posterior covariance `Σ = H⁻¹` (H = Hessian at the MAP) and the mean-field correction `p ∝ softmax_k(μ_k / sqrt(1 + π v_k/8))`, `v_k = φᵀ Σ_kk φ`. `prior_var=eb`: σ² = argmax over {0.003 … 3} of the summed Laplace evidence `−f(W̃*) − (KD/2) log σ² − ½ log|H|` of the backbone's held-out validation participants (dev) — an empirical-Bayes population prior | σ² {eb, 0.01, 0.03, 0.1, 0.3, 1} × m {10, 25} × predictive {map, laplace} (24) | K·(H+1) |
| `exemplar_mix` | `p_t = (1 − w) p_pop,t + w p_kern,t`, `p_kern,t(a) = (Σ_{s∈S_t} k_ts 1[a_s = a] + 1/K) / (Σ_{s∈S_t} k_ts + 1)`, `k_ts = exp(−‖h_t − h_s‖² / 2ℓ²) γ^{t−1−s}`, ℓ² = bw × median within-person squared hidden-state distance on dev | w {0.05, 0.1, 0.2, 0.4} × bw {0.25, 1, 4} × γ {1, 0.98} (24) | non-parametric memory (the person's (h_s, a_s) pairs) |
| `reptile_gru` | meta-training (Reptile, Nichol et al. 2018, first order) from the population backbone: 300 outer iterations × 4 tasks; task = one dev participant (fit split); inner loop = `n` Adam steps (lr) on growing prefixes `[0, T·j/n)`, j = 1…n (the online regime); `θ ← θ + ε (φ̄ − θ)`, ε = 0.5 annealed linearly to 0. Test: `ft_full` from the meta-learned weights, no L2 | lr {0.001, 0.003} × n {3, 10}, m = 25 (4) | all weights |

`mode="frozen"` returns the backbone (meta-learned weights for Reptile); `donor=` fits the individual parameters on
the donor's entire sequence (one fit, `u = T_donor`) and holds them fixed.

### 2.3 Hierarchical Bayesian cognitive model `hb_cog` (`baselines/hb_cog.py`)

Per particle θ = (α, ρ, β, κ, δ): option values `Q` (start 0.5); after each step every option whose outcome was
observed (v3 observed-outcome mask) is updated `Q ← Q + r (o − Q)` with rate α (chosen) or ρα (counterfactual); a
choice without feedback is credited with the next feedback step's chosen outcome (one-step eligibility, e.g. the
spaceship of a two-stage day); unobserved values decay `Q ← Q + δ (0.5 − Q)`; optional reset at block starts.
Choice: `softmax_a(β Q[key_a] + κ 1[a = last choice in this state])`. Keys: state = selected context features
(`state_features`) plus the action index, or per-option features (`option_features`) so a value follows its option
into new pairings. Particles: α {.05, .15, .3, .5, .8} × ρ {0, .5, 1} × β {1, 3, 6, 10, 20} × κ {−.5, 0, .5, 1.5}
× δ {0, .1} = 600. Population prior: mixing weights π over particles by EM on dev (NPMLE with `c` pseudo-counts per
particle). Person: `w_t(θ) ∝ π(θ) Π_{s∈S_t} p_θ(a_s)^τ`; prediction = posterior-predictive mixture. Grid:
c {0.01, 0.1, 1} × τ {0.5, 1} (6). Family settings: restless4 none; garcia options = letter codes (features 0–3,
4–7); described options = described gambles (features 0–5, 6–11); two_step state = (stage, start pair, planet)
(features 0, 2, 3). No model-based component (a hybrid MB/MF variant for two_step was not built; §6).

### 2.4 Per-person Phase II models (reference only)

The Phase II per-person models (PVL-Delta, exemplar, developmental, hybrid exemplar) are reported in Phase II
(`docs/phase2/HUMAN_VALIDATION_V2.md`, X-009/X-010) on the Phase II partitions; they were not rerun here.

## 3. Results on tune

_Filled from committed result directories (see §4 for the paths)._

## 4. Result directories

## 5. Best comparator per family (tune)

## 6. Failures, deviations and known gaps

## 7. Compute notes
