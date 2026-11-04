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
* Adaptive models are also scored in the `frozen` variant (person-level state = `init_state()` throughout;
  recurrent state and any within-model learning still run; diagnostic only — Gate B1 uses a separately trained
  no-adaptation twin) and the `mismatched` variant (the harness updates the person-level state from another tune
  participant's first `min(j·m, T_donor)` trials — a deterministic derangement, length-matched to the own schedule).
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
| `pop_gru` | 1-layer GRU over v3, linear head, masked log-softmax; core trainer (Adam, batch 8, ≤ 150 epochs, plateau lr halving, patience 10, early stopping on the dev split), input dropout | hidden {16, 32, 64, 128} × lr {0.003, 0.01} × dropout {0, 0.2, 0.4} (24) | core |
| `pop_lstm` | same with an LSTM cell | same (24) | core |
| `pop_tf` | causal transformer: linear projection to d = 32, 2 pre-LN blocks of 4-head self-attention + 2d GELU MLP, final LN + linear head. Attention restricted to a sliding causal window `t − W + 1 … t` (band mask) with a learned relative-position bias per head for offsets 0…W−1 (no absolute positions). Same trainer | W {32, 128} × lr {0.001, 0.003} × dropout {0, 0.1} (8) | baselines (`baselines/tf.py`) |
| `ref_gru_p2` | the Phase II numpy GRU (`hpe.evaluation.human.gru`, manual backprop, truncated BPTT window 64, Adam, early stopping on an 80/20 split drawn with `default_rng(seed)` as in Phase II) refitted on Phase III dev, input rebuilt from v3 exactly as Phase II built it: features (12), previous action one-hot (K_family), previous chosen reward, previous correct action one-hot (= argmax of the previous outcomes when every option's outcome was shown; zero otherwise), previous-feedback flag | Phase II grid: hidden {8, 12} × lr {0.003, 0.01}, max_epochs 15, patience 4, ≤ 60,000 steps per epoch (4) | baselines (`baselines/ref_gru_p2.py`) |

### 2.2 Conventional adapters on the population GRU backbone (`baselines/adapters.py`)

**Harness-owned adaptation (critic finding G-1).** Every adaptive comparator (the six adapters below and
`hb_cog`) subclasses `hpe.population.core.adaptation.AdaptiveBase` and does not override `predict_logprobs`. It
defines `update_every = m`, `init_state()` (the population prior), `update_state(state, history)` and
`predict_with_states(seq, states)`. The core harness computes the person-level state at rows `m, 2m, …` from a
truncated, re-keyed copy of the first `j·m` trials (own trials in `adapt`; the donor's first `min(j·m, T_donor)`
trials in the length-matched `donor` control; `init_state()` throughout in `frozen`). Rows `[j·m, (j+1)·m)` are
predicted with that state while the recurrent state runs over `x_{0..t}`; row t therefore depends only on
`x_{0..t}` and `a_{0..t−1}`. Each state update restarts from the population (or meta-learned) parameters, so no
drift accumulates (Bayesian-head Newton iterations are warm-started from the previous MAP, which does not move the
optimum). Predicting never mutates the model (unit-tested).

**Backbone.** Trained inside `fit` exactly like `pop_gru` (`baselines/backbone.py` replicates
`PopRNNPredictor.fit`: same `setup_torch` key, module, trainer call and defaults). Its configuration and seed are
model hyper-parameters (`backbone`, `backbone_seed`). In the Gate C pool runs they are set to the **tune-selected
`pop_gru` of the same family and the seed the runner gave that configuration**, so every adapter's `[frozen]`
variant reproduces that `pop_gru` (unit test `test_adapter_frozen_variant_equals_pop_gru`; exact for the
head-level adapters, whose frozen state is the untouched backbone). For Gate B2 the backbone is the candidate's
(§3). The backbone is trained once per family and process and cached; its CPU cost is logged in each adapter's
fit log (`backbone_cpu_s`) and counted once. `twin_config()` returns the backbone configuration.

| Model | Person-level state and its update from the history `D` (scored trials `S`) | Grid | Per-person parameters |
|---|---|---|---|
| `ft_head` | `(W, b)` after `n` Adam steps (lr) on `mean_{s∈S} −log softmax(W h_s + b)[a_s] (+ l2/2‖(W,b) − (W_0,b_0)‖²)` from `(W_0, b_0)` | m {10, 25} × n {5, 20} × lr {0.003, 0.01, 0.03} (12) | K·(H+1) |
| `ft_full` | all GRU and head weights θ after `n` Adam steps on the same loss by BPTT over `D`, gradient clip 1, penalty `l2/2 ‖θ − θ_pop‖²` (dynamic evaluation) | n {3, 10} × lr {0.001, 0.003} × l2 {0, 1}; m fixed per family (8) | all backbone weights |
| `bayes_head` | prior `vec(W̃) ~ N(vec(W̃_pop), σ² I)` on `W̃ = [W | b]`; state = MAP `W̃* = argmin Σ_{s∈S} −log softmax(W̃ φ_s)[a_s] + ‖W̃ − W̃_pop‖²/(2σ²)` (φ = (h, 1); damped Newton) and the Hessian H there; `predictive=laplace`: `Σ = H⁻¹` and `p ∝ softmax_k(μ_k / sqrt(1 + π v_k/8))`, `v_k = φᵀ Σ_kk φ`. `prior_var=eb`: σ² = argmax over {0.003 … 3} of the summed Laplace evidence `−f(W̃*) − (KD/2) log σ² − ½ log|H|` of the backbone's held-out validation participants (dev) — an empirical-Bayes population prior | σ² {eb, 0.01, 0.03, 0.1, 0.3, 1} × m {10, 25} × predictive {map, laplace} (24) | K·(H+1) |
| `exemplar_mix` | the memory `{(h_s, a_s, s) : s ∈ S}`; prediction `p_t = (1 − w) p_pop,t + w p_kern,t`, `p_kern,t(a) = (Σ k_ts 1[a_s = a] + 1/K) / (Σ k_ts + 1)`, `k_ts = exp(−‖h_t − h_s‖² / 2ℓ²) γ^{t−1−s}`, ℓ² = bw × median within-person squared hidden-state distance on dev | w {0.05, 0.1, 0.2, 0.4} × bw {0.25, 1, 4} × γ {1, 0.98}; m = 5 (24) | non-parametric memory |
| `reptile_gru` | meta-training (Reptile, Nichol et al. 2018, first order) from the backbone: 300 outer iterations × 4 tasks; task = one dev participant (fit split); inner loop = `n` Adam steps (lr) on growing prefixes `[0, T·j/n)`, j = 1…n; `θ ← θ + ε (φ̄ − θ)`, ε = 0.5 annealed linearly to 0. State: `ft_full` (no L2) from the meta-learned weights | lr {0.001, 0.003} × n {3, 10}; m fixed per family (4) | all weights |
| `cavia_gru` | CAVIA (Zintgraf et al., 2019): context `c ∈ R^d` concatenated to the GRU input at every step and to the head input; shared weights initialised from the backbone, context weights ~ N(0, 0.1²) (so `c = 0` reproduces the backbone); meta-trained with the core trainer on dev: per participant a random split u, inner loop `c ← c − α ∇_c mean NLL(scored trials < u)` from `c = 0` (`n_inner` = 2 steps, fixed α), outer loss = NLL of the scored trials ≥ u, backpropagated through the inner loop (second order); early stopping on the backbone's validation participants with u = T/2. State: `c` after the same inner loop on `D` | c_dim {4, 8} × α {0.3, 1.0}; meta lr 0.003, ≤ 40 epochs, patience 5; m fixed per family (4) | d |

### 2.3 Hierarchical Bayesian cognitive model `hb_cog` (`baselines/hb_cog.py`)

Per particle θ = (α, ρ, β, κ, δ[, w]): option values `Q` (start 0.5); after each step every option whose outcome
was observed (v3 observed-outcome mask) is updated `Q ← Q + r (o − Q)` with rate α (chosen) or ρα
(counterfactual); a choice without feedback is credited with the next feedback step's chosen outcome (one-step
eligibility, e.g. the spaceship of a two-stage day); unobserved values decay `Q ← Q + δ (0.5 − Q)`; optional reset
at block starts. Choice: `softmax_a(β V[key_a] + κ 1[a = last choice in this state])`, `V = Q` or, with the
model-based hybrid (`mb: true`, two_step), `V = w Σ_s' p̂(s'|s,a) max_a' Q[s', a'] + (1 − w) Q[s, a]` with
transition counts p̂ learned from the person's own sequence. Keys: state = selected context features
(`state_features`) plus the action index, or per-option features (`option_features`) so a value follows its option
into new pairings. Particles: α {.05, .15, .3, .5, .8} × ρ {0, .5, 1} × β {1, 3, 6, 10, 20} × κ {−.5, 0, .5, 1.5}
× δ {0, .1} (= 600) × w {0, .5, 1} with `mb`. Population prior: mixing weights π over particles by EM on dev
(NPMLE with `c` pseudo-counts per particle). Person-level state (harness): posterior log-weights
`log π + τ Σ_{s∈S} log p_θ(a_s)` recomputed every m = 5 trials; prediction = posterior-predictive mixture; each
particle's value learning runs in context. Grid: c {0.01, 0.1, 1} × τ {0.5, 1} (6). Family settings: restless4
none; garcia options = letter codes (features 0–3, 4–7); described options = described gambles (features 0–5,
6–11); two_step state = (stage, start pair, planet) (features 0, 2, 3) with `mb: true`.

### 2.4 Per-person Phase II models (reference only)

The Phase II per-person models (PVL-Delta, exemplar, developmental, hybrid exemplar) are reported in Phase II
(`docs/phase2/HUMAN_VALIDATION_V2.md`, X-009/X-010) on the Phase II partitions; they were not rerun here.

## 3. Gate B2 comparator rule (fixed 2026-11-04, before any B2 tune number exists)

Answering critic finding G-2 (and the director's instruction of 2026-11-04):

1. **Backbone.** Every B2 adapter uses the **same backbone configuration as the candidate**: the candidate's
   population network with its adaptation pathway removed, as returned by the candidate's
   `AdaptiveBase.twin_config()` (a `pop_gru` configuration: hidden size, learning rate, dropout, trainer settings),
   passed to the adapters as `backbone`, trained inside each adapter's `fit` exactly like `pop_gru`. The
   tune-selected `pop_gru` of the family (the Gate C pool backbone) is reported as a second reference.
2. **Adapter set** (closed list): `ft_head`, `ft_full`, `bayes_head`, `exemplar_mix`, `reptile_gru`, `cavia_gru`.
   `cavia_gru`'s grid includes the candidate's latent dimension as a `c_dim` value.
3. **Tuning.** Each adapter is tuned on **tune** under the same caps as the candidate: ≤ 24 configurations,
   ≤ 2700 CPU-s of fitting per adapter and family (the shared backbone's cost counted once and reported), the
   runner's lapse grid, harness-owned adaptation modes (`AdaptiveBase`), `update_every` as in its grid or fixed per
   family below.
4. **Selection.** Per counted family, the B2 comparator is the adapter with the lowest tune NLL (main `adapt`
   variant) among the six; it is fixed at the freeze and named in the freeze record with its configuration and
   lapse.
5. **Contrast.** Candidate (adapt) − selected adapter (adapt), per participant, on the frozen test run; margin and
   Holm correction as in thesis §6 (B2: ≥ 0.005 nats/choice in ≥ 2 counted families).
6. **Compute.** Per-participant prediction CPU (test-time adaptation) is reported for the candidate and every
   adapter (critic F-4); fitting caps as above.

Per-family update intervals (fixed before fitting; m in trials): `ft_head`/`bayes_head` grid m ∈ {10, 25}
(described {25, 75}); `ft_full`/`reptile_gru`/`cavia_gru` m = 25 (restless4, garcia), 50 (two_step), 75 (described);
`exemplar_mix` m = 5 (described 15); `hb_cog` m = 5 (described 25).

## 4. Results on tune

_Filled from committed result directories (see §5 for the paths)._

## 5. Result directories

## 6. Best comparator per family (tune)

## 7. Failures, deviations and known gaps

## 8. Compute notes
