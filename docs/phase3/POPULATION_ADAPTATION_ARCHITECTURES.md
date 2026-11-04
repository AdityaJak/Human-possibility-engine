# Population Adaptation Architectures (Phase III design team)

_Lane `research/adaptation`. Everything here is **exploratory** (dev fit, tune select and report; synthetic
toys). No gate is evaluated in this document. Results are copied from committed artifacts under
`reports/phase3/candidates/`; numbers are not rounded in our favour. A name is not novelty: for every
mechanism we name the existing method it most resembles and the baseline that would make it redundant._

## 0. Common ground

**The bar.** A recurrent population model already adapts to each person *in context*: its hidden state is
updated by every choice and outcome (Phase II X-009/X-010; thesis §1). A candidate adds value only if it
beats (i) the same population network with its recurrent state running and the explicit adaptation
switched off, (ii) a separately trained same-size `pop_gru` twin (Gate B1 comparator after critic G-1),
(iii) conventional adapters on the same backbone (Gate B2; evaluation lane), and (iv) the strongest
tune-selected comparator (Gate C).

**Interface (harness-owned modes).** Every candidate subclasses `hpe.population.core.adaptation.AdaptiveBase`
and implements only `init_state` (the prior person state), `update_state(state, history)` (history = a
truncated, re-keyed copy of the first t trials), `predict_with_states(seq, states)` and `twin_config`.
The harness builds the adapt schedule from the person's own prefixes every `update_every` trials, the
frozen schedule (`init_state` throughout) and the length-matched donor schedule (Gate A2 control). The
candidates do not implement `predict_logprobs`; `check_causal` passes in every mode (unit tests).

**Equal treatment.** Every population network here (`pop_gru`, the F1/F3 backbone, the F2 world model,
the F3 GRU challenger) is trained by `core.nets.train` with the core defaults (Adam, batch 8, ≤ 150
epochs, patience 10, plateau lr halving, early stopping on a seeded 20 % participant split of dev). The
F1/F3 backbone *is* a `pop_gru` (same class `nets.PopRNN`, same trainer, same config keys); it is
trained once per (training data, config) inside a process and shared by the F1/F3 variants so that their
ablations are exactly paired. Its CPU cost is attributed to every model that uses it.

**Code.** `packages/hpe/population/candidates/` — `common.py` (shared backbone, trainer settings, twin
config), `f1_latent.py`, `f3_arbiter.py`, `f2_world_policy.py`, `toy.py` (synthetic populations),
`analysis.py` (paired ablation contrasts from the runner's parts). Tests: `tests/population/candidates/`.
Experiment definitions: `experiments/phase3/candidates/`. Results: `reports/phase3/candidates/`.

## 1. F1 `f1_latent` — population GRU + explicit individual latent, inferred online

**(1) Definition.** Shared weights θ = (backbone GRU, read-out W, b) + latent pathway (z columns of the
GRU input and read-out, bilinear modulation V). Per-person latent z = [z_trait ∈ ℝ^{d_tr}, z_ctx ∈ ℝ^{d_ctx}]
(default 2 + 2):

    h_t = GRU([x_t, z_trait], h_{t−1}),    logits_t = W[h_t; z] + b + Σ_j z_j V_j h_t

z_trait enters the recurrence and the read-out; z_ctx enters only the read-out, so a context change does
not rewrite the recurrent history. Person state = z. Every m trials (harness schedule) z is re-inferred
from the prior mean using only the person's first t trials, by k = 2 block Gauss–Newton (Fisher-scoring)
steps on

    J_trait = Σ_{s<t} ℓ_s(z) + |z_trait|²/(2σ_tr²)                       (σ_tr = 0.5, all past trials)
    J_ctx   = γ_t Σ_{s<t} ρ^{t−1−s} ℓ_s(z) + |z_ctx|²/(2σ_ctx²)           (σ_ctx = 1, half-life 20 trials)
    z_b ← z_b − (Σ_s w_s J_sᵀ(diag p_s − p_s p_sᵀ)J_s + σ_b⁻² I)⁻¹ (Σ_s w_s J_sᵀ(p_s − y_s) + σ_b⁻² z_b)

J_s = ∂logits_s/∂z_b by forward differences. **Proposed mechanism (surprise gating):**
γ_t = clip(exp(β S_t), ¼, 4), S_t = mean over the last 20 free choices of ℓ_s(0) − H[p_s(0)], the person's
excess NLL over what the population prediction (z = 0) expects of itself. γ scales the context block's
evidence relative to its prior (its effective step size): a person who suddenly behaves unlike the
population's expectation moves z_ctx faster. Prediction for rows [t, t+m) re-runs the GRU from trial 0
with the latent inferred at t.

Training: stage 1 = the population backbone (a `pop_gru`, z = 0). Stage 2 trains only the latent pathway
(backbone frozen, so z = 0 reproduces the backbone exactly). Default `train_mode="joint"`: joint MAP
("auto-decoder", empirical Bayes) with a free code per fitting participant — z_trait,i and one z_ctx,i,j per
block of m trials — under the same Gaussian priors, early-stopped on the **online** adapt-mode NLL of
held-out dev participants. `train_mode="meta"`: CAVIA-style first-order meta-training (sample cuts c, infer
z on [0, c), outer loss on [c, c+m)).

**(2) Closest existing methods.** CAVIA (Zintgraf et al., 2019, *Fast context adaptation via
meta-learning*, ICML): context parameters fed to the network and adapted by gradient steps while shared
weights are meta-learned; MAML (Finn et al., 2017, ICML) and Reptile (Nichol et al., 2018, arXiv
1803.02999) for the outer loop; auto-decoder / per-instance latent codes (Park et al., 2019, *DeepSDF*,
CVPR); individual latent embeddings for behavioural RNNs (Dezfouli et al., 2019, *Disentangled behavioural
representations*, NeurIPS); hierarchical Bayesian models of individual differences with a population
prior (e.g. Lee & Wagenmakers, 2013, *Bayesian Cognitive Modeling*); Laplace / Fisher-scoring posterior-mode
updates. Surprise-scaled learning rates: Pearce & Hall (1980), Behrens et al. (2007, *Nat. Neurosci.*),
Nassar et al. (2010, *J. Neurosci.*), Bayesian online change-point detection (Adams & MacKay, 2007, arXiv
0710.3742). The trait/context split resembles two-timescale latent models (slow trait, fast state).

**(3) Hypothesised mechanism.** The recurrent state can only adapt through what the population network
learned to integrate in context; an explicit low-dimensional latent fitted by (approximate) Bayesian
inference to the person's whole history captures stable individual differences (trait) more data-
efficiently, and a surprise-gated context latent tracks policy changes faster than the trait.

**(4) Challenging baselines.** The same backbone with z = 0 (frozen); a separately trained same-size
`pop_gru` twin (B1); `f1_cavia` — the standard single-latent fixed-step CAVIA adapter (critic G-2 asked
for it); the evaluation lane's adapters on the same backbone (head / full fine-tuning, Bayesian head,
exemplar mixture, Reptile; B2).

**(5) Implementation.** `packages/hpe/population/candidates/f1_latent.py` (`F1Latent`, `LatentGRU`,
`_infer_gn`, `_infer_gd`, `F1Latent._fit_joint`); ablations registered in `candidates/__init__.py`:
`f1_trait` (one latent of the same total size, all-trial prior, recurrence + read-out), `f1_context`
(one context latent of the same total size, read-out only, surprise-gated), `f1_fixedstep` (β = 0),
`f1_cavia` (single latent, k fixed-step gradient steps on the mean past NLL, meta-trained), `f1_bolt`
(latent pathway left at its random initialisation; inference only).

## 2. F3 `f3_arbiter` — episodic memory over population states with a learned reliability gate

**(1) Definition.** Backbone `pop_gru` gives p_pop,t and hidden state h_t. Person state = episodic memory
M_t = {(h_s, a_s, s): s < t free choice} plus reliability statistics. Retrieval:
w_s = exp(−|h_t − h_s|²/(2Hb²))·ρ^{t−1−s}, p_epi,t(a) = (Σ_s w_s 1[a_s = a] + ε/K_t)/(Σ_s w_s + ε). Gate
g_t = σ(wᵀ r̃_t + w_0) with r_t = [EW mean log p_pop(a_s), EW mean log p_epi(a_s) (each scored with the
memory it had before s, decay λ = 0.9), log(1 + Σ_s w_s), H[p_pop,t]/log K_t, log(1 + |M_t|)], standardised;
p_t = g_t p_epi,t + (1 − g_t) p_pop,t; with an empty memory p_t = p_pop,t. The six gate weights are fitted by
maximum likelihood on the backbone's held-out early-stopping participants (dev), never on the people the
backbone was trained on. Memory is updated after every trial (`update_every = 1`).

**(2) Closest existing methods.** Continuous cache / kNN language models (Grave et al., 2017, ICLR;
Khandelwal et al., 2020, *kNN-LM*, ICLR) — the closest: hidden-state-keyed memory interpolated with the
parametric prediction (fixed λ); episodic control (Lengyel & Dayan, 2007, NeurIPS; Blundell et al., 2016,
arXiv 1606.04460; Pritzel et al., 2017, ICML); episodic contributions to human choice (Gershman & Daw,
2017, *Annu. Rev. Psychol.*); reliability-based arbitration between controllers (Daw, Niv & Dayan, 2005,
*Nat. Neurosci.*; Lee, Shimojo & O'Doherty, 2014, *Neuron*); mixture of experts with a gating network
(Jacobs et al., 1991, *Neural Comput.*); tracking the best expert / fixed share (Herbster & Warmuth, 1998,
*Mach. Learn.*).

**(3) Hypothesised mechanism.** A person's idiosyncratic state→action mapping (what *this* person did in
internal states like the current one) is captured non-parametrically and immediately; the gate trusts it
when it has recently predicted this person better than the population model and when retrieval support is
high.

**(4) Challenging baselines.** `f3_fixedmix` (constant g fitted on the same split — kNN-LM style),
`f3_fixedshare` (Bayesian fixed-share mixture of the two experts, weights in the person state),
`f3_gruepi` (a second GRU initialised from the backbone that receives [p_epi, log(1 + mass)] as input and
is trained on dev — learned arbitration by recurrence with the same information), the frozen backbone,
the `pop_gru` twin, and the evaluation lane's exemplar-memory mixture adapter.

**(5) Implementation.** `packages/hpe/population/candidates/f3_arbiter.py` (`F3Arbiter`,
`_kernel_probs`, `_fit_gate`; incremental `update_state` continues the backbone GRU from the stored hidden
state).

## 3. F2 `f2_world_policy` — population world model + individual policy posterior

**(1) Definition.** World model (population; trained on observed outcomes only, never on choices): a GRU
over encoding v3 with the choice slices (previous action, instructed action and flag) zeroed, outputting
for every option a Gaussian (μ_k,t, σ_k,t) over the outcome of trial t; loss = Gaussian NLL on the outcomes
revealed after t (every option under full feedback, the chosen one under bandit feedback). This is a
predictive model of outcome dynamics, **not** a causal model of the task. Policy:
logits_k,t = θᵀ f_k,t with f = [μ − μ̄, σ − σ̄, 1[prev = k], choice trace (decay 0.7), option bias (K−1)]
(inverse temperature, uncertainty bonus, perseveration, choice kernel, side bias). Population prior
θ ~ N(m₀, S₀): m₀ = pooled MAP on dev; S₀ = c·PSD(Cov(per-person MAP) − mean per-person posterior
covariance) + 10⁻⁴I (method-of-moments empirical Bayes, scale c on the grid). Person state = Gaussian
posterior N(θ_t, P_t⁻¹), updated after every free choice by one online Laplace (extended-Kalman) Newton
step. Ablation `f2_delta`: μ from a delta rule per option (rate α, reset at block start),
σ = (1 + decayed observation count)^{−½}. The population-only policy (θ = m₀) is the frozen variant.

**(2) Closest existing methods.** Hierarchical Gaussian Filter's split into a perceptual (belief) model
and a response model with individual parameters (Mathys et al., 2011, *Front. Hum. Neurosci.*); Kalman-
filter learners with uncertainty bonuses in restless bandits (Daw et al., 2006, *Nature*; Speekenbrink &
Konstantinidis, 2015, *Top. Cogn. Sci.*); hierarchical Bayesian cognitive models with a population prior
(e.g. Piray et al., 2019, *PLoS Comput. Biol.*); online Laplace / assumed-density filtering for GLMs.

**(3) Hypothesised mechanism.** Population data are best used to learn *the world* (how outcomes evolve),
which is common to everyone, while individuals differ in a few interpretable policy parameters that can be
estimated online with a well-calibrated population prior.

**(4) Challenging baselines.** The population-only policy (frozen), the delta-rule world model, the
`pop_gru` twin, and the evaluation lane's hierarchical Bayesian cognitive model `hb_cog`. Known structural
limit: the world model predicts the *objective* outcome process, so individual differences in learning
(e.g. learning rate) can only be absorbed through the policy weights.

**(5) Implementation.** `packages/hpe/population/candidates/f2_world_policy.py` (`F2WorldPolicy`,
`WorldGRU`, `world_loss`, `pooled_fit`, `laplace_step`).

**(6–8) Results, cost and verdicts for F1, F2 and F3** are in §4 below.

## 4. Results, ablations, cost and verdicts (exploratory, tune partitions; filled by the director)

The design lane stopped on the account's API spend limit after its restless4 run was partly done; the
director resumed the lane's own definitions unchanged (`experiments/phase3/candidates/*.yaml`) and committed the
results. All numbers are tune NLL per free choice (nats), lower is better; every candidate passed the strict
causality check. "twin" = a separately trained `pop_gru` of the candidate's backbone configuration (one config,
one seed — biased in the candidates' favour, critic N-3); "frozen" = the harness's prior-state mode; "donor" =
the harness's length-matched mismatched-person control.

### 4.1 Toy verification (does adaptation help only when individuals differ?)

From the lane's scratch run (`toy_verification.py`; synthetic populations, 80 people × 60 trials): on the
heterogeneous cue toy, F1 adapt 0.655 vs frozen 0.682 vs donor 0.706, and the **trait-only ablation 0.620**
(better than the full trait/context split); on homogeneous toys every F1 variant ≈ frozen (correct null). The
committed slow-tier test `test_toy_adaptation_helps_only_when_individuals_differ` **fails for F1, F2 and F3**
in the director's and the critic's environments (adapt − frozen: F1 +0.0014, F3 −0.0061, F2 −0.0124 ± 0.0068
against a −0.01 threshold; critic N-13). **No mechanism claim is made while the positive control is unreliable.**

### 4.2 Human families

| family (tune n) | F3 arbiter | F1 latent | F2 world+policy | same-size `pop_gru` | strongest comparator |
|---|---|---|---|---|---|
| restless4 (174) | **0.7237** (frozen 0.7302, donor 0.7314, twin 0.7387) | 0.7280 (= bolt-on 0.7280) | 0.8306 | 0.7322 | `pop_gru_ens` 0.7102 |
| garcia (68) | **0.4348** (frozen 0.4457, donor 0.4525, twin 0.4541) | 0.4426 (= bolt-on 0.4426) | 0.5793 | 0.4438 | `pop_tf` 0.4339 |
| described (41) | 0.3571 (frozen 0.3571 — gate ≈ off) | 0.3568 | 0.3811 | **0.3550** | `pop_gru` 0.3550 |
| GBE external (1,000) | 0.4934 (frozen 0.4940) | bolt-on 0.4936 (frozen 0.4940) | — | 0.4892 | `pop_gru_ens` 0.4826 |

Matched ensembles on restless4 (X3-007): F3 ×5 0.7113 vs `pop_gru` ×5 0.7127 (−0.0014 [−0.0048, +0.0019]);
F3 ×5 vs its own frozen −0.0009.

### 4.3 Ablations of F3's learned arbitration

| family | F3 learned gate | fixed mixture | fixed-share BMA | GRU fed the episodic prediction (matched complexity) |
|---|---|---|---|---|
| restless4 | 0.7237 | 0.7285 | 0.7368 | 0.7293 |
| garcia | 0.4348 | 0.4453 | 0.4419 | 0.4367 |
| described | 0.3571 | 0.3572 | 0.3674 | 0.3571 |

### 4.4 Cost

| model | params (restless4) | fit CPU-s (selected config, restless4) | prediction ms / trial |
|---|---|---|---|
| `pop_gru` | 8,004 | 59 | 0.05 |
| F3 arbiter | 8,010 | 239 | 0.36 |
| F1 latent | 8,724 | 71 | 2.10 |
| F2 world + policy | 8,171 | 3.5 | 0.37 |
| `pop_gru_ens` (5) | 40,020 | 117 | 0.11 |

### 4.5 Verdicts (dev/tune only; none is a confirmatory finding)

* **F3 learned arbitration — not supported as an advance.** It beats its own ablations (fixed mixture by 0.005 /
  0.011, fixed-share by 0.013 / 0.007 on restless4 / garcia) and own history beats a donor's, so the gate does
  learn something person-specific; but it is 0.0135 nats/choice behind the population seed ensemble on the primary
  family (critic CI [+0.0082, +0.0183]), level with the transformer on garcia, slightly behind on described and
  the external GBE data, and its single-model gain shrinks to −0.0014 once both arms are ensembled. At 4–7× the
  fitting cost of `pop_gru`, it does not pay for itself.
* **F1 individual latent with the trait/context split (proposed novel mechanism) — not supported.**
  Meta-training adds nothing over bolting the latent onto the plain backbone (identical means on restless4 and
  garcia), adaptation gains are ≤ 0.003, and on the toy the trait-only ablation beat the full split. Prediction
  costs 40× `pop_gru`.
* **F2 world model + individual policy — rejected.** 0.03–0.14 nats/choice worse than the same-size `pop_gru` on every family (restless4 +0.098, garcia +0.136, described +0.026):
  predicting the objective outcome process and fitting a few policy weights loses the behavioural regularities a
  sequence model learns directly.
* **General reading.** In these single-session tasks a population recurrent model already infers the person in
  context; explicit person-level adaptation adds ≤ 0.009 nats/choice to a single model and ≈ 0.001–0.003 to a seed
  ensemble — within the seed noise of single fits (critic N-4).

