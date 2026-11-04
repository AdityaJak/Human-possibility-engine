# Causal Validity Report — Phase II, lane research/causal

_All numbers come from committed artifacts under `reports/phase2/causal/` produced by
`python -m hpe.evaluation.causal`. Data: HeartSteps V1 `suggestions.csv` (CC-BY-4.0; Klasnja et al.
2019), SHA-256 `60ee7896…`, downloaded by `datasets/adapters/heartsteps.py`; never committed._

## 0. What this report can and cannot establish

* **Can:** whether the estimator Phase I used is valid (by simulation with known truth); whether the
  suggestion effect varies with pre-listed context (one pre-specified confirmatory test); what
  retrospective off-policy estimates say about simple suggestion policies, under explicit assumptions.
* **Cannot:** Gate E (prospective decision improvement). No retrospective analysis of 37 participants
  can show that model-guided action selection improves outcomes for new people; that requires a
  prospective, randomised study with the policy actually deployed. Nothing here is a personalised
  prediction, and nothing here transfers to the synthetic society or to the Future Self product.

## 1. Identification assumptions (all three analyses)

1. At every *available* decision point, sending a suggestion was randomised with a known, constant
   probability (design 0.6; empirical 0.594 over 6,590 points). **Data note:** the column
   `is.randomized` coincides exactly with `send` among available points (3,908 True/True, 2,682
   False/False); per the dataset documentation it records the *realised selection to send*, not
   whether a point was randomised. Every available point is therefore a randomised decision point,
   as Phase I assumed; the column must not be used as a sample filter.
2. Consistency; no interference between participants; positivity (0.6 ∈ (0, 1)).
3. The estimand is the **causal excursion effect** (Boruvka et al. 2018): the effect of sending vs not
   at an available point, marginal over the realised history. Control covariates
   (`log(pre-steps + 0.5)`, study day) only improve precision; the working model may be misspecified
   without biasing the marginal effect when centring uses the true probability (verified in §2).
4. Outcome: `log(jbsteps30.zero + 0.5)`, zero-imputed missing minutes (dataset convention).
5. For off-policy evaluation: the policy's action depends only on observed context at the decision
   point, and the logging probability is known (same randomisation), so importance weights are exact.

## 2. Estimator validation by simulation (`simulation_validation.json`)

Synthetic micro-randomised trials with HeartSteps' structure (37 participants, 5 points/day, 42
days, availability 0.8, P(send) 0.6, participant random effect, log-scale outcome), 500 replicates
per scenario, WCLS as implemented in `hpe.evaluation.human.causal.wcls` (cluster-robust CR1
sandwich, t(G−1)).

| Scenario (truth) | Bias of β̂₀ | 95% CI coverage | Rejection at α = 0.05 | Moderator β̂₁: bias / coverage |
|---|---|---|---|---|
| constant effect (β₀ = 0.115) | −0.003 | 0.952 | 0.68 (power) | — |
| context-moderated (binary context, β₁ = 0.20) | +0.001 | 0.962 | 0.74 | −0.006 / 0.932 |
| carry-over of previous suggestion (γ = 0.10) | −0.003 | 0.948 | 0.70 | — |
| day-in-study decay moderation (β₁ = 0.20) | −0.000 | 0.952 | 0.73 | +0.004 / 0.954 |
| misspecified nuisance (quadratic day, interaction) | +0.003 | 0.932 | 0.72 | — |
| null (β₀ = 0) | −0.001 | 0.956 | **0.044** (type I) | — |

Power at the HeartSteps sample size for a constant effect: 0.71 at β₀ = 0.115 (Phase I point
estimate), 0.83 at 0.131 (published +14%), 0.26 at 0.06.

**Reading.** The estimator is unbiased and its intervals are close to nominal in every scenario,
including carry-over and a misspecified control model; type I error is at nominal level. The trial
is adequately powered for the published marginal effect but not for effects half that size, and
**moderator effects of 0.20 are detected only 50–74% of the time at 37 participants** — which bounds
what §3 can show with 15.

## 3. Heterogeneous treatment effects

Pre-listed moderators (fixed before analysis): `evening` (local hour ≥ 17), `weekend`, `log_pre_c`
(centred prior 30-min activity), `home` (location category), `day_c` (day-in-study decay). Model:
WCLS with the moderator in both the control and the effect part; Holm over the five.

### 3a. Discovery on the dev partition (22 participants, 3,885 points) — EXPLORATORY (`hte_dev.json`)

Marginal effect: +0.080 log-points (+8.3%), 95% cluster-bootstrap CI [−0.085, +0.288], p = 0.41.

| Moderator | β̂₁ | cluster SE | raw p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| evening | −0.069 | 0.232 | 0.77 | 1.00 | [−0.514, +0.401] |
| **weekend** | **−0.366** | 0.212 | **0.099** | 0.50 | [−0.784, +0.032] |
| log_pre_c | +0.031 | 0.031 | 0.34 | 1.00 | [−0.029, +0.095] |
| home | +0.046 | 0.218 | 0.84 | 1.00 | [−0.381, +0.417] |
| day_c | −0.323 | 0.344 | 0.36 | 1.00 | [−0.979, +0.305] |

No moderator survives Holm. The pre-fixed discovery rule (≤ 2 moderators with raw p < 0.10) selected
`weekend` only.

### 3b. Confirmation on the test partition (15 participants, 2,705 points) — CONFIRMATORY, single run (`hte_test.json`, spec `experiments/phase2/causal/hte_confirm.yaml`)

Pre-specified rule: Holm p < 0.05 and negative sign. Result: **β̂₁(weekend) = −0.088, SE 0.199,
p = 0.667, bootstrap CI [−0.443, +0.306] → NOT supported.** (Marginal effect on these participants:
+0.149, +16.1%, CI [−0.007, +0.310], p = 0.099.)

**Reading.** We found no reproducible heterogeneity of the suggestion effect across the five
pre-listed contexts. Given §2's power analysis this is weak evidence of absence: an interaction of
−0.37 (the dev point estimate) would be detected only about half the time even with all 37
participants. No HTE claim is permitted.

## 4. Off-policy evaluation of suggestion policies

Policies: `never`, `always`, `behaviour` (the trial's P(send) = 0.585), and `contextual_greedy` —
send iff a linear outcome model fitted **on dev only** (controls + suggestion × [1, five moderators])
predicts a positive conditional effect. Estimators: IPW with exact logging probabilities and doubly
robust (DR) with the dev outcome model; participant cluster bootstrap (2,000). Outcome scale: mean
log(30-min steps + 0.5) per available decision point.

### 4a. Dev partition — EXPLORATORY and in-sample (`ope_dev.json`)

contextual − never, DR +0.154 [+0.004, +0.305]; contextual − always +0.076 [−0.025, +0.195]. In-sample
by construction (the policy was learned on these rows); shown only to document the selection.

### 4b. Test partition — CONFIRMATORY, pre-specified (`ope_test.json`, spec `experiments/phase2/causal/ope_confirm.yaml`)

| Policy | share sent | IPW value [95% CI] | DR value [95% CI] | DR − never [CI] | DR − always [CI] |
|---|---|---|---|---|---|
| never | 0.00 | 2.462 [2.091, 2.763] | 2.639 [2.218, 2.979] | — | −0.167 |
| always | 1.00 | 2.938 [2.507, 3.357] | 2.806 [2.434, 3.145] | **+0.167 [+0.018, +0.320]** | — |
| behaviour (p = 0.585) | 0.59 | 2.740 [2.351, 3.069] | 2.737 [2.346, 3.064] | +0.098 [+0.010, +0.187] | −0.069 |
| contextual (dev-learned) | 0.66 | 2.927 [2.509, 3.296] | 2.827 [2.441, 3.157] | **+0.188 [+0.066, +0.324]** | **+0.021 [−0.107, +0.156]** |

Pre-specified rule (contextual − never, DR, CI excludes 0): **passed**. But the comparison that
matters for "tailoring" — contextual vs always — is +0.021 with a CI spanning [−0.107, +0.156]
(IPW: −0.011 [−0.161, +0.143]). The dev-learned policy sends 66% of the time and achieves what
always-sending achieves. IPW's always − never contrast (+0.476 [+0.163, +0.783]) is larger and much
noisier than DR's; both are consistent with the WCLS marginal effect on the same participants (+0.149).
Effective sample size of the contextual policy's IPW weights: 1,423 of 2,705 points. The test run was
executed once under the spec; a second, flagged run (`--force`, recorded in `ope_test_run.marker`)
only added the vs-always contrast and reproduced every number exactly (fixed seed).

**Reading.** On held-out participants, sending suggestions beats not sending on the 30-minute outcome
(DR +0.167, consistent with the trial's published marginal effect), and a context-tailored policy
learned from 22 participants is **not distinguishable from simply always sending**. There is no
evidence here that contextual tailoring adds value; there is also no evidence that it costs anything.

## 5. Limitations

* 37 participants (22/15 split): heterogeneity and policy contrasts are underpowered (§2).
* Availability is a condition of the estimand (effects at available points only); non-availability
  (connectivity, snooze, transit) is not modelled.
* Zero-imputed missing tracker minutes; Phase I's complete-case sensitivity gave a larger marginal
  effect (+16.1%); not repeated here.
* The published analysis' exact covariates are unverified (abstract only); our control set differs.
* The proximal 30-minute outcome says nothing about long-run activity, habituation (the trial reports
  declining effects over time), or engagement burden of always-sending.
* The contextual policy is a linear, deterministic rule over five hand-listed features; a richer
  policy could differ — but any such policy must be evaluated on genuinely new participants.
* The logging probability was taken as the empirical dev share (0.585); the design value 0.6 changes
  weights by < 3%.

## 6. What may and may not be claimed

**May be claimed (with the assumptions in §1 and the data caveats in §5):**
1. The WCLS estimator used in Phase I is unbiased with nominal coverage under constant, moderated,
   carry-over and misspecified-nuisance data-generating processes (simulation, 500 replicates each).
2. On held-out HeartSteps participants, delivering a suggestion raises the proximal 30-minute step
   outcome relative to not delivering one (DR +0.167 log-points, 95% CI [+0.018, +0.320]).
3. A pre-specified test of effect heterogeneity by weekend was not supported; no heterogeneity across
   five pre-listed contexts was found.
4. A context-tailored suggestion policy learned on 22 participants performed no better than always
   sending on 15 held-out participants (+0.021, CI [−0.107, +0.156]).

**May not be claimed:**
* Any prospective benefit of model-guided action selection (Gate E) — not tested and not testable
  from this data.
* Individual-level counterfactuals ("this person would have walked more if…") — the estimand is a
  population excursion effect at available points.
* Transfer of any of this to the synthetic agents, to the Future Self product, or to other behaviours.
* That tailoring is useless in general — only that this policy, from this little data, showed no
  gain on this proximal outcome.

## 7. What Gate E would require

A prospective micro-randomised or SMART-style study that deploys the candidate policy against a
fixed comparator (e.g. the behaviour policy), with a preregistered primary distal outcome, sample
size from §2-style power analysis (hundreds of participants for tailoring effects of the size seen
here), participant consent and ethics review, and an independent analysis. Until then, Gate E is
**not started**.

## 8. Reproduction and cost

```
python datasets/adapters/heartsteps.py                       # download + sha256 verify
python -m hpe.evaluation.causal partition                    # refuses to change the frozen split
python -m hpe.evaluation.causal simulate --n 500             # ~194 CPU-s
python -m hpe.evaluation.causal hte --partition dev          # ~15 CPU-s
python -m hpe.evaluation.causal hte --partition test --spec experiments/phase2/causal/hte_confirm.yaml --force
python -m hpe.evaluation.causal ope --partition dev
python -m hpe.evaluation.causal ope --partition test --spec experiments/phase2/causal/ope_confirm.yaml --force
```

CPU used in this lane: ≈ 260 CPU-seconds (0.07 CPU-h) of the 3 CPU-h budget. Tests:
`pytest tests/evaluation/causal` (9 tests: partition tamper-evidence, simulator probabilities,
WCLS recovery of planted effects, Holm, planted-moderator detection, IPW/DR identities, policy
rule, CSV feature parsing).
