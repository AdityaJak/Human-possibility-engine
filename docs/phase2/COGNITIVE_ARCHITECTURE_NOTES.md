# Cognitive Architecture Notes — Candidate A (hybrid parametric + exemplar)

_Lane `research/cognitive-architecture`, Phase II. **Everything here is exploratory**: developed and
measured on dev/tune partitions only (synthetic seeds 1000–1099, human dev/tune participants from
`reports/phase2/partitions/psych101.json`). No confirmatory test has been run. Every number below is
produced by a committed script with a committed log under `reports/phase2/cognitive/`._

## 1. Mechanism (`hpe.development.hybrid`, policy class `hybrid_exemplar`)

Two predictors share one input and are mixed by a learned gate:

* **Parametric pathway** — the unchanged Phase I `DevelopmentalLearner` (shared encoder, per-family
  heads, replay memory, mastery estimates; also handles activity/social/practice decisions).
* **Exemplar pathway** — one `ExemplarStore` per family (capacity 512): stored contexts with the
  information the world revealed (label under instructional feedback; action and reward under
  outcome feedback). Prediction is kernel-weighted evidence
  `evid = Σ_i K_i rec_i T_i`, `K_i = exp(-Σ_j w_j (x_j - x_ij)² / 2h²)` with a **learned diagonal
  metric** `w = exp(u)` (GCM-style attention, trained by gradient ascent on the log-likelihood of
  observed outcomes), recency weights `rec_i`, a pseudo-count prior `kappa`, and on **ordinal families**
  an action-distance kernel `Ky` so neighbouring levels of a scale share mass. Pair families use the
  slot-swapped exemplar with the complementary label (order invariance).
* **Arbitration gate** — per family, `g = σ(w·[1, log(1+retrieval mass), max p_par, max p_ex, EMA of
  the exemplar-minus-parametric log-likelihood advantage])`, trained by gradient ascent on the
  mixture log-likelihood. `p = g p_ex + (1-g) p_par`.
* **Consolidation** replays stored exemplars through the parametric pathway every 25 updates;
  **forgetting** prunes the 10% of exemplars the parametric pathway already captures best every 100.
* **Ordinal/metric output**: declared per family (config `ordinal_families`, `Observation.meta["ordinal"]`
  or `AgentInit.families[f]["ordinal"]`); both pathways' distributions are smoothed by `Ky`.

Controls as flags: fixed 50/50 gate (`use_learned_gate=False`), exemplar-only (`use_parametric=False`),
parametric-only (`use_exemplar=False`), no ordinal structure, no consolidation/forgetting, isotropic
kernel, latent-space kernel (`exemplar_space="latent"`). Conformant (including the learning check) for
the default and every control; exact JSON state round-trip; vectorised batch assessment.
`use_learning=False` freezes gate, kernel, store and the inner learner.

## 2. Human dev-partition results — MCPL (Psych-101, exploratory)

Protocol: ≤ 24 configurations per model selected on 47 tune participants, each model then scored once
on 93 dev participants; lapse and (for IGT) reward scale fitted identically for every model; test
participants never loaded. Script `experiments/phase2/cognitive/run_human_dev.py`, grids
`human_dev_grids.yaml`, log `reports/phase2/cognitive/human_dev/mcpl.log`. NLL per choice, chance 2.197.

| model | dev NLL [95% CI] | selected | lapse |
|---|---|---|---|
| **hybrid_exemplar** | **1.436 [1.314, 1.553]** | h=1.0, s=0.5, gate_lr=0.05 | 0.0 |
| hybrid_exemplar_only | 1.449 [1.322, 1.583] | h=1.0, s=0.5 | 0.05 |
| hybrid_no_ordinal | 1.474 [1.329, 1.616] | h=1.0 | 0.1 |
| ref_instance_based (Phase I best) | 1.476 [1.374, 1.585] | h=0.5, s=0.5 | 0.1 |
| hybrid_fixed_gate (50/50) | 1.485 [1.377, 1.601] | h=1.0, s=0.5 | 0.0 |
| memory_only | 1.494 [1.366, 1.625] | — | 0.3 |
| developmental (Phase I) | 1.718 [1.633, 1.804] | — | 0.2 |
| plain_rl | 1.754 [1.690, 1.817] | — | 0.02 |
| ref_delta_rule | 1.805 [1.715, 1.892] | α=0.03, σ=0.75 | 0.3 |

Paired contrasts (participant bootstrap, n = 93; negative favours the first):

| contrast | mean | 95% CI | share better |
|---|---|---|---|
| hybrid − ref_instance_based | **−0.040** | [−0.062, −0.016] | 0.62 |
| hybrid − developmental | −0.282 | [−0.334, −0.232] | 0.91 |
| hybrid − memory_only | −0.058 | [−0.076, −0.041] | 0.63 |
| fixed-gate − ref_instance_based | +0.009 | [−0.011, +0.027] | 0.40 |
| exemplar-only − ref_instance_based | −0.027 | [−0.054, +0.000] | 0.56 |
| no-ordinal − ref_instance_based | −0.002 | [−0.048, +0.044] | 0.55 |

**Reading.** The hybrid closes the Phase I gap on MCPL (−0.28 vs the developmental learner) and edges
the best exemplar model by 0.04 nats with a CI excluding zero — on dev participants. The controls say
where the gain comes from: the *learned* gate (a fixed 50/50 mixture has no advantage) and the
*ordinal* output structure (removing it removes the advantage); the learned kernel alone (exemplar-only)
is borderline. The hybrid is also the best-calibrated model (selected lapse 0.0; the reference needs
0.1). By predictability tercile the hybrid wins in the middle and most-predictable thirds and is
average in the least-predictable third (report.md). Caveat: the effect is small (0.04 nats ≈ 4% of a
choice's likelihood) and dev-partition only; Gate D requires the frozen test partition and ≥ 3 families.

## 3. Human dev-partition results — IGT (exploratory)

Same protocol: 103 tune / 204 dev participants; log `human_dev/igt.log`. NLL per choice, chance 1.386.

| model | dev NLL [95% CI] | selected | lapse |
|---|---|---|---|
| **hybrid_exemplar** | **1.151 [1.116, 1.188]** | β_ex=3, gate_lr=0.2, decay=0.9, scale 250 | 0.01 |
| developmental (Phase I) | 1.153 [1.117, 1.189] | — | 0.0 |
| hybrid_fixed_gate | 1.153 [1.115, 1.189] | β_ex=3, decay=0.9 | 0.02 |
| hybrid_exemplar_only | 1.165 [1.128, 1.202] | β_ex=3, decay=0.9 | 0.1 |
| memory_only | 1.277 [1.247, 1.307] | — | 0.3 |
| ref_instance_based | 1.278 [1.259, 1.296] | β=6, decay=0.5 | 0.2 |
| ref_delta_rule | 1.297 [1.282, 1.313] | α=0.4, β=6 | 0.2 |
| plain_rl | 1.329 [1.319, 1.339] | — | 0.0 |

Paired contrasts (n = 204): hybrid − developmental **−0.002 [−0.005, +0.001]** (no change);
hybrid − ref_instance_based −0.127 [−0.154, −0.100] (68% better); hybrid − memory_only −0.126;
exemplar-only − developmental +0.012 [+0.007, +0.016]. Terciles: the hybrid and the developmental
learner are tied in every tercile and widen their lead on the most-predictable third (0.99 vs 1.13–1.27).

**Reading.** Adding the exemplar pathway costs nothing on the task where the parametric learner was
already the best model: the gate leans parametric here (the fixed 50/50 control is equally good only
because the exemplar pathway on a constant context reduces to recency-weighted action values, itself a
strong IGT model at 1.165). Phase I's IGT advantage is therefore retained, on dev participants.

## 4. v2 task families and the synthetic dev experiment

`hpe.tasks.v2.V2World`: conjunction (XOR of two feature-group majorities; balanced; a linear probe on
raw features scores 0.58 on holdout), context_switch (context-dependent class readout), nonstationary
(readout replaced every 150 training samples; per-regime eval sets), delayed (feedback delivered 5
episodes late), ordinal_cue (two 5-level cues → 9-level criterion with a non-linear interaction, 5 of
25 configurations held out). `shared_structure=False` is the specialist-should-win control.

**Pilot finding (dev seeds 1004, 1011; `pilot_conjunction.log`).** After 400/800/1,200 instructional
episodes, no current learner exceeds 0.68 held-out accuracy on `conjunction` (exemplar-only 0.57–0.68,
hybrid 0.58–0.65, memory-only 0.57–0.68, plain RL 0.37–0.53 — below chance, the linear fit
anti-generalises). The family is learnable in principle (deterministic, balanced) but not by any
architecture in this repository: a concrete target for Candidates B/C (non-linear representation
learning), and a family where "nobody wins" must be reported as such.

**Synthetic dev experiment** (`run_synthetic_dev.py`; 6 arms × 5 dev seeds × 6 agents, 300
episodes/family, 25% instructional feedback, label noise 0.05; held-out accuracy after the budget,
mean ± sd over seeds; `results.json`, `report.md`, `run.log`; CPU 248 s). Shared-structure world:

| arm | conjunction (chance .50) | context_switch (.33) | delayed (.50) | ordinal_cue (.11) |
|---|---|---|---|---|
| hybrid_exemplar | 0.616 ± 0.036 | 0.541 ± 0.072 | **0.919 ± 0.036** | 0.197 ± 0.210 |
| hybrid_fixed_gate | 0.562 ± 0.068 | 0.504 ± 0.037 | 0.917 ± 0.033 | 0.218 ± 0.233 |
| hybrid_exemplar_only | 0.616 ± 0.045 | **0.573 ± 0.073** | 0.824 ± 0.052 | **0.319 ± 0.290** |
| developmental | 0.489 ± 0.032 | 0.417 ± 0.026 | 0.899 ± 0.035 | 0.139 ± 0.211 |
| plain_rl | 0.494 ± 0.055 | 0.469 ± 0.051 | 0.897 ± 0.054 | 0.154 ± 0.193 |
| memory_only | **0.643 ± 0.052** | 0.547 ± 0.076 | 0.836 ± 0.069 | 0.270 ± 0.175 |

Paired-by-seed contrasts, hybrid − control (95% CI): conjunction vs memory_only −0.027 [−0.039,
−0.015], vs developmental +0.127 [+0.108, +0.142]; context_switch vs exemplar-only −0.032 [−0.053,
−0.011], vs developmental +0.124 [+0.068, +0.181]; delayed vs developmental +0.019 [+0.001, +0.038],
vs exemplar-only +0.094 [+0.069, +0.119]; ordinal_cue vs exemplar-only −0.122 [−0.183, −0.064].
`nonstationary` (`nonstationary_recovery.md`): before the first switch 0.696, immediately after 0.342,
100 episodes later 0.614 for the hybrid; developmental 0.666 → 0.361 → 0.641; plain_rl 0.692 → 0.374 →
0.608; exemplar-only 0.584 → 0.348 → 0.507; memory_only 0.619 → 0.347 → 0.468. (The runner's "final"
column for this family is the post-switch drop at episode 300, a metric-placement error kept in the
committed report rather than silently redefined.)

**Reading.** On synthetic v2 families the hybrid beats both parametric baselines on four of five
families and recovers from regime switches as fast as the best parametric learner, but it does **not**
beat the strongest exemplar baseline: memory_only wins conjunction, and the hybrid's own exemplar-only
control wins context_switch and ordinal_cue. The learned gate helps against a fixed 50/50 mixture
(conjunction +0.053, context_switch +0.037) yet still under-weights exemplars where they are better —
its features (retrieval mass, confidences, recent advantage) do not identify the better pathway
reliably on these families. The specialist-structure world shows the same pattern; there is no
evidence here of a shared-structure benefit. Against the Gate B bar (≥ 0.05 over the strongest baseline
in ≥ 2 of 3 families) Candidate A would currently **fail on synthetic data** while showing a real
dev-partition advantage on human MCPL — reported as exactly that.

## 5. What was tried and abandoned

* Defining `conjunction` as sign agreement of two latent projections: labels were 212/28 imbalanced
  and rough at single-bit-flip scale (exemplar 0.55, linear 0.63); replaced by XOR of group majorities.
* Unbalanced regime/context readouts (12/96/132): fixed with Phase I-style bias balancing.
* Ordinal jitter clipped at the scale's edge onto the correct label: now reflects inward.

## 6. What would falsify Candidate A

* On the frozen MCPL test partition (never read), hybrid − ref_instance_based with CI including 0, or
  worse than −0.02 nats (the Gate D margin) — the dev advantage would then be selection noise.
* The fixed-gate control matching the learned gate on test data (the arbitration claim fails).
* Losing the Phase I developmental learner's IGT advantage (the gate failing to favour the parametric
  pathway where it should).
* On v2 synthetic families: not beating the strongest baseline by ≥ 0.05 in ≥ 2 of 3 families.

## 7. Summary for the Research Director

* Human MCPL (dev): first model to beat the Phase I best (instance-based) — −0.040 nats [−0.062,
  −0.016]; the learned gate and the ordinal output structure are the active ingredients (controls).
* Human IGT (dev): Phase I advantage retained exactly (−0.002 [−0.005, +0.001] vs developmental).
* Synthetic v2 (dev): beats both parametric baselines on 4/5 families and matches the best recovery
  after regime switches, but does not beat the strongest exemplar baseline on 3/5 — the gate is not
  yet a reliable arbiter on synthetic families; `conjunction` is unsolved by every architecture.
* Status against the pre-fixed gates: Gate D plausible for MCPL+IGT pending the frozen test partition
  and a third family; Gate B would fail on the current synthetic families. No confirmatory run exists.
* Next: (1) gate features that measure *pathway disagreement calibration* (e.g. held-out-by-time
  validation of each pathway) rather than raw confidence; (2) a non-linear parametric head for
  conjunction-type structure; (3) freeze the configuration selected here and hand the frozen test
  evaluation to the Independent Critic — never to this lane.

## 8. Compute

MCPL dev comparison 236 CPU-s; IGT dev comparison 788 CPU-s; synthetic dev experiment 248 CPU-s;
conjunction pilot < 10 s; lane tests ≈ 11 s per run. Total ≈ 0.36 CPU-h of the 12 CPU-h budget.
