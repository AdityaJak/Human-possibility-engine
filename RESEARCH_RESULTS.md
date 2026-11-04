# RESEARCH_RESULTS

_Updated 2026-11-04. Every number below is read from a committed artifact under `reports/` produced by
code in this repository; the path is given with each table. Regenerate any experiment's result with
`python -m hpe.evaluation analyze reports/experiments/<id>` (byte-identical)._

> **Synthetic research prototype; not validated for predicting human futures.**
> Simulated results come from a synthetic world. Effects may simply reflect assumptions built into the
> simulator and are not estimates of real-world causal effects.

## 0. Protocol in one paragraph

Agents practise on symbols drawn from a 64-symbol train pool under hidden rules; skill is measured only
on a fixed 60-item set per family built from 32 **held-out symbols never seen in practice**, with learning
disabled and ties credited fractionally (a uniform policy scores exactly chance). `barter` is a fourth
family never practised during development (transfer test). Four policy classes implement one protocol
and receive the identical item/feedback stream (common random numbers). Each experiment is a YAML
preregistration whose hash is written to disk before any run; analysis regenerates every number and
figure from raw logs. Unit of analysis is the seed (10 seeds). Primary test: paired percentile bootstrap
(4,000 resamples), Holm-corrected secondaries. See `docs/RESEARCH_PROTOCOL.md`.

## 1. Summary of preregistered verdicts

| Experiment | Question | Primary result | Verdict | Artifact |
|---|---|---|---|---|
| H1 `exp_baselines_heldout` | Does the developmental learner beat the strongest baseline by ≥10% on held-out skill? | +0.012 [−0.008, +0.031] vs plain RL (+1.4%) | **FAILED** | `reports/experiments/exp_baselines_heldout` |
| Exp 1 `exp01_acquired_learning` | Does instructional feedback vs none change held-out skill? | +0.391 [+0.374, +0.410] | passed | `reports/experiments/exp01_acquired_learning` |
| Exp 6 / H3 `exp06_transfer` | Does a shared representation speed learning of an unseen family? | +0.090 [+0.050, +0.123] vs specialists | passed | `reports/experiments/exp06_transfer` |
| Ablations `exp_ablations` | Is held-out skill caused by learning rather than scripts? | freezing learning: −0.409 [−0.425, −0.394] | passed | `reports/experiments/exp_ablations` |
| Exp 2 `exp02_twin_tutoring` | Twin societies: does tutoring access raise held-out skill through learning? | +0.029 [+0.022, +0.037]; non-learning society: exactly 0 | passed (manipulation check **failed** as defined, §4) | `reports/experiments/exp02_twin_tutoring` |
| Exp 3–5 | social influence, midlife education, multigenerational | running at time of writing | pending | — |
| Exp 7b `exp07b_causal_grounding` | Can our estimator recover a randomised human effect (HeartSteps V1)? | +12.2% [−0.5%, +27.4%]; published +14% inside CI | passed (see caveat §5) | `reports/human_validation/exp07b_causal_grounding` |
| Exp 7 `exp07_human_heldout` | Do our policy classes predict held-out human choices (Psych-101)? | developmental class not in first-pass build | inconclusive; rerun in progress | `reports/human_validation/exp07_human_heldout` |

## 2. H1 — the investor's decisive question (failed)

Budget: 20 agents/arm, 300 practice episodes per family (25% instructional, 75% outcome-only), label
noise 0.05, ages 8→18 with the perceptual age gate on. Primary metric: mean held-out accuracy over
sort/compare/signal after the budget.

| Arm | Held-out accuracy | Log loss | ECE (self-estimate vs measured) | Train–holdout gap |
|---|---|---|---|---|
| developmental | 0.849 [0.831, 0.868] | 0.367 | 0.108 | 0.055 |
| plain_rl | 0.837 [0.815, 0.863] | 0.383 | 0.128 | 0.080 |
| memory_only | 0.773 [0.746, 0.799] | 0.573 | 0.173 | 0.170 |
| static_persona | 0.438 [0.425, 0.449] (chance 0.444) | 2.905 | 0.483 | 0.005 |

| Contrast | Estimate | 95% CI | Verdict |
|---|---|---|---|
| developmental − plain_rl (primary, threshold +0.0837) | +0.0121 | [−0.0080, +0.0306] | **failed** |
| calibration non-inferiority (log loss, margin 0.05) | −0.0163 | [−0.0523, +0.0215] | passed |
| developmental − memory_only | +0.0763 | [+0.0564, +0.0963] | passed |
| developmental − static_persona | +0.4109 | [+0.3915, +0.4323] | passed |
| episodes to 0.70 criterion on compare (dev − plain_rl) | +9.0 | [+4.7, +13.7] | developmental is **slower** (exploratory) |

Sensitivity (primary contrast): outcome-only feedback +0.037 [+0.020, +0.054] (below threshold);
label noise 0.15 **−0.046 [−0.062, −0.030]** (developmental worse); 100 episodes −0.005
[−0.023, +0.012]; no age gate +0.017 [+0.001, +0.032].

Independent audit (`reports/audit/AUDIT.md`): own numpy bootstrap gives +0.0121 [−0.0081, +0.0313];
excluding seed 7 (which overlapped the learner's tuning seeds) +0.0135 [−0.0089, +0.0344] — verdict unchanged.

**Reading.** Within a task family, the developmental learner is statistically indistinguishable from
a plain per-family online learner and is more fragile to label noise. Its advantages are elsewhere
(§3) — and the honest answer to "why does an agent that develops from experience beat a biography or
a memory-based agent?" is: it beats both of those by a wide margin, and it does not yet beat plain RL.

## 3. Where development helps: transfer (Exp 6, passed) and what the ablations say

Transfer probe: after 300 episodes/family on sort/compare/signal, a *clone* of each agent practises the
never-seen `barter` family for 0–64 shots; AUC = mean held-out barter accuracy across shot counts.

| Arm | Shared-structure world AUC | Orthogonal-structure world AUC | Zero-shot barter |
|---|---|---|---|
| developmental (shared encoder) | 0.742 [0.697, 0.777] | 0.552 [0.542, 0.563] | 0.500 |
| developmental_specialists (separate encoders) | 0.652 [0.640, 0.664] | 0.644 [0.632, 0.655] | 0.500 |
| plain_rl | 0.692 [0.673, 0.713] | 0.690 [0.674, 0.706] | 0.500 |
| memory_only | 0.669 [0.651, 0.688] | 0.665 [0.649, 0.682] | 0.500 |

Primary: shared − specialists = +0.0895 [+0.0503, +0.1228] (passed, threshold 0.03). vs plain_rl
+0.0493 [+0.0134, +0.0823] (passed, Holm). **Negative transfer** in the orthogonal control:
−0.0919 [−0.1045, −0.0784] vs specialists. Zero-shot barter is exactly chance for every class — the
transfer family is genuinely unseen. A from-scratch re-run of this experiment reproduced all 40 raw
outcome logs bit-for-bit.

Ablations (primary metric = held-out accuracy after 300 episodes + 30 demonstrations per family):

| Ablated component | Δ vs full learner | 95% CI | Reading |
|---|---|---|---|
| learning (frozen parameters) | −0.409 | [−0.425, −0.394] | skill is learned, not scripted (**primary, passed**) |
| feedback | −0.273 | [−0.286, −0.260] | skill needs outcome information |
| shared representation | +0.001 | [−0.004, +0.005] | no within-family effect; its effect is transfer (0.652 vs 0.731 AUC) |
| replay memory | −0.007 | [−0.011, −0.003] | tiny |
| social learning | −0.004 | [−0.013, +0.004] | none in this lab (demonstrations are a small share of input) |
| plasticity schedule | **+0.012** | [+0.008, +0.016] | the age schedule **hurts** |
| curriculum | 0.000 | — | inert by design in a fixed-schedule lab |
| forgetting | +0.001 | [−0.003, +0.006] | none within the lab horizon; retention −0.007 |

## 4. Society experiment — Exp 2 twin tutoring (passed; one check failed)

100-agent twin societies, 20 years, matched `population_seed`/`task_seed`, per-agent per-purpose RNG
streams; treatment: tutoring access (extra instructional-feedback episodes) for ages 5–15.

| Arm | Held-out accuracy, ages 16–20 | Ages 8–12 | Employment 18–30 | Gini |
|---|---|---|---|---|
| dev_control | 0.831 [0.812, 0.853] | 0.757 | 0.584 | 0.873 |
| dev_tutoring | 0.860 [0.842, 0.881] | 0.758 | 0.587 | 0.877 |
| rl_control | 0.831 [0.803, 0.862] | 0.770 | 0.581 | 0.834 |
| rl_tutoring | 0.850 [0.825, 0.876] | 0.806 | 0.656 | 0.843 |
| static_control | 0.451 [0.435, 0.467] | 0.472 | 0.577 | 0.734 |
| static_tutoring | 0.451 [0.435, 0.468] | 0.472 | 0.579 | 0.736 |

* Primary (dev_tutoring − dev_control, threshold 0.02): **+0.0289 [+0.0216, +0.0367]**, passed.
* Mechanism check (static persona arms, equivalence margin 0.02): **+0.0000 [0, 0]**, passed — in a
  society that cannot learn, tutoring changes nothing, so the effect flows through learning.
* plain_rl: +0.0192 [+0.0006, +0.0366]. Transfer family (barter): 0 in all arms (never practised).
* Sensitivity: label noise 0.15 → +0.0013 [−0.0058, +0.0112] (**effect disappears**); no age gate
  +0.0134 [+0.0046, +0.0238]; 2 or 8 tutoring episodes/tick: +0.023, +0.027 (robust to dose).
* **Manipulation check failed as preregistered.** It required `tutored_share_16_20 ≥ 0.5`, but the
  world recomputes `AgentState.tutored` every tick from *current* eligibility
  (`packages/hpe/world/engine.py:706`), so at ages 16–20 the flag is 0 in both arms. The intervention
  did apply (the world lane's tests show tutoring episode counts rise only in the treated arm, and
  the effect replicates in plain_rl); the check measured the wrong thing. Recorded as failed; a
  corrected check (ever-tutored share) belongs in a new preregistration, not a silent edit.
* 140 runs, 4,766 s wall, 0 LLM calls, $0 API cost.

## 5. Real human data

### Exp 7b — causal grounding on HeartSteps V1 (CC-BY-4.0; 37 participants, 6,590 available decision points)

| Suggestion vs none | Our estimate | 95% cluster-bootstrap CI | p | Published (verified from abstract, PMID 30192907) |
|---|---|---|---|---|
| any | +12.2% | [−0.5%, +27.4%] | 0.083 | +14% (p = .06) — inside our CI |
| walking | +21.2% | [+4.2%, +42.0%] | 0.024 | +24% (p = .02) — inside our CI |
| anti-sedentary | +1.4% | [−13.3%, +20.8%] | — | — |

Complete-case sensitivity: +16.1% (p = 0.043). **Caveats (from the audit):** the preregistered
consistency rule is weak — the CI also contains 0, so a null estimate would also have "passed"; the
public data has 37 of the paper's 44 participants; exact covariates unverified (abstract only). This
validates an estimation pipeline on randomised human data. It says nothing about our agents.

### Exp 7 — held-out human choice prediction on Psych-101 (Apache-2.0)

Primary: per-choice negative log-likelihood on held-out participants (70/30 split by participant),
multiple-cue probability learning (MCPL, 9 actions, 69 held-out people); secondary: Iowa Gambling Task
(4 actions, 153 people). First pass (policy classes not yet registered in that build):

| Model | MCPL NLL (chance 2.197) | IGT NLL (chance 1.386) |
|---|---|---|
| instance-based | 1.505 [1.388, 1.616] | 1.281 [1.253, 1.307] |
| population-static | 1.602 [1.516, 1.693] | n/a |
| delta rule | 1.802 | 1.304 |
| fitted bias | 1.936 | 1.350 |

Secondary contrast (best adaptive − best static): MCPL −0.097 [−0.136, −0.055]; IGT −0.070
[−0.095, −0.046]. **Primary inconclusive**: no verdict on the developmental learner yet. A rerun with
all five policy classes is in progress; results will replace this section when committed.

## 6. The stored 100-agent, 80-year society (descriptive)

`reports/runs/demo_developmental_100a_80y_s42`: 138,607 events, 99 agents at year 80 (91 births, 103
deaths over the run), 117 s wall / 114 s CPU, 356 MB peak RSS, 0 LLM calls. Replay from seed and resume
from the year-40 checkpoint both reproduce the event-chain and final-state hashes
(`replay_verification.json`). Mean held-out accuracy by simulated age band
(`python scripts/describe_run.py reports/runs/demo_developmental_100a_80y_s42 --markdown`):

| age band | sort (chance .33) | compare (.50) | signal (.50) | barter (.50, never practised) | assessments |
|---|---|---|---|---|---|
| 0–3 | 0.393 | 0.580 | 0.556 | 0.500 | 113 |
| 6–10 | 0.500 | 0.669 | 0.662 | 0.500 | 503 |
| 14–18 | 0.742 | 0.828 | 0.849 | 0.500 | 549 |
| 18–30 | 0.774 | 0.854 | 0.876 | 0.500 | 1,557 |
| 50–65 | 0.748 | 0.843 | 0.853 | 0.500 | 1,675 |
| 80–120 | 0.550 | 0.683 | 0.674 | 0.500 | 223 |

The rise-plateau-decline shape partly reflects **scripted** age gates (perception, plasticity,
forgetting) applied to every policy class; it is not evidence about human ageing.

## 7. Compute and cost

* Learning-lab experiments: H1 200 runs / 1,316 s; Exp 1 ~209 s; Exp 6 40 runs / 355 s; ablations 90
  runs / 420 s. Society Exp 2: 140 runs / 4,766 s. Hardware: 4-vCPU x86_64, 15.7 GiB RAM, shared with
  other jobs (load average 3–5 during runs).
* Scale records (`reports/scale/`): 100 agents × 2 years, developmental 5.45 s (38.7 agent-years/s,
  100 MB); static persona 5.68 s. **No 1,000- or 10,000-agent benchmark has been run yet.**
* Cost: $0 in API calls (no language model anywhere). CPU: 114 s per 100-agent × 80-year run
  ≈ 0.014 CPU-s per agent-year; at an *assumed* $0.05 per vCPU-hour that is ≈ $0.002 per such run.

## 8. Claims permitted by these results (see docs/CLAIMS_POLICY.md)

Permitted: the synthetic learning claims in §2–§4 with their CIs and the synthetic caveat; the
reproducibility and replay claims in §6; the HeartSteps pipeline check in §5 with its caveats.
Not permitted: any statement that the developmental learner beats plain RL on within-family skill;
any human-validity claim for the agents; any real-world causal claim from Exp 2; any scale claim
beyond 100 agents.
