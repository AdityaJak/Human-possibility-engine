# Breakthrough Verdict — Phase III

_Issued by the Research Director on 2026-11-04 under `PHASE3_RESEARCH_THESIS.md` §8, after the independent
critic's design review, its re-verification of the fixes and its results review (`INDEPENDENT_SCIENTIFIC_AUDIT.md`).
Every number is exploratory (development / tuning participants or development seeds). No test partition was opened._

## Verdict

**No scientific breakthrough. Stage 8 decision: NO-GO for hypothesis H3 as preregistered; the programme pivots to
new, separately preregistered questions.**

| level | achieved? |
|---|---|
| Narrow benchmark advance (beat the strongest comparator on held-out people in ≥ 2 counted families) | **No** |
| Generalizable methodological advance (an ablation-supported mechanism that survives strong baselines) | **No** |
| Validated improvement in human decisions (prospective, causal) | **No — not attempted; requires a randomised study** |

Why NO-GO rather than PIVOT for H3: every individual-adaptation effect measured at matched seeds or against
conventional adapters is ≤ 0.005 nats/choice — within the seed noise of a single fit — while the preregistered
margin is 0.02; the best candidate's single-model gain disappears once both arms are ensembled. The independent
critic reached the same recommendation. Continuing to tune these candidates against the same data would be
optimising on noise.

## Answers to the mandate's questions

| question | answer |
|---|---|
| **Which architecture performed best?** | Among candidates: **F3** (population GRU + personal episodic memory with a learned reliability gate). Among all models: **population models without explicit personalisation** — a 5-seed GRU ensemble (restless4 0.7102; external GBE 0.4826), a small causal transformer (garcia 0.4339), a single GRU (described 0.3550); a 5-seed ensemble of the Bayesian-head adapter (0.7096) in the matched control |
| **Which existing methods did it beat?** | F3 beat a single same-size or 24-config-tuned population GRU (−0.0088 [−0.0156, −0.0024], restless4), an LSTM, a causal transformer and every single conventional adapter on restless4 (by 0.003–0.009), the hierarchical Bayesian cognitive model, the Phase II GRU, and its own fixed-mixture and fixed-share ablations. It did **not** beat the seed ensemble (+0.0135 [+0.0082, +0.0183]), the transformer on garcia (+0.0009), the population GRU on described (+0.0022) or on the external dataset (+0.0108 vs ensemble) |
| **On what datasets?** | Psych-101 (Apache-2.0): restless 4-armed bandit (Bahrami & Navajas 2020), experience-based choice (Garcia et al. 2023), described gambles (Plonsky et al. 2018); external: Great Brain Experiment risky choice and happiness (Rutledge; CC0). All participants held out from fitting; tuning participants also used for selection |
| **By how much, with what CIs?** | See `PHASE3_RESEARCH_RESULTS.md` §2–§3; e.g. F3 ensemble − GRU ensemble −0.0014 [−0.0048, +0.0019] (restless4) |
| **At what computational cost?** | F3: ≈ 4× the fitting CPU and 7× the prediction cost of a same-size GRU; F1: 40× prediction cost; the 5-seed ensemble that beat them costs 5× a single GRU. Whole Phase III ≈ 10 CPU-hours, no GPU, no language-model calls |
| **Does the advantage survive ablations?** | F3's learned gate beats its fixed-mixture and fixed-share ablations as a single model; the advantage does **not** survive matched ensembling, and the candidates' synthetic positive-control tests fail in two environments (critic N-13). F1's meta-training equals a bolt-on latent; F2 and F4 fail their own baselines |
| **Was the evaluation preregistered?** | Yes for hypotheses, gates, margins, statistics and stop rules (thesis committed before any fit, amendments A3-1/A3-2 before any relevant result). Because no stop rule was met, **no confirmatory evaluation was run** — every number here is exploratory |
| **Was it independently reproduced?** | Partly. The critic recomputed every central contrast from committed per-participant scores (all matched) and refitted 23 single GRUs to measure seed variance; it did not refit the candidates. No external replication |
| **Is it scientifically novel or an application of existing techniques?** | The candidates are combinations of existing methods (CAVIA-style context adaptation, mixture-of-experts / Bayesian model averaging over episodic memory, Kalman-style world models with hierarchical policy priors, RL² meta-learning). The one proposed novel mechanism (F1's trait/context split with surprise-gated step sizes) was **not supported**. No novelty claim is made |
| **What evidence remains missing?** | Confirmatory tests (not warranted); the `two_step` family and the described-gambles comparator pool; the social dataset (Najar 2020) run; an external replication beyond GBE; seed-robust gate statistics; a prospective causal study (Gate G); the blocking items the critic lists for any future freeze (N-1, N-4, N-5, N-6, N-7, N-3) |
| **What can we accurately claim to investors?** | See below |

## What can be claimed — and what cannot

**Accurate claims**

* We built a leakage-hardened research platform for human behavioural prediction: frozen participant partitions,
  a test lock that has withstood the independent critic's attacks, strict causality checks that excluded all seven
  adversarial models, harness-owned ablations, gate rules in code, and full experiment registries including
  failures.
* We tested four candidate personalisation architectures and six conventional adaptation methods against strong
  population models on three Psych-101 learning tasks and one external CC0 dataset (47,067 people), and found that
  **explicit personalisation does not beat strong population models on single-session tasks** at a meaningful
  margin. This is a negative result, reported in full.
* On the external dataset, **a person's earlier sessions are worth 0.015–0.02 nats/choice** when a model cannot
  keep them in context — a measured, specific opportunity for the next phase.
* Observed momentary happiness ratings did **not** improve choice prediction on 9,413 held-out people.

**Claims we cannot make**

* That we have a breakthrough, a state-of-the-art human-behaviour model, or a model better than existing methods.
* That any model predicts an individual's future, or that any model improves real decisions.
* That simulated agents have emotions, or that the synthetic society reproduces human development.
* That results transfer to Centaur-style foundation-model comparisons (not run; Psych-101 participants were in
  Centaur's training data).

## Continuation (pivot) plan — each item needs its own preregistration

1. **Cross-session person representation (H4).** In session-based deployments (Future Self), the in-context state
   is lost between sessions. Hypothesis: a persistent, fixed-size person representation learned from earlier
   sessions recovers ≥ 0.01 of the measured 0.015–0.02 nats/choice headroom over a session-only population ensemble,
   on held-out people of an external dataset (GBE plays; a second dataset to be found). Falsifier: it recovers less
   than half the headroom, or not more than a recency-weighted choice-frequency summary does.
2. **Population-model strength.** Seed ensembles, transformers and cross-family pretraining moved NLL more than
   personalisation. A population foundation model trained across all licensed families (with external held-out
   studies) is the stronger comparator any future candidate must beat.
3. **Protocol fixes before any freeze** (critic): seed-ensembled gate arms or a seed × participant bootstrap (N-4);
   per-participant subprocess or order-invariance checks (N-1); committed access log and approval bound to the
   freeze-record hash (N-5, N-6); sealed loader before unpickling (N-7); core-built tuned twin (N-3); a
   configuration-count ledger (F-3).
4. **Gate G** remains the only route to a decision-support claim: a prospective randomised study after a model has
   passed Gates A–C (`CAUSAL_DECISION_RESEARCH.md`).
