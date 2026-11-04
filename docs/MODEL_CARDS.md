# Model cards

All models below are small, inspectable, CPU-only, and trained only on synthetic data generated in this
repository (except where an evaluation on licensed human data is stated). None is trained on personal data.

## Developmental learner (`hpe.development`, policy_class `developmental`, v0.1.0)

* **Architecture.** Shared linear symbol encoder (k̂ = 4 × 12 features) with per-family linear heads
  (sort 3-way, signal 2-way, pairwise scoring for compare/barter); online SGD with cross-entropy under
  instructional feedback and an unbiased reward-based update under outcome-only feedback; bounded
  replay memory with age-dependent capacity; Beta mastery self-estimates driving exploration and
  practice choice; trust-weighted social learning with per-source credibility; Bayesian linear bandits
  for activity and trust-game decisions. Nine ablation flags.
* **Scripted assumptions.** Plasticity schedule (higher learning rate in childhood, lower in old age),
  forgetting toward initial parameters for unpractised families, perceptual age gate (shared with
  all classes). The plasticity schedule *reduced* held-out accuracy in ablation (+0.012 when removed).
* **Intended use.** Research comparison within this protocol. Not a model of any person.
* **Evaluation.** `RESEARCH_RESULTS.md` §2–§4. Held-out accuracy 0.849 after 300 episodes/family;
  transfer AUC 0.742 (shared world), 0.552 (orthogonal, negative transfer).
* **Known weaknesses.** No within-family advantage over plain RL; fragile to label noise 0.15; slower to
  criterion; hyperparameters chosen on task seeds 7 and 21 (seed 7 overlaps evaluation seeds).
* **Hyperparameter provenance.** 6-config sweep by the research lane; not committed as a script (audit M-4).

## Plain RL baseline (`hpe.evaluation.baselines`, `plain_rl`)

Independent per-family online linear learners on raw features; linear bandit for activity/social.
Learning rate 0.05 tuned on task seeds 1001–1003 (never evaluation seeds). Strongest baseline on
within-family held-out skill (0.837). No transfer mechanism (transfer gain exactly 0 by construction).

## Memory-only baseline (`memory_only`)

Bounded episodic store of (family, context, action, outcome) with a fixed similarity-weighted retrieval
rule; stores observed demonstrations; no parametric learning. The "memory-based chatbot" analogue.
Held-out 0.773; largest train–holdout gap (0.170) — memorises rather than generalises.

## Static persona baseline (`static_persona`)

Fixed trait-derived heuristic readout; never updates. The "written biography" analogue. Held-out 0.438
(chance 0.444). Shows exactly zero response to any intervention — the mechanism-check control.

## Novel-symbol task world (`hpe.tasks.SymbolWorld`)

96 random ±1 feature vectors (12-d) with generated nonsense names; 64 train / 32 holdout; four families
whose correct answers are linear readouts of a shared 3-d latent projection (orthogonal projection for
the control world). Label noise corrupts feedback only. Limitation: linear rules make a raw-feature
linear learner near-optimal within a family (ADR-0003).

## Society kernel (`hpe.world.World`)

Deterministic event-driven state machine: households, 30-day ticks, school 6–18, tutoring, caregiving
demonstrations, self-study, jobs tied to task families, abstract wages and consumption, Gompertz
mortality, partnerships and births, shocks, eight intervention kinds, yearly held-out assessments.
Every parameter is an assumption (`packages/hpe/world/ASSUMPTIONS.md`); nothing is calibrated to a
population. Agents at year 0 receive a condensed seeded practice prehistory (≤400 episodes).

## Reference cognitive models used on human data (`hpe.evaluation.human`)

Fitted-bias, delta-rule softmax, instance-based (kNN) and population-static models, fitted on training
participants only with small preregistered grids; several fits sat at grid edges (possible under-fit).
Used solely as references for held-out human choice prediction.
