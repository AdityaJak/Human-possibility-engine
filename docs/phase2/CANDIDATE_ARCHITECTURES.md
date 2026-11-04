# Candidate Architectures — Phase II

_Each candidate names the mechanism, the strongest baselines it must beat, the matched-complexity
control that would expose a trivial explanation, and what result would make us abandon it. Candidates
are developed only on dev/tune partitions (`EXPERIMENT_REGISTRY.md`)._

## Stronger baselines (built first; shared by all candidates)

| Baseline | Why | Where |
|---|---|---|
| Exemplar kNN with learned similarity kernel and recency decay (GCM/instance-based family) | best Phase I model on human MCPL | human + synthetic |
| Plain RL (Phase I `plain_rl`, re-tuned on the tune partition) | strongest synthetic baseline | both |
| GRU sequence model fitted per task family (modern sequence baseline, matched parameter count) | the mandate's "modern sequence-model baseline" | human |
| Bayesian rule learner (posterior over linear/conjunctive rules) | upper bound on rule-based learning in v2 families | synthetic |
| PVL-Delta / prospect-valence learning | standard IGT cognitive model | human IGT |
| Age-script-only control (scripted maturation, no learning mechanisms) | the H2 null | synthetic |

## Candidate A — Hybrid parametric + exemplar learner (responds to Gaps 4, 1)

**Mechanism.** Two predictors share one encoder: a parametric head (Phase I) and an exemplar head that
retrieves stored episodes by learned similarity with recency/consolidation weighting. An arbitration
unit (a small learned gate conditioned on retrieval density, parametric confidence and recent
prediction errors) mixes the two. Consolidation moves regularities from exemplars into parameters;
forgetting prunes exemplars. Uncertainty-aware action selection from the mixture.
**Must beat.** exemplar kNN, plain RL, GRU on human MCPL and on v2 synthetic families.
**Control.** Fixed 50/50 mixture (no learned arbitration) and exemplar-only ablation: if these match
the hybrid, the "learning when to rely on examples" claim is false.
**Abandon if.** On the human dev partition the hybrid is not within 0.02 nats of the exemplar model on
MCPL *and* within 0.02 of the Phase I learner on IGT after budget.

## Candidate B — Hierarchical developmental world model (responds to Gaps 1, 3)

**Mechanism.** Separate latent states for immediate observation, short-term context, stable structure
(learned transition/reward model), and persistent individual state; model-based planning for delayed
rewards where a model exists, model-free fallback otherwise; explicit uncertainty over the
environment model that drives exploration.
**Must beat.** plain RL and GRU on v2 families with delayed reward, partial observability and regime
changes.
**Control.** Model-free variant with the same capacity; a "transition-predictor-only" variant to show
that predicting transitions ≠ causal knowledge.
**Abandon if.** No v2 family shows a dev-partition advantage ≥ 0.05 after budget.

## Candidate C — Continual developmental learner (responds to Gap 3)

**Mechanism.** Skill modules with selective consolidation (importance-weighted protection of
parameters, e.g. EWC/synaptic-intelligence style), replay prioritised by surprise, representation
stability regularisers, and recovery-after-failure via retained alternatives.
**Must beat.** established continual-learning baselines (fine-tuning, EWC, experience replay) on a
lifetime curriculum of v2 families with rule changes.
**Control.** Replay-only; EWC-only.
**Abandon if.** Forgetting on earlier families is not reduced relative to replay-only at equal
performance on the current family.

## Candidate D — Modular cognitive-affective architecture (responds to Gap 8)

**Mechanism.** Affective state variables with literature-grounded dynamics — mood as a running average
of reward prediction errors (Eldar & Niv style), arousal from uncertainty, frustration from repeated
negative prediction errors, perceived control from outcome contingency — that modulate learning rate,
exploration temperature, memory retrieval weighting and effort allocation. Computational state only;
no claim of experienced emotion.
**Must beat.** the same architecture with affect variables removed, and a matched-complexity control
where the extra parameters are not tied to prediction errors (random modulators).
**Human tests.** Human datasets where mood/affect manipulations or self-reports accompany choices
(Phase II human-validation lane to find licensed ones; Psych-101 subsets with feedback-valence
manipulations as a start).
**Abandon if.** Affect variants do not beat the matched-complexity control on human data.

## Candidate E — Socially learned representations (responds to Gap 2)

**Mechanism.** Relationship formation driven by repeated interaction with instrumented tie strength
and trust updated from observed reliability (not from a reward for having friends); teaching and
demonstration that transfer *information* (labelled hard items the learner is uncertain about);
conflicting-information episodes; belief revision through weighted evidence; cooperation tasks whose
payoff depends on joint competence.
**Must beat.** isolated agents with equal practice budgets; and the Phase I social mechanism.
**Control.** "Social reward only" (agents socialise but receive no information): if this matches the
full mechanism, the effect is not learning from others.
**Measurements.** ties formed, interactions, demonstrations received, knowledge transferred (held-out
accuracy on items only ever seen via demonstration), misinformation damage and recovery.
**Abandon if.** Information-transfer measures stay at 0 or the control matches the full mechanism.

## Allocation of lanes

| Lane | Candidate(s) | Also owns |
|---|---|---|
| cognitive-architecture | A, B, C | synthetic v2 learning-lab integration |
| affective | D | affect metrics, human affect datasets (with human-validation) |
| social-developmental | E | world mechanism repair, uptake metrics |
| causal | contextual-bandit intervention selection, HTE estimation on HeartSteps | `CAUSAL_VALIDITY_REPORT.md` |
| human-validation | partitions, datasets, equal tuning protocol, sequence baseline | `HUMAN_VALIDATION_V2.md` |
| critic | falsification, leakage, matched-complexity controls | `INDEPENDENT_REVIEW_V2.md` |
| infrastructure | registry tooling, budgets, clean-tree confirmatory runs | `EXPERIMENT_REGISTRY.md` tooling |
