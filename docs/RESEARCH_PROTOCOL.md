# Research protocol

## 1. Preregistration

Every experiment is a YAML `ExperimentDefinition` in `experiments/` stating hypothesis, falsification
criterion, arms, control arm, seeds, primary metric with its exact definition, comparison, test, alpha
and minimum effect of interest. The runner computes `preregistration_hash = stable_hash(definition)`
and writes `definition.json` **before** any simulation runs. Results reference that hash. Changing the
definition changes the hash; old results are never relabelled.

## 2. Matched comparisons

* **Common random numbers.** Arms share `population_seed` and `task_seed`; per-agent, per-purpose RNG
  streams mean an intervention on one agent does not perturb other agents' randomness.
* **Equal inputs.** In learning-lab experiments, every policy class sees the identical item and
  feedback stream (streams keyed by seed, agent index and episode index, never by arm).
* **Equal budget.** Same episodes, same feedback; compute counters and wall time reported per class.

## 3. Measurement

* Skill = `hpe.tasks.assess` accuracy on held-out symbols (never seen in practice), learning disabled,
  ties credited fractionally (uniform policy = chance exactly). Log loss and expected accuracy are secondary.
* Transfer = `hpe.tasks.transfer_probe` few-shot curve on the never-practised `barter` family, on a clone.
  Control: `transfer_mode="orthogonal"` where a shared representation should not help.
* Leakage checks: train/holdout symbol disjointness, eval-set fixity, no-mutation during assessment.

## 4. Statistics

* Unit of analysis: the seed (one society or one learning-lab cohort per seed) unless stated.
* Primary: paired bootstrap (≥2,000 resamples, seeded) 95% CI of the mean paired difference; secondary
  tests Holm-corrected. Effect sizes: Cohen's dz (paired) / Hedges' g (unpaired).
* Status rules: **passed** = CI excludes 0 in the predicted direction AND the preregistered minimum
  effect is met; **failed** = CI excludes the minimum effect or points the wrong way;
  **inconclusive** otherwise.
* Ablations and sensitivity sweeps are reported in full, including those that weaken the result.

## 5. Reporting

Analysis regenerates every number and figure from raw logs on disk (`reports/experiments/<id>/runs`),
never from in-memory objects or hand entry. Each report carries the synthetic disclaimer and lists
limitations. Negative results are published with equal prominence (`docs/CLAIMS_POLICY.md`).

## 6. Experiments

| ID | Question | Primary metric |
|---|---|---|
| exp01_acquired_learning | Does feedback condition (none/bandit/full) change held-out skill acquisition? | held-out accuracy after fixed practice budget |
| exp_baselines_heldout (H1) | Does the developmental learner beat static persona, memory-only and plain RL? | mean held-out accuracy over training families |
| exp_ablations | Which components matter? | Δ held-out accuracy vs full learner |
| exp06_transfer (H3) | Does a shared representation speed learning of an unseen family? | AUC of barter few-shot curve, shared vs specialist, plus orthogonal control |
| exp02_twin_tutoring (H2, primary intervention) | Does tutoring access change measured skill through learning? | held-out accuracy of treated cohort at 16–20; mechanism check with non-learning policy |
| exp03_social_influence | Isolation vs connected communities; misinformation robustness | held-out accuracy; misinformation damage |
| exp04_midlife_education | Do adults adapt when education opens at 40? | Δ accuracy after opening vs control |
| exp05_multigenerational | Inherited initial conditions vs acquired behaviour (cross-fostering) | parent–child skill association by household of rearing vs birth |
| exp07_human_heldout | Do policy classes predict held-out human choices (Psych-101 bandits)? | choice negative log-likelihood on held-out participants |
| exp07b_causal_grounding | Can we recover a randomised effect from HeartSteps V1? | effect estimate + CI vs published estimate |
