# Preregistered experiments

Each YAML file is one preregistered experiment: a frozen `ExperimentDefinition` (hypothesis, falsification
criterion, arms, seeds, primary metric) plus a `protocol` (how arms run, which contrasts are judged, sensitivity
variants). The preregistration hash covers both and is written to `definition.json` **before** any run; the runner
refuses to write into a directory whose stored hash differs.

| File | Protocol | Question | Primary contrast |
|---|---|---|---|
| `exp_baselines_heldout.yaml` | learning_lab | H1 / investor question (§44): does a developing agent beat persona, memory-only and plain-RL agents on held-out tasks? | developmental vs strongest baseline, ≥10% relative, CI > 0 |
| `exp01_acquired_learning.yaml` | learning_lab | Exp 1: identical agents randomised to none / outcome / instructional feedback | dev_full − dev_none ≥ 0.05 |
| `exp06_transfer.yaml` | learning_lab | Exp 6 / H3: transfer to the never-practised `barter` family; orthogonal negative-transfer control | shared representation − specialists ≥ 0.03 AUC |
| `exp_ablations.yaml` | learning_lab | Which components produce skill? (memory, feedback, learning, shared rep, social, plasticity, curriculum, forgetting) | learning frozen drops accuracy ≥ 0.10 |
| `exp02_twin_tutoring.yaml` | society | Exp 2 (PRIMARY education-access intervention): twin societies, tutoring for ages 5-15 | dev_tutoring − dev_control at ages 16-20 ≥ 0.02; static-persona mechanism check |
| `exp03_social_influence.yaml` | society | Exp 3: connected vs isolated; misinformation robustness | connected − isolated > 0 |
| `exp04_midlife_education.yaml` | society | Exp 4: education opening at ages 38-45 | midlife − control ≥ 0.02 with pre-period balance |
| `exp05_multigenerational.yaml` | society | Exp 5: cross-fostering separates inherited initial conditions from acquired transmission | rearing-household correlation, learner vs non-learner |

## Commands

```bash
python -m hpe.evaluation list
python -m hpe.evaluation run experiments/exp_baselines_heldout.yaml --workers 4       # preregistered run
python -m hpe.evaluation run experiments/*.yaml --pipeline-validation --seeds 3       # scaled-down code check
python -m hpe.evaluation analyze reports/experiments/exp_baselines_heldout           # regenerate from raw logs
python -m hpe.evaluation contamination --out reports/experiments/contamination_audit.json
```

## Verdict rules (applied by `hpe.evaluation.analysis`, fixed before data)

* **superiority** — passed if the point estimate ≥ threshold (absolute, or relative × |control mean|) **and** the
  95% CI lower bound > 0; failed if the CI upper bound < threshold; otherwise inconclusive.
* **equivalence** (margin m) — passed if the CI lies inside (−m, m); failed if it lies entirely outside; else inconclusive.
* **non_inferiority** (margin m) — passed if the CI lower bound of the improvement > −m.
* **exploratory** — reported, never counted as a pass or fail.
* The experiment status is the verdict of the single `primary` contrast. Secondary confirmatory p-values are
  Holm-adjusted. A `strongest_of` control is resolved to the best-scoring baseline (conservative for the hypothesis).
* Unit of analysis: one value per seed (independent task world + population); paired by seed.

## Matching and budget

Learning-lab arms share agent initial conditions and the exact item / feedback-noise / teacher-demonstration
streams (keyed by seed, agent and family — never by arm); each arm gets the same number of episodes and the same
assessment items. Society arms are clones (equal `seed`, `population_seed`, `task_seed`; per-agent RNG streams).

Baseline hyperparameters (plain_rl lr = 0.05, REINFORCE with running baseline) were selected on task seeds
1001-1003 from {REINFORCE, reward regression} × lr {0.02, 0.05, 0.1, 0.2, 0.4}; evaluation seeds are 0-9.
