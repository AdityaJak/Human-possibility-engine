# [PIPELINE VALIDATION] Ablations: which developmental components produce held-out skill?

> **PIPELINE VALIDATION RUN** — scaled-down budget/seeds to validate code paths. Not a research result.

> Simulated result from a synthetic world. Effects may simply reflect assumptions built into the simulator and are not estimates of real-world causal effects.

- Experiment id: `exp_ablations__pipeline_validation`  
- Preregistration hash: `76304d1e1f1c7c30161edfc23ff9935d` (written to definition.json before any run)  
- Protocol: `learning_lab`; seeds: [0, 1, 2]; arms: developmental, ablate_learning, ablate_feedback, ablate_memory, ablate_shared_representation, ablate_social_learning, ablate_plasticity_schedule, ablate_curriculum, ablate_forgetting  
- Status: **INCONCLUSIVE** — primary contrast not computed: no data for arm(s) ['developmental', 'ablate_learning']

## Hypothesis and falsification criterion

**Hypothesis.** Held-out skill comes from experience-driven parameter learning, not from scripted age gates: freezing policy learning or withholding feedback removes the gain; other components contribute measurably or not at all.

**Falsification criterion.** The learning claim is refuted if freezing policy learning (ablate_learning) does not reduce held-out accuracy by >= 0.10 with a paired 95% CI excluding 0 — that would mean measured skill comes from scripted mechanisms.

**Primary metric.** `final_heldout_accuracy` — Mean held-out accuracy over training families after 300 episodes/family + 30 teacher demonstrations/family.

## Arm summaries (mean over seeds, 95% bootstrap CI)

| arm | n seeds |  |
|---|---|

## Preregistered contrasts

| contrast | kind | treatment | control | estimate | 95% CI | relative | p | verdict |
|---|---|---|---|---|---|---|---|---|
| learning_is_necessary (primary) | superiority | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_learning'] |
| feedback_is_necessary | superiority | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_feedback'] |
| memory_contribution | exploratory | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_memory'] |
| shared_representation_contribution | exploratory | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_shared_representation'] |
| social_learning_contribution | exploratory | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_social_learning'] |
| plasticity_schedule_contribution | exploratory | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_plasticity_schedule'] |
| curriculum_contribution | exploratory | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_curriculum'] |
| forgetting_effect_on_retention | exploratory | developmental | None | — | — | — | — | not_computed: no data for arm(s) ['developmental', 'ablate_forgetting'] |

## Ablations (ablated arm minus full learner, primary metric)

| ablation | estimate | 95% CI | n |
|---|---|---|---|
| ablate_learning | — | — | no data for arm(s) ['ablate_learning', 'developmental'] |
| ablate_feedback | — | — | no data for arm(s) ['ablate_feedback', 'developmental'] |
| ablate_memory | — | — | no data for arm(s) ['ablate_memory', 'developmental'] |
| ablate_shared_representation | — | — | no data for arm(s) ['ablate_shared_representation', 'developmental'] |
| ablate_social_learning | — | — | no data for arm(s) ['ablate_social_learning', 'developmental'] |
| ablate_plasticity_schedule | — | — | no data for arm(s) ['ablate_plasticity_schedule', 'developmental'] |
| ablate_curriculum | — | — | no data for arm(s) ['ablate_curriculum', 'developmental'] |
| ablate_forgetting | — | — | no data for arm(s) ['ablate_forgetting', 'developmental'] |

## Compute

- Runs: 0; total wall time 0.0 s; LLM calls: 0; paid API cost: $0.

## Limitations

- PIPELINE VALIDATION RUN: scaled-down budget and seeds; not the preregistered experiment. Do not cite as a research result.
- Curriculum only affects self-directed practice choice; in this fixed-schedule lab it may have no effect by design.
- Ablation flags are implemented by the research/development lane; this experiment only toggles them.
- Arm 'ablate_curriculum' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_feedback' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_forgetting' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_learning' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_memory' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_plasticity_schedule' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_shared_representation' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'ablate_social_learning' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Arm 'developmental' was NOT run: policy class 'developmental' not registered in this build (KeyError: "no policy registered for 'developmental'; known: ['memory_only', 'plain_rl', 'random', 'static_persona']").
- Simulated result from a synthetic world. Effects may simply reflect assumptions built into the simulator and are not estimates of real-world causal effects.

## Reproduce

```bash
python -m hpe.evaluation run experiments/exp_ablations.yaml --pipeline-validation
python -m hpe.evaluation analyze <out>/exp_ablations__pipeline_validation   # regenerates result.json from raw logs
```
