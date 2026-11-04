# [PIPELINE VALIDATION] Exp 2: twin-society tutoring-access intervention (primary preregistered education experiment)

> **PIPELINE VALIDATION RUN** — scaled-down budget/seeds to validate code paths. Not a research result.

> Simulated result from a synthetic world. Effects may simply reflect assumptions built into the simulator and are not estimates of real-world causal effects.

- Experiment id: `exp02_twin_tutoring__pipeline_validation`  
- Preregistration hash: `69e4454129aa0f75f98b64caa16dd4b8` (written to definition.json before any run)  
- Protocol: `society`; seeds: [0, 1, 2]; arms: dev_control, dev_tutoring, rl_control, rl_tutoring, static_control, static_tutoring  
- Status: **INCONCLUSIVE** — primary contrast not computed: no data for arm(s) ['dev_tutoring', 'dev_control']

## Hypothesis and falsification criterion

**Hypothesis.** In a society of developmental learners, giving every child aged 5-15 access to tutoring (extra instructional feedback episodes) raises held-out task accuracy at ages 16-20 relative to a cloned control society, and the effect appears only when agents can learn.

**Falsification criterion.** Not supported if dev_tutoring - dev_control on cohort_accuracy_16_20 is < 0.02 or its paired 95% CI includes 0 across 10 seed pairs. The mechanism claim fails if static_tutoring - static_control falls outside +/-0.02.

**Primary metric.** `cohort_accuracy_16_20` — At simulated year 20: mean over alive agents aged 16-20 of their latest held-out assessment accuracy averaged over the training families (world-run hpe.tasks.assess, learning disabled); one value per society (seed).

## Arm summaries (mean over seeds, 95% bootstrap CI)

| arm | n seeds |  |
|---|---|

## Preregistered contrasts

| contrast | kind | treatment | control | estimate | 95% CI | relative | p | verdict |
|---|---|---|---|---|---|---|---|---|
| tutoring_effect_developmental (primary) | superiority | dev_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['dev_tutoring', 'dev_control'] |
| mechanism_check_static_persona | equivalence | static_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['static_tutoring', 'static_control'] |
| manipulation_check | superiority | dev_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['dev_tutoring', 'dev_control'] |
| tutoring_effect_plain_rl | superiority | rl_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['rl_tutoring', 'rl_control'] |
| tutoring_effect_on_transfer_family | exploratory | dev_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['dev_tutoring', 'dev_control'] |
| tutoring_effect_on_employment | exploratory | dev_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['dev_tutoring', 'dev_control'] |
| spillover_to_adults | exploratory | dev_tutoring | None | — | — | — | — | not_computed: no data for arm(s) ['dev_tutoring', 'dev_control'] |

## Compute

- Runs: 0; total wall time 0.0 s; LLM calls: 0; paid API cost: $0.

## Limitations

- PIPELINE VALIDATION RUN: scaled-down budget and seeds; not the preregistered experiment. Do not cite as a research result.
- Simulated effects reflect the simulator's assumption that tutoring = extra instructional-feedback episodes; they are not estimates of real tutoring effects.
- Cohorts after year 0 include agents born during the run; births can differ between arms through downstream dynamics.
- Depends on hpe.world semantics for tutoring_access and AgentState.tutored (sim/world lane).
- Arm 'dev_control' was NOT run: hpe.world.World not available in this build.
- Arm 'dev_tutoring' was NOT run: hpe.world.World not available in this build.
- Arm 'rl_control' was NOT run: hpe.world.World not available in this build.
- Arm 'rl_tutoring' was NOT run: hpe.world.World not available in this build.
- Arm 'static_control' was NOT run: hpe.world.World not available in this build.
- Arm 'static_tutoring' was NOT run: hpe.world.World not available in this build.
- Simulated result from a synthetic world. Effects may simply reflect assumptions built into the simulator and are not estimates of real-world causal effects.

## Reproduce

```bash
python -m hpe.evaluation run experiments/exp02_twin_tutoring.yaml --pipeline-validation
python -m hpe.evaluation analyze <out>/exp02_twin_tutoring__pipeline_validation   # regenerates result.json from raw logs
```
