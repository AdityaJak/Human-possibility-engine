# Gate B family 3 (`composition`) and Gate A controls — implementation notes

_Lane `science/evaluation-v2` (candidate-free), Phase II. Spec: `GATE_B_FAMILY_3_AND_GATE_A_CONTROLS.md`.
**Everything measured here is exploratory**: dev seeds 1000–1009 only; seeds 500–509 were never run (the
shared driver refuses them). Every number comes from a committed script with a committed log under
`reports/phase2/evaluation_v2/`. The candidate's hybrid policy code (`hpe.development.hybrid*`) was not
read or modified; it is used only through `hpe.contracts.registry` and the `Policy` protocol._

## 1. What was built

### 1.1 `hpe.tasks.v2.composition` — family `composition` (Gate B family 3)

`CompositionWorld(seed, CompositionConfig)` is duck-typed to the Phase I `SymbolWorld` interface
(`sample_item`, `eval_set`, `observe`, `feedback`, `perceive`, `families_meta`, `fingerprint`,
`leakage_report`, …) so `hpe.tasks.assess` / `practice` and the learning lab run unchanged.

* **Symbols**: 12 binary (±1) features; 64 train / 32 holdout symbols drawn without replacement from the
  4096 codes; pools are redrawn (≤ 500 tries) until every family has ≥ 8 symbols of each label in each
  pool, so label-stratified sampling never starves.
* **Primitives** (per task seed, stream `tasks_v2/composition/rules`): `r1` = conjunction of 2 signed
  literals over features 0–5; `r2` = parity of 3 features over 6–11 (optionally negated). `op` ∈ {AND,
  OR, XOR, IMPLIES}. Label of a symbol: `op(r1(x), r2(x))`.
* **Families** (all 2 actions, chance 0.5): `primitive_a` (teaches `r1`), `primitive_b` (teaches `r2`),
  `composition` (single slot), `composition_pair` (2 slots: which symbol satisfies the composition).
* **Shared vs specialist control**: `shared_structure=True` composes the taught primitives;
  `shared_structure=False` composes primitives drawn from an independent stream (same symbols, same
  taught families, different composition labels; fingerprints differ). The world is self-contained: the
  "taught" primitives are its own `primitive_a` / `primitive_b` families rather than the candidate lane's
  v2 `conjunction` rules (see §6, deviation 1).
* **Sampling**: practice and eval items are label-stratified (label by coin, symbol within label); eval
  sets alternate labels so every family's 60-item holdout set is exactly 30/30; pairs are one positive
  + one negative in seeded order. Eval sets come only from the holdout pool, practice only from the
  train pool; a ledger counts practice items and holdout symbols seen in practice (0 by construction).
* **Feedback**: label noise (default 0.05) flips the told label / reward only; the world's own labels are
  untouched. Difficulty = margin proxy = number of single-feature flips that change the composed label
  (12 − flips, larger = easier; `sample_item(difficulty)` picks among 8 candidates).
* **Lab registration**: learning-lab param `task_world: "composition"` (+ `task_config` → `CompositionConfig`);
  `make_world()` builds it; v1-only probes (`zero_shot/*`, transfer) are skipped for non-v1 worlds.

### 1.2 Gate A controls

* **`feedback_permutation: bool`** (learning-lab param, default false). In every practice episode the
  learner sees its item and acts on it exactly as before; the `Feedback` it receives is computed from
  an *independently sampled* train item of the same family (stream `lab/feedback_permutation/<agent>/<family>`).
  Items, decision count and update calls are identical to the normal run (test
  `test_permuted_feedback_keeps_items_and_update_calls_identical`); only the outcome information is
  replaced by a draw from the same marginal. Demonstrations are not permuted (none are used in the
  Gate A protocol).
* **`age_script_only`** (`hpe.evaluation.baselines.AgeScriptOnlyPolicy`, registered): the `static_persona`
  readout (fixed hand-written rule per family, no learnable state) whose softmax temperature is
  `1 / (decisiveness · plasticity(age))`, with `scripted_plasticity` a verbatim copy of the Phase I
  schedule (`test_scripted_schedule_matches_phase1_schedule_exactly`). The perceptual age gate is applied
  by the world (`perceive`) for every policy, so the null sees exactly the gated input the candidates
  see. It never learns (state hash unchanged after practice; accuracy unchanged after 200 more episodes).
* **Registry**: `hpe.contracts.registry._PROVIDERS["age_script_only"] = "hpe.evaluation.baselines"` (the only
  edit to the contracts package).

### 1.3 Tests, YAML, scripts

* `tests/tasks/test_v2_composition.py` (13 tests): determinism and fingerprints; ≥ 8 symbols per label per
  pool and exactly 30/30 eval sets on dev seeds 1000–1003; holdout-only eval, train-only practice and ledger;
  labels follow `op(r1, r2)`, pairs have exactly one positive; specialist control differs only in the
  composed primitives; difficulty ordering; noise corrupts feedback only; random policy exactly 0.5; a
  least-squares probe stays < 0.9 (the measured ceiling is in §2).
* `tests/evaluation/test_gate_a_controls.py` (7 tests): schedule identity; registration + conformance;
  age-dependent sharpness without learning; permuted feedback removes plain RL's gain (3 agents, 120
  episodes); identical items/decisions under permutation; composition world through the lab (no v1-only
  metrics, zero leakage, specialist world = same symbols, different fingerprint); unknown `task_world` rejected.
* `experiments/phase2/gate_a_controls.yaml`: dev-seed `ExperimentSpec` for the arms the frozen
  `PolicyClass` enum can express (developmental ± frozen, plain_rl ± frozen, static_persona), the
  `permuted_feedback` sensitivity variant and equivalence contrasts (margin 0.02). Loads and hashes
  (`f43ce5f3…`). It cannot name `age_script_only` or `hybrid_exemplar` (§6).
* `reports/phase2/evaluation_v2/scripts/`: `_lab_runner.py` (job driver, CPU accounting, seed guard),
  `composition_ceiling.py`, `gate_a_dev.py`, `composition_dev.py`.

## 2. Measured probe ceilings for `composition` (`composition_ceiling.json`, log alongside)

Probes are fitted on the 64 train symbols and scored on the fixed balanced holdout eval sets that
`assess` uses, so each number is what a frozen policy with that readout would measure. Mean over 10 dev
seeds [95% percentile-bootstrap CI]; chance 0.5. Code `aa699ea` ("+dirty" = this untracked notes draft),
CPU 11 s.

| world | probe | primitive_a | primitive_b | composition | composition_pair |
|---|---|---|---|---|---|
| shared | lstsq_raw (linear) | 0.997 [0.990, 1.000] | 0.538 [0.473, 0.597] | 0.625 [0.575, 0.682] | 0.748 [0.650, 0.848] |
| shared | logistic_raw (linear) | 1.000 | 0.535 [0.468, 0.602] | **0.672 [0.612, 0.745]** | 0.742 [0.647, 0.840] |
| shared | logistic_quad (pairwise products) | 0.947 [0.917, 0.972] | 0.472 [0.397, 0.538] | 0.655 [0.573, 0.735] | 0.640 [0.547, 0.730] |
| shared | oracle_prims (`[r1, r2, r1·r2]`) | 1.000 | 1.000 | **1.000** | 1.000 |
| control | lstsq_raw | 0.972 [0.930, 1.000] | 0.573 [0.508, 0.635] | 0.505 [0.420, 0.587] | 0.675 [0.512, 0.828] |
| control | logistic_raw | 1.000 | 0.563 [0.483, 0.637] | 0.527 [0.437, 0.608] | 0.650 [0.495, 0.793] |
| control | logistic_quad | 0.917 [0.872, 0.955] | 0.495 [0.447, 0.548] | 0.518 [0.443, 0.598] | 0.557 [0.512, 0.600] |
| control | oracle_prims (taught primitives) | 1.000 | 1.000 | 0.583 [0.523, 0.655] | 0.642 [0.552, 0.733] |

**Reading.** The parity primitive is invisible to linear and quadratic probes (≈ 0.5), so no raw-feature
linear learner can solve `composition`; a readout over the two abstracted primitives solves it exactly in
the shared world and not in the specialist control — the intended contrast. The linear ceiling on the
composition family is **0.67 on average but operator-dependent**: with a conjunction `r1` (25 % positive),
AND and IMPLIES leave a linear shortcut on balanced eval sets (predict `r1` / `¬r1`), XOR leaves none
(`by_op` in the JSON, logistic probe, shared world: AND 0.800 (1 seed), IMPLIES 0.761 (3), OR 0.608 (4),
XOR 0.600 (2); per-seed values range 0.55–0.92). The spec's
"≈ 0.75" expectation therefore holds on average but a learner must be compared with the per-seed (or
per-op) ceiling, and a confirmatory use of the family should consider restricting `op` to {XOR, OR} or
stratifying the eval set by `r1` (§6).

## 3. Gate A controls on dev seeds (`gate_a_dev/`, exploratory)

Script `scripts/gate_a_dev.py`, log `gate_a_dev/run.log`, data `gate_a_dev/results.json`, code `70437a0`.
Phase I v1 world (families sort / compare / signal), learning-lab defaults (bandit feedback, label noise
0.05, no demonstrations, 300 episodes per family, ages 8→18, gate on), 20 agents per seed, 10 dev seeds,
common random numbers across arms; 110 lab runs, 11.0 CPU-min. Frozen copies: `developmental`
`use_learning=false`, `hybrid_exemplar` `use_learning=false`, `plain_rl` `lr=0.0`. The script first verifies
through the Policy protocol that every frozen arm and both nulls leave their held-out action
probabilities unchanged after 60 full-feedback practice episodes and that every learner changes them
(all eight checks passed; `plain_rl_frozen` changes its state hash only through update counters).

Final held-out accuracy (mean over families and agents; mean over seeds [95 % CI]); tick-0 accuracy is
0.444 for every learner (0.440 for the two nulls):

| arm | main | permuted feedback |
|---|---|---|
| developmental | 0.832 [0.816, 0.849] | 0.437 [0.431, 0.443] |
| developmental_frozen | 0.444 | — |
| hybrid_exemplar | 0.830 [0.815, 0.849] | 0.433 [0.424, 0.442] |
| hybrid_exemplar_frozen | 0.444 | — |
| plain_rl | 0.814 [0.787, 0.840] | 0.446 [0.436, 0.454] |
| plain_rl_frozen | 0.444 | — |
| age_script_only | 0.440 [0.429, 0.453] | — |
| static_persona | 0.440 [0.429, 0.453] | — |

Gains, paired over seeds (paired percentile bootstrap, 4000 resamples; Cohen dz; sign-flip p):

| learner | frozen-copy gain (expected ≥ 0.10) | permuted-feedback gain (expected ≤ 0.02) | age-script-only gain (required ≥ 0.10) |
|---|---|---|---|
| developmental | **+0.388 [+0.372, +0.404]**, dz 14.0 | **−0.007 [−0.014, −0.001]**, dz −0.67 | **+0.392 [+0.374, +0.412]**, dz 12.3 |
| hybrid_exemplar | **+0.386 [+0.371, +0.404]**, dz 13.8 | **−0.011 [−0.020, −0.003]**, dz −0.73 | **+0.390 [+0.371, +0.409]**, dz 11.9 |
| plain_rl | **+0.369 [+0.343, +0.396]**, dz 8.2 | **+0.001 [−0.008, +0.009]**, dz 0.10 | **+0.373 [+0.341, +0.404]**, dz 7.0 |

Per-family gains are in `gate_a_dev/report.md` (every frozen-copy and age-script gain ≥ +0.35 on every
family; every permuted gain within [−0.02, +0.02] except developmental/sort −0.019 [−0.034, −0.005]).

**Reading (dev, exploratory).** On the v1 families all three learners show the Gate A pattern: the gain
disappears when learning is frozen, disappears when the outcome information is permuted (the learners
end slightly *below* their frozen copies — learning from permuted feedback is mildly harmful, which is
the expected sign), and is ≈ 0.39 over the age-script null. The two-sided ±0.02 equivalence check is met
by developmental and plain_rl; hybrid_exemplar's lower CI bound sits at −0.020 (the spec's criterion is
one-sided, "gain ≤ 0.02", which all three meet with room). This does not distinguish the candidate
from plain RL — Gate A is a necessary check, and these are the easy Phase I families.
**Caveat on the H2 null.** `age_script_only` and `static_persona` have identical accuracy: a scripted
plasticity schedule acting as a temperature cannot change the readout's argmax, and the perceptual gate
is applied to every policy by the world, so on *accuracy* the age-script null adds nothing to the static
persona (it differs in log loss / calibration, which Gate A does not score). A null whose scripted
maturation changes *which* heuristic is used would be a stronger H2 null; that is a design decision for
the orchestrator (§6).

## 4. `composition` on dev seeds (`composition_dev/`, exploratory)

Script `scripts/composition_dev.py`, log `composition_dev/run.log`, data `composition_dev/results.json`
(code `70437a0`, "+dirty" only because the preceding run's untracked outputs and the per-op edit of the
ceiling script were present). Learning lab with `task_world: composition`, families primitive_a /
primitive_b / composition / composition_pair interleaved in blocks of 10, 300 episodes per family, bandit
feedback, label noise 0.05, 20 agents per seed, 10 dev seeds, shared world and specialist control;
80 lab runs, 15.4 CPU-min. Final held-out accuracy (mean over agents; mean over seeds [95 % CI]); tick-0
is 0.50 everywhere; practice never touched a holdout symbol (ledger 0 in every run).

| world | arm | primitive_a | primitive_b (parity) | composition | composition_pair |
|---|---|---|---|---|---|
| shared | plain_rl | 0.912 [0.893, 0.929] | 0.526 [0.467, 0.578] | 0.701 [0.627, 0.775] | 0.739 [0.646, 0.837] |
| shared | memory_only | 0.892 [0.872, 0.909] | 0.595 [0.527, 0.668] | 0.684 [0.605, 0.764] | 0.741 [0.645, 0.837] |
| shared | developmental | 0.965 [0.955, 0.973] | 0.516 [0.479, 0.548] | 0.704 [0.631, 0.784] | 0.733 [0.640, 0.831] |
| shared | hybrid_exemplar | 0.976 [0.967, 0.984] | 0.628 [0.583, 0.674] | 0.709 [0.632, 0.790] | 0.734 [0.640, 0.834] |
| control | plain_rl | 0.910 [0.891, 0.928] | 0.535 [0.485, 0.579] | 0.643 [0.526, 0.760] | 0.681 [0.537, 0.812] |
| control | memory_only | 0.879 [0.854, 0.901] | 0.577 [0.520, 0.640] | 0.656 [0.565, 0.744] | 0.682 [0.550, 0.797] |
| control | developmental | 0.948 [0.931, 0.964] | 0.526 [0.489, 0.562] | 0.602 [0.513, 0.688] | 0.656 [0.514, 0.789] |
| control | hybrid_exemplar | 0.958 [0.942, 0.971] | 0.611 [0.566, 0.659] | 0.627 [0.545, 0.705] | 0.655 [0.519, 0.781] |

Shared − control, paired over seeds: composition — developmental +0.102 [+0.016, +0.175] (dz 0.77),
hybrid_exemplar +0.082 [+0.001, +0.157] (dz 0.60), plain_rl +0.058 [−0.029, +0.145], memory_only
+0.028 [−0.049, +0.101]; composition_pair — all four between +0.058 and +0.079 with CIs including 0.

**Reading (dev, exploratory).** Every learner masters the conjunction primitive (0.89–0.98) and none
learns the parity primitive (0.52–0.63; the hybrid is best at 0.63 but far from 1.0). Consequently every
learner sits at 0.68–0.71 on `composition` in the shared world, i.e. at the level of the measured linear
shortcut (0.672 mean; §2), and **no arm's CI lies above the linear ceiling** — by the spec's own rule the
family is *uninformative for these learners at this budget* and is reported as such, not dropped. The
shared-minus-control differences for the two developmental learners are positive with CIs just clear of
zero, but at the size of the linear shortcut (the composed labels in the control world are built from
primitives the learner never practised, so even the `r1` shortcut is lost there) — they are not evidence of
composing abstracted primitives. In the control world memory_only and hybrid_exemplar exceed the (lower)
control ceiling 0.527, which only says exemplar storage captures some non-linear structure of the
composed labels from the 64 train symbols. What would make the family informative: a learner that
acquires `primitive_b` (parity); until one does, `composition` cannot separate "composes abstractions"
from "uses the linear shortcut".

## 5. CPU time (budget 2 CPU-h)

Per-job CPU time is measured inside each worker process (`time.process_time`) and summed; the logs hold
the per-run values.

| item | CPU | wall (4 workers) |
|---|---|---|
| `composition_ceiling.py` (two runs) | 25 s | 25 s |
| `gate_a_dev.py` — 110 lab runs × 20 agents | 658 s (11.0 min) | 346 s |
| `composition_dev.py` — 80 lab runs × 20 agents | 925 s (15.4 min) | 763 s |
| smoke runs (2 seeds × 2 agents, outputs discarded), timing calibration | ≈ 45 s | — |
| lane test gate (one full run ≈ 40 s; several partial runs) | ≈ 4 min | — |
| **total** | **≈ 0.5 CPU-h** of the 2 CPU-h budget | |

## 6. Known gaps, deviations and contract-change requests

1. **Deviation — self-contained primitives.** The spec says `r1`, `r2` are "shared with the `conjunction`
   family's latent rules". The candidate's lane authored and viewed dev results on the v2 `conjunction`
   family; coupling Gate B family 3 to those rules would reintroduce exactly the exposure the family is
   meant to remove, and would require this lane to read that lane's task code. The composition world
   therefore teaches its own primitives (`primitive_a`, `primitive_b`) and the shared/specialist
   manipulation is made inside the world. If the orchestrator prefers the literal coupling, it is a
   config-level change (`Primitives` from an external rule) and the tests still apply.
2. **Contract-change request (`hpe.contracts.schemas.PolicyClass`).** *What*: add `"age_script_only"` and
   `"hybrid_exemplar"` members. *Why*: `ArmDefinition.policy_class` is this enum, so neither the YAML runner
   nor `RunManifest` can name those arms; the full Gate A protocol had to be run by a script. *Backward
   compatible*: yes (additive enum members; existing values unchanged).
3. **Runner request (`hpe.evaluation.runner.execute_job`, not in this lane's write set).** The learning-lab
   task hash is computed as `SymbolWorld(seed, TaskWorldConfig(**task_config)).fingerprint()`, which
   raises for a composition `task_config`; it should use `learning_lab.make_world(seed, p).fingerprint()`.
   Until then the composition family runs only through scripts, not through `python -m hpe.evaluation run`.
4. **Transfer probe.** The lab's few-shot transfer probe is v1-only (`transfer_modes` act on
   `SymbolWorld`); no few-shot transfer was measured on `composition`.
5. **Operator-dependent ceiling** (§2): with op ∈ {AND, IMPLIES} a linear reader of `r1` alone scores
   ≈ 0.9 on balanced eval sets. The spec fixes the op set, so this is reported, not changed.
6. **Frozen copy of `plain_rl`** is `lr: 0.0` (its state hash still changes through the `n_updates`
   counters, but its held-out action probabilities do not; the script's freeze check verifies the
   latter for every arm before running).
7. **Budget choices**: Gate A dev protocol uses the lab defaults (bandit feedback, no demonstrations,
   no instructional feedback) so that outcome information is the only teaching signal the permutation
   can remove; transfer and retention probes are skipped (not needed for the three gains).

## 7. Lane gate (`ruff check . && ruff format --check . && mypy && pytest -q tests/tasks tests/evaluation tests/contracts`)

Run from the worktree root at the final commit of this lane (2026-11-04):

```
$ ruff check .
All checks passed!
$ ruff format --check .
191 files already formatted
$ mypy
Success: no issues found in 87 source files
$ pytest -q tests/tasks tests/evaluation tests/contracts
166 passed in 33.41s
```

The 166 tests include the 13 composition tests and the 7 Gate A control tests added by this lane; the
rest are the pre-existing tasks / evaluation / contracts suites, unchanged.
