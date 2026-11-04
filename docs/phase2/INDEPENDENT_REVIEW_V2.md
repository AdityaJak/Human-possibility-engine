# Independent Review v2 — Phase II

_Independent Scientific Critic. I authored none of the material under review. Each finding carries a
severity (blocking / high / medium / low), a file:line reference, and the change required. Later passes
append sections; nothing here is edited retroactively._

## Pass 1 — Design review (2026-11-04, before any confirmatory experiment)

### 0. What I re-verified myself

| Check | Result | Artifact |
|---|---|---|
| Regenerate partitions with the committed script and compare | identical file; test SHA-256 `7dc2e7798f57d3fa…` reproduced; recomputed independently from the id lists; 726 test keys; dev/tune/test disjoint and exhaustive in all 23 experiment files | `reports/phase2/critic/guard_run_1.json` (partitions skipped by design), this section |
| X-002 used only dev+tune participants | recomputed eligible set: 140 MCPL / 307 IGT dev+tune participants; participant×split rows with both models = **125 / 274**, matching the diagnosis | `reports/phase2/critic/x002_recheck.json` |
| Scan the whole repository for test-partition keys and confirmatory seeds | **clean**: 2,181 files, 0 violations. Disclosed Phase I contact: `reports/human_validation/exp07_human_heldout/fits.json` (204 distinct test keys, 612 occurrences) and `…/outcomes.jsonl.gz` (200 keys, 6,654 rows): per-participant NLL/accuracy of every Phase I model for **63 MCPL and 137 IGT test participants** is readable from the repo | `reports/phase2/critic/guard_run_1.json` |
| Budgets | all six lanes at 0 CPU-h spent (no Phase II run manifests yet) | `reports/phase2/critic/budget_run_1.json` |

### 1. Gate thresholds (`docs/phase2/RESEARCH_THESIS_V2.md:58-63`)

| # | Severity | Finding | Required change |
|---|---|---|---|
| G1 | blocking | **Gate A is not a Phase II discriminator** (`:58`). "Held-out gain ≥ 0.10 vs a frozen copy" is passed by every learner including Phase I's (+0.41) and by any online linear model; it cannot fail. | Keep A but add the attribution arm: a *permuted-feedback control* (same number of updates, labels shuffled within family) must gain ≤ 0.02, and the *age-script-only control* (Candidate table, `CANDIDATE_ARCHITECTURES.md:16`) must not match the candidate within 0.02. |
| G2 | high | **Gate B's "≥ 0.05 absolute OR ≥ 10% relative" is looser than Phase I at the accuracies that matter** (`:59`). At 0.84 accuracy, Phase I's threshold was 0.084; 0.05 absolute is a relaxation with no stated reason. | Use max(0.05 absolute, 10% relative), or state the reason (lower ceilings in v2 families) in the preregistration. |
| G3 | blocking | **"≥ 2 of 3 families" with families chosen by the candidate's own lane** (`:59` + the cognitive lane owning `packages/hpe/tasks/v2/`). The environment designer and the candidate designer are the same lane; selecting which 3 families count after seeing dev results is a garden of forking paths. | The three gate families and the specialist-should-win control are named in the preregistration *by the Director with Critic sign-off* before any candidate dev result on them is viewed; at least one family is authored by a lane that owns no candidate; all v2 families run on dev are reported, not only the three. |
| G4 | high | **Gate D's "best fitted model" is undefined as to where it is selected** (`:61`). Selected on test, the comparator is biased in the candidate's favour or against it unpredictably. | Comparator = best model *by tune-partition NLL* (primary); best-by-test reported as a conservative secondary. The comparator set must include, by name, the Phase I winner per family (instance-based on MCPL, developmental on IGT) and `memory_only` (see §4). |
| G5 | high | **Gate D's "≥ 3 families from ≥ 2 datasets" cannot currently be met** (only Psych-101 is licensed and parsed). Fine as a bar, but the status page must not be able to show "passed" on one dataset. | `BREAKTHROUGH_STATUS.md` rows for D and F may only change from a confirmatory-ledger row that names ≥ 2 datasets; the guard will refuse a freeze record for D with one dataset. |
| G6 | medium | **Multiplicity** across ≥ 5 candidates × 3 families × 2 gates is uncontrolled. | Pre-register the number of candidates that will be evaluated confirmatorily; Holm across candidates within a gate; one confirmatory shot per candidate class per partition (see §6). |
| G7 | medium | Gate C (`:60`) does not fix the shot set or require the zero-shot leak check. | Preregister the shot grid and require zero-shot accuracy = chance (±0.02) on the transfer family. |
| G8 | low | Stop rules (`:70`) use "fails to beat the strongest baseline" without a margin; a +0.001 dev result keeps a candidate alive indefinitely. | Stop rule = fails the Gate B margin on dev after budget. |

Thresholds are pre-specified and falsifiable; apart from G2 I do not find them tuned to Phase I outcomes.

### 2. Partition scheme (`docs/phase2/EXPERIMENT_REGISTRY.md:12-35`, `scripts/phase2_partitions.py:23,63`)

| # | Severity | Finding | Required change |
|---|---|---|---|
| P1 | blocking | **Procedural protection is insufficient against in-memory use.** The guard detects test keys that are *written*; a lane can load test participants, compute an aggregate and report only the number. Nothing in the repo prevents that except the (not yet merged) harness lock. | Every Phase II evaluation output must carry a `participants` list of keys used; the harness refuses `test` without a freeze record (`EXPERIMENT_REGISTRY.md:79-81`) **in code**, with a unit test; the Critic re-runs every confirmatory evaluation from the frozen config. Guard extension (next pass): `check-run-manifests`. |
| P2 | high | **Phase I artifacts already expose test-participant performance** (§0: 63 MCPL, 137 IGT test participants scored by all Phase I models in `reports/human_validation/exp07_human_heldout/`). Any Phase II design decision informed by *which participants* Phase I models found hard is contamination. The registry's contact disclosure (`:25-31`) understates this. | State the numbers in the registry; forbid per-participant use of Phase I human artifacts in Phase II development (aggregate use only, logged as exploratory); make the **primary Gate D family a pristine one** (bandits/horizon/reversal/experience-based), with MCPL and IGT secondary. |
| P3 | high | **Tiny files make 40/20/40 meaningless** (`phase2_partitions.py:63`): wilson2014 exp2 n=7 → test 3, tune 1; igt exp2 n=13 → test 5; speekenbrink n=23 → tune 5, test 9. A participant-bootstrap CI over 3 people is noise, and "select on tune" with 1 participant is selection on noise. | Minimum-n rule: a family contributes to Gate D only with pooled test n ≥ 30 and pooled tune n ≥ 15 (pool across experiment files within the family); files below n = 20 are reported but never gated; `probability_learning` (n = 23 total) is excluded from gating unless merged with a sister family. |
| P4 | medium | Partition fractions are per experiment file, but the gate is per family; the stratification unit and the gating unit differ, so family-level test counts are not controlled. | Report family-level pooled counts in the partitions file (the guard can compute them). |
| P5 | medium | Seeds for synthetic confirmatory runs (`:14`, 500–509) are protected only by the guard's manifest scan; a run that never writes a manifest is invisible. | Require manifests for every Phase II run (the learning-lab runner already writes them); CI runs `phase2_guard.py check-test-access`. |

What a sceptical reviewer will require for Gate D, in this order: a second dataset with verified licence; the primary family pristine from Phase I; comparator chosen on tune; lapse-free NLL and ECE alongside lapse-mixed NLL; the participants list of every run; the Critic's independent re-run.

### 3. Equal-tuning protocol (`docs/phase2/EXPERIMENT_REGISTRY.md:43-46`; `packages/hpe/evaluation/human/harness.py:127,163-170`)

| # | Severity | Finding | Required change |
|---|---|---|---|
| T1 | blocking | **Phase I selected lapse and hyper-parameters on the *fitting* participants** (`harness.py:163-170`: the objective is mean NLL over `train`, the same list the reference models are fitted on at `:127`). Carried into Phase II unchanged, "tune" is just more of "train". | Fit on dev, select on tune, score on test; three disjoint lists in the harness signature; unit test that the selection objective never sees test keys. |
| T2 | high | **Equal grid *size* is not equal *fairness*** (`:44`). Grids authored by the candidate's lane can be dense for the candidate and coarse for references. The Phase I references on IGT sat at grid edges (β = 6.0 for instance-based and delta rule; lapse 0.3) — under-tuned references inflate the candidate. | Grids are authored by the human-validation lane, span literature ranges for reference models, are committed with hashes before any dev run, and any edge fit triggers one preregistered grid extension *on tune* before test. |
| T3 | high | **Ablation flags are hyper-parameters.** A candidate with 9 flags set from dev results has an effective search space far larger than 24 points. | Flags frozen before tuning and listed in the freeze record; a candidate with different flag settings is a different candidate (multiplicity, G6). |
| T4 | medium | **Lapse mixture rescues miscalibrated models and caps well-calibrated ones** (lapse floor: 0.31 nats at λ = 0.3 with 9 actions; 0.255 at λ = 0.3 with 4 actions). The Phase I learner carries λ = 0.3 on MCPL; the IGT references carry λ = 0.3 (see §4). | Report NLL with fitted lapse *and* with λ = 0; report ECE; treat an edge lapse as a calibration failure to be explained, not a fit. |
| T5 | medium | **Information asymmetry between model classes.** Reference models are fitted on other participants (`:127`); policy classes learn only within a participant. Neither is wrong, but Gate D mixes them. | Label every model `individual-only` or `population-informed`; report Gate D within each class and overall. |
| T6 | low | GRU early stopping on tune plus lapse selection on tune is two-stage selection on the same data; acceptable if test is untouched, but the number of tune evaluations should be logged. | Log tune-evaluation counts per model. |

### 4. X-002 diagnosis (`docs/phase2/SCIENTIFIC_GAP_ANALYSIS.md:53-58, 69-76`) — alternative explanations and my check

I recomputed, on dev+tune participants only, each model's NLL per predictability tercile and the lapse floors (`reports/phase2/critic/x002_recheck.json`):

| MCPL tercile (by instance-based top-1) | n | developmental | instance-based | **memory_only** | plain RL |
|---|---|---|---|---|---|
| least predictable (top-1 0.32) | 43 | 2.098 | 2.067 | 2.206 | 2.020 |
| middle (0.59) | 40 | 1.596 | 1.403 | 1.446 | 1.611 |
| most predictable (0.83) | 42 | 1.469 | 0.981 | **0.847** | 1.657 |

| IGT tercile | n | developmental | instance-based | memory_only | plain RL |
|---|---|---|---|---|---|
| least (0.25) | 92 | 1.238 | 1.404 | 1.345 | 1.371 |
| middle (0.37) | 91 | 1.227 | 1.293 | 1.325 | 1.343 |
| most (0.59) | 91 | 0.974 | 1.111 | 1.115 | 1.273 |

Fitted lapses (all three splits): MCPL developmental 0.3 vs instance-based 0.1; IGT developmental 0.0 vs instance-based 0.3.

| # | Severity | Finding | Consequence |
|---|---|---|---|
| X1 | — (confirmed) | The MCPL predictability effect is **not** a pure regression-to-the-floor artefact: the gap correlates +0.66 with an *independent* predictability proxy (population-static model's top-1), +0.77 with the reference's own, +0.49 with the developmental model's own top-1. | The structural reading survives in direction. |
| X2 | medium | **The lapse floor explains part but not most of it.** At λ = 0.3 the learner cannot score below 0.31 nats on a perfectly predictable participant (reference: 0.093), but its observed top-tercile NLL is 1.469 — the floor accounts for ≤ 0.2 of the 0.49 gap. | Gap 4's "fault is structural, not a tuning matter" (`:57-58`) should read "mostly structural; ≈ 0.2 nats is the lapse mixture". Cheap check: refit λ per tercile on tune. |
| X3 | high | **The ordinal-head claim is untested, and the evidence points elsewhere.** `memory_only` — the project's own exemplar baseline, no learned kernel, no ordinal structure — is the *best* model on the most predictable participants (0.847 < 0.981). Plain exemplar retrieval suffices; nothing in X-002 shows that "the categorical head cannot express the ordinal, graded structure" (`:55-57`). | Candidate A must separate the two mechanisms: parametric+ordinal-head-only, exemplar-only (= `memory_only`), both; and `memory_only` must be in Candidate A's "must beat" list (`CANDIDATE_ARCHITECTURES.md:25` omits it). |
| X4 | high | **The IGT advantage is confounded with reference miscalibration.** The IGT references carry λ = 0.3 (grid edge) and β = 6.0 (grid edge); their NLL floor for a perfectly predictable participant is 0.255, larger than the observed top-tercile gap (0.137). The gap correlates −0.78 with the developmental model's own top-1 and +0.38 with the bias model's: it is largest where the developmental model happens to predict well, not where exploration matters. | Gap 5's exploration hypothesis (`:69-70`) has a cheaper rival: under-tuned references. Required before any IGT claim: extend the reference grids (β up to ≥ 20, λ finer) on tune; report λ = 0 NLL; add PVL-Delta. |
| X5 | low | The "least predictable" MCPL tercile is at chance for every model (2.02–2.21): those participants carry no discriminating information and should be reported but down-weighted. | Report per-tercile results in every human table. |

### 5. Candidate controls (`docs/phase2/CANDIDATE_ARCHITECTURES.md`)

| Candidate | Missing or weak control | Required addition |
|---|---|---|
| A (`:25-28`) | Ordinal head and exemplar head are confounded; learned arbitration vs *heuristic* arbitration untested; `memory_only` absent from "must beat" | parametric+ordinal only; exemplar only; fixed density-based arbitration rule; `memory_only` and the instance-based reference with the *same* similarity kernel (parameter-matched) |
| B (`:37-41`) | "transition-predictor-only" shows prediction ≠ causation but no control shows the *content* of the model matters | shuffled-dynamics model (trained on permuted transitions) with equal capacity; planning horizon 0 |
| C (`:48-51`) | no upper bound | train-from-scratch-per-family (no transfer, no forgetting) bounds both effects |
| D (`:61-66`) | random modulators control is good; missing the "feature not modulator" control | the same RPE running average supplied as a policy *input* without modulating learning rate/temperature; a capacity-matched GRU that can learn running averages itself |
| E (`:75-80`) | "social reward only" tests information vs reward, not *relationships* | "information without structure": demonstrations drawn from a global pool at the same rate; misinformation with a known-truth oracle so damage is measured, not inferred |
| all | the age-script-only H2 null (`:16`) is listed but no candidate names it as a control | every synthetic claim reports the age-script-only arm |

### 6. Trivial ways to pass a gate, and the guard for each

| Trivial pass | Guard |
|---|---|
| Score test participants in memory, report an aggregate | harness lock (P1); mandatory `participants` list; `phase2_guard.py check-test-access`; Critic re-run |
| Pick the 3 families that worked, after dev | families named in the preregistration before dev results are viewed (G3); `check-freeze` verifies the preregistration file hash at the frozen commit |
| Select lapse/hyper-parameters on test | selection restricted to tune in code (T1) |
| Re-preregister a tweaked candidate until it passes | one confirmatory shot per candidate class per partition; the ledger counts attempts; a second attempt needs a new dataset |
| Coarse reference grids | grids authored by the human-validation lane and hashed before dev (T2); edge fits trigger a logged extension |
| Report only lapse-mixed NLL | λ = 0 NLL and ECE mandatory (T4) |
| Run seeds 500–509 during development | `check-test-access` scans manifests and YAML for confirmatory seeds outside `reports/phase2/confirmatory/` |
| Build the family the candidate is good at | critic-authored family + specialist-should-win control (G3); "not worse by > 0.02 in any family" |
| Dirty-tree confirmatory run (Phase I M-3) | `check-freeze --require-clean`; manifests without `+dirty` |
| Hide failed runs to stay in budget | `phase2_guard.py budget` sums every manifest; failed runs must write manifests |
| Declare Gate D on one dataset | status page updates only from a confirmatory row naming ≥ 2 datasets (G5) |
| Omit the Phase I winner from the comparator set | comparator set named per family in the preregistration, including `memory_only` and the instance-based model (G4, X3) |

### 7. Checklist for `STATE_OF_THE_ART_BENCHMARK.md` ("directly reproduced vs published figure")

A comparison may be called *direct* only if every item below is present; otherwise it is labelled
*not comparable* and carries no evidential weight.

1. Exact source: DOI, table/figure number, page; the quoted sentence or cell.
2. The published quantity's definition: metric, units, data subset, split, exclusions, lapse handling.
3. Our quantity computed by the identical definition, or a numbered list of every deviation.
4. Participant and trial counts on both sides; differences explained (e.g. HeartSteps 37 vs 44).
5. The data version: file SHA-256, licence, retrieval date (`docs/DATA_SOURCES.md`).
6. Numbers produced by a committed script with a committed log; code commit; clean tree.
7. Uncertainty for ours (participant-clustered CI) and for the published value if reported.
8. The consistency rule stated *before* computing (a CI that also contains the null does not count as
   "consistent" without a precision criterion — Phase I M-5).
9. Whether the published model saw the data we hold out (Centaur was trained on Psych-101: any Centaur
   figure on these participants is not a held-out comparison).
10. Whether the published model is population-informed or individual-only (T5).
11. A verdict from {consistent, inconsistent, not comparable} with the rule that produced it.
12. No published number transcribed from memory; each is quoted from the fetched source text.

### 8. Conditions before any `hybrid_exemplar` number may be cited

1. `phase2_guard.py check-test-access` clean over the lane's reports and code; every result file lists
   its `participants`; synthetic runs carry manifests with seeds outside 500–509.
2. `run_policy_conformance(check_learning=True)` passes for every flag combination reported.
3. Ablations reported side by side: parametric+ordinal only, exemplar only, fixed 50/50, learned
   arbitration, no ordinal head; plus `memory_only`, the instance-based reference with the same
   similarity kernel, plain RL, and the GRU baseline, all tuned on tune with the committed ≤ 24-point
   grids (flags counted, T3).
4. Leakage checks: exemplar store reset per participant and containing only *past* trials of that
   participant under teacher forcing (unless declared population-informed); synthetic zero-shot on the
   transfer family = chance; no holdout symbols in practice streams; no test participants.
5. NLL with fitted λ and with λ = 0; ECE; per-predictability-tercile tables (§4 format), reproduced by
   the Critic from the lane's outcome files.
6. Every number labelled exploratory (dev) — no gate language, no "passed", no status-page change.
7. Compute cost (CPU seconds, parameter count, exemplar-store size) per model.

### 9. Required changes before the first confirmatory run (summary)

Blocking: G1, G3, P1, T1. High: G2, G4, G5, P2, P3, T2, T3, X3, X4. The rest are medium/low and
should be fixed in the same pass. I will re-review when the human-validation harness and the first
candidate are merged, and I will re-run every confirmatory evaluation myself.
