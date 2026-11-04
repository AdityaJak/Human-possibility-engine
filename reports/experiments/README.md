# reports/experiments

Every file here is produced by code in this repository (`python -m hpe.evaluation ...`). Nothing is hand-written.

## Contents

* `pipeline_validation/<experiment>__pipeline_validation/` — **PIPELINE VALIDATION RUNS ONLY.** Scaled-down
  (3 seeds, 25% budget, first sensitivity variant) to prove the preregistration → raw logs → analysis → report path
  works. They are **not** research results and must not be cited as such. At the time they were produced the
  `developmental` policy class and `hpe.world.World` were not yet integrated into this branch, so every primary
  contrast is `inconclusive` ("no data for arm(s) ...") and all society arms were skipped. Baseline-arm numbers
  in them are real outputs of the baseline code at reduced budget.
* `contamination_audit.json` — leakage / measurement-integrity audit (`python -m hpe.evaluation contamination`)
  for every policy class registered when it ran.
* `<experiment_id>/` (added when preregistered experiments are run) — the real results: `definition.json`
  (with preregistration hash), `protocol.json`, `runs/<run_id>/{outcomes.jsonl.gz, manifest.json}`, `result.json`,
  `contrasts.json`, `report.md`, `figures/`.

## Regenerating

```bash
python -m hpe.evaluation analyze reports/experiments/pipeline_validation/*   # rebuild results from raw logs
```
