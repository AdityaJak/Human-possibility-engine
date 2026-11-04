# Dataset adapters

Adapters download external human data on demand into the gitignored `datasets/external/`
directory and verify SHA-256 hashes recorded on 2026-11-04. **No raw external data is committed.**
Licence review: `docs/DATA_SOURCES.md`. Parsing/validation code lives in `hpe.evaluation.human`.

| Script | Dataset | Licence | Output | Env override |
|---|---|---|---|---|
| `psych101.py` | Psych-101 (Binz et al. 2024) → learning-experiment subset (MCPL, IGT, bandits) | Apache-2.0 (dataset card) | `datasets/external/psych101_learning_subset.jsonl` | `HPE_PSYCH101_SUBSET`, `HPE_PSYCH101_FULL` |
| `heartsteps.py` | HeartSteps V1 `suggestions.csv` (Klasnja et al. 2019) | CC-BY-4.0 (`LICENSE` in repo) | `datasets/external/heartsteps_v1_suggestions.csv` | `HPE_HEARTSTEPS_SUGGESTIONS` |

```bash
python datasets/adapters/psych101.py      # ~859 MB download, then filter + verify
python datasets/adapters/heartsteps.py    # ~4.4 MB
python -m hpe.evaluation.human run experiments/exp07_human_heldout.yaml
python -m hpe.evaluation.human run experiments/exp07b_causal_grounding.yaml
```

Hashes: Psych-101 full `7a370a94…75ccef01` (859,179,475 bytes); learning subset `9b085cfb…f2330297`
(1,816 transcripts); HeartSteps suggestions `60ee7896…ce1745c5ddb` (4,371,854 bytes). If an upstream file
changes, verification fails loudly rather than silently analysing different data.
