# U1_ORIGINAL_INTELLIGENCE — research track

A separate track from the Phase I–III code in `packages/hpe` (left unchanged). Read in this order:

1. `ORIGINALITY_THESIS.md` — questions, six candidate principles, novelty review, selection.
2. `NEW_INTELLIGENCE_PRINCIPLE.md` — the selected hypothesis in the eight required parts.
3. `MATHEMATICAL_FORMULATION.md` — environment and learner equations.
4. `PRIOR_ART_AND_NOVELTY.md` — the novelty ledger.
5. `EXPERIMENTS_AND_BASELINES.md` — preregistration (frozen by hash before the confirmatory run).
6. `TRANSFER_RESULTS.md` — results.
7. `INDEPENDENT_CRITIQUE.md` — critique.
8. `SCIENTIFIC_VERDICT.md` — verdict.

Code: `U1_IMPLEMENTATION/u1/` (numpy + scipy; torch only for the meta-RL comparator). No language model and no API
calls anywhere in the track.

```bash
cd U1_ORIGINAL_INTELLIGENCE/U1_IMPLEMENTATION
../../.venv/bin/python -m pytest -q tests          # 15 fast tests
../../.venv/bin/python -m u1.experiment --help     # train / eval / accumulate / freeze / summarize
bash run_confirm.sh 128 0.001 20000                # confirmatory run (needs results/FREEZE.json)
```

Results: `results/` (`u0` = development universe; `u1`–`u3` = confirmatory universes; `dev/` = development logs and
rejected designs).
