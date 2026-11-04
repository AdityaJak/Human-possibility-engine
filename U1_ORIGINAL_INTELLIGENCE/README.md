# U1_ORIGINAL_INTELLIGENCE — research track

A separate track from the Phase I–III code in `packages/hpe` (left unchanged). Read in this order:

1. `ORIGINALITY_THESIS.md` — questions, six candidate principles, novelty review, selection.
2. `NEW_INTELLIGENCE_PRINCIPLE.md` — the selected hypothesis in the eight required parts.
3. `MATHEMATICAL_FORMULATION.md` — environment and learner equations.
4. `PRIOR_ART_AND_NOVELTY.md` — the novelty ledger.
5. `EXPERIMENTS_AND_BASELINES.md` — preregistration (frozen by hash before the confirmatory run).
6. `TRANSFER_RESULTS.md` — results.
7. `INDEPENDENT_CRITIQUE.md` — critique.
8. `SCIENTIFIC_VERDICT.md` — verdict: **REJECT**. No new principle, algorithm or empirical finding. The hypothesis
   failed both preregistered confirmatory attempts (`results/` = attempt 1, `results_r1/` = attempt 2).

Code: `U1_IMPLEMENTATION/u1/` (numpy + scipy; torch only for the meta-RL comparator). No language model and no API
calls anywhere in the track.

```bash
cd U1_ORIGINAL_INTELLIGENCE/U1_IMPLEMENTATION
../../.venv/bin/python -m pytest -q tests          # 25 fast tests
../../.venv/bin/python -m u1.experiment --help     # train / eval / accumulate / freeze / summarize
bash run_confirm_r1.sh                             # attempt-2 confirmatory run (needs results_r1/FREEZE.json)
```

Results:
* `results/` — attempt 1, environment v2, frozen agent. `u0` is the development universe, `u1`–`u3` the
  confirmatory universes, `dev/` the development logs and rejected designs.
* `results_r1/` — attempt 2 (revision R1), environment v3. `u0` is the development universe, `u4`–`u8` the
  confirmatory universes.

Attempt 1's code is at commit 6776cfc; the working tree holds R1.
