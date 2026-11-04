# Experiment Registry v3 (Phase III)

Rules: `PHASE3_RESEARCH_THESIS.md` §5–§7. Exploratory rows (`X3-`) use dev/tune or development seeds and
never pass a gate. Confirmatory rows (`C3-`) exist only after a freeze record and run once. Every
attempt is logged, including abandoned ones and hyper-parameter searches (their logs live in the
artifact directory named in the row).

## 1. Partitions

| Source | Seed | Test digest | Families |
|---|---|---|---|
| `reports/phase2/partitions/psych101.json` | 20261105 | `7dc2e779…` | garcia, described, bandit2, srl3, horizon, casino, wpt (test never accessed); mcpl, igt (contaminated) |
| `reports/phase3/partitions/psych101_p3.json` | 20261106 | `eb14a4d9…` | restless4, two_step, spatial_bandit, horizon_external (frozen 2026-11-04 before any fitting, commit `278f2817`) |

## 2. Amendments

| Date | Change | Reason | Seen confirmatory data? |
|---|---|---|---|
| 2026-11-04 | **Deviation D3-1 (disclosed).** Before the Phase III partitions were generated, the director printed the opening ≤ 900 characters (≤ 4 choice lines) of the first transcript of 16 study files to scope parsers. After partitioning, 5 of these participants fell into test partitions (kool2016when/exp2#0, kool2017cost/exp1#0, sadeghiyeh2020temporal/exp1#0, schulz2020finding/exp4#0, zorowitz2023data/exp1#0). They are **excluded from every Phase III confirmatory scoring**, decided now, before any model was fitted (`reports/phase3/partitions/exposed_participants.json`). | format inspection preceded the partition freeze | no outcome statistics, no model scores; ≤ 20 choices of one participant per study were displayed |
| 2026-11-04 | **Deviation D3-2 (disclosed).** While measuring sequence lengths for budget planning, the director parsed every transcript of the seven Phase II generic families **including the Phase II test partition** through the unlocked record iterator, printing only per-family counts (participants; min/median/max trials). No choices, outcomes or scores were computed or displayed. Remedy: all Phase III data access goes through `hpe.population.core.data`, which applies the partition lock before parsing; the guard was extended to Phase III. | tooling misuse (bypassed the lock) | no: trial-count metadata only (e.g. garcia 164–208 trials, described up to 750, horizon 610–2,400 including instructed trials) |

## 3. Exploratory ledger

| ID | Date | Lane | Question | Partition | Result | Artifacts |
|---|---|---|---|---|---|---|

## 4. Confirmatory ledger

| ID | Date | Gate | Preregistration | Freeze record | Result | Verdict | Artifacts |
|---|---|---|---|---|---|---|---|

## 5. Abandoned attempts

| Date | Lane | What | Why abandoned | Artifacts |
|---|---|---|---|---|
