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

## 3. Exploratory ledger

| ID | Date | Lane | Question | Partition | Result | Artifacts |
|---|---|---|---|---|---|---|

## 4. Confirmatory ledger

| ID | Date | Gate | Preregistration | Freeze record | Result | Verdict | Artifacts |
|---|---|---|---|---|---|---|---|

## 5. Abandoned attempts

| Date | Lane | What | Why abandoned | Artifacts |
|---|---|---|---|---|
