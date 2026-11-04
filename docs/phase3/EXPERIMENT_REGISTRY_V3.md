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
| 2026-11-04 | **Amendment A3-1**: joint freeze per candidate; B1 vs a separately trained twin; harness-owned adaptation modes; B2 adapter pool specified; gate rules in code; max(bootstrap, t) p-values; comparator adequacy and config-counting rules; explicit counted families; strict isolation and causality checks; hardened test lock (thesis §11) | independent critic's design review (`INDEPENDENT_SCIENTIFIC_AUDIT.md`) | no |
| 2026-11-04 | **Corrections to D3-1/D3-2 wording** (critic D-1): D3-1 concerned 18 study files (not 16), and each previewed participant is the first record of its file in source order (ids listed in `exposed_participants.json`, computed from the file); "test never accessed" for garcia/described means test outcomes were never computed or viewed — their transcripts were parsed once for trial counts (D3-2); `two_step` has 273 test participants in the partition and 270 scorable after the D3-1 exclusions | critic D-1 | no |
| 2026-11-04 | **Deviation D3-3 (disclosed).** A Phase II-era scratch file `experiment_heads.json` holds the opening of the first transcript of each of 76 experiment files (used in Phase II for dataset selection). The critic counted choice lines without reading text: only the restless4 opening contains choices (3), and that participant (bahrami2020four/exp.csv#0) is in dev. No Phase III test participant's choices are in it. | Phase II tooling | no |
| 2026-11-04 | **Amendment A3-2**: Gate D (leave-one-family-out, history-only inputs, k ∈ {0,16,64}, specialist / pretrained pop_gru / hb_cog comparators) and Gate E (synthetic switching suite with oracle and Bayes-filter references; development seeds 0–199; confirmatory seeds 3000–3049 reserved untouched) fixed (thesis §12) | critic G-4, G-5 | no |

## 3. Exploratory ledger

| ID | Date | Lane | Question | Partition | Result | Artifacts |
|---|---|---|---|---|---|---|

## 4. Confirmatory ledger

| ID | Date | Gate | Preregistration | Freeze record | Result | Verdict | Artifacts |
|---|---|---|---|---|---|---|---|

## 5. Abandoned attempts

| Date | Lane | What | Why abandoned | Artifacts |
|---|---|---|---|---|
| 2026-11-04 | director | X3-001a: torch `pop_gru` (v3 encoding) sanity run, first trainer defaults (≤ 40 epochs, batch 16, no learning-rate decay) | **Under-converged comparator**: 3 of 4 restless4 fits stopped at the epoch cap while validation NLL was still falling; garcia tune 0.460 vs Phase II numpy GRU 0.447. Stopped after restless4 and garcia; trainer defaults changed (≤ 150 epochs, batch 8, plateau halving, patience 10, optional input dropout) before any lane ran an experiment | `reports/phase3/director/x3_001_core_sanity/` (restless4 tune 0.734, garcia 0.460) |
