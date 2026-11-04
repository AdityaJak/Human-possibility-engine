# Phase III — Human data validation (evaluation team, lane `science/validation-v3`)

_Written 2026-11-04. Every number below is copied from a committed file named next to it. Exploratory: dev and
tune partitions only; the test partition was never requested by this lane (the validation CLI refuses it)._

## 1. Families, sources and parsers

All human data are Psych-101 transcripts (Binz et al., 2024; Apache-2.0), file `prompts_training.jsonl`
(sha256 `7a370a94…`), local Phase III subset sha256 `76f335e0…` (never committed). Data are read **only**
through `hpe.population.core.data.load_participants` / `load_family`, which applies the partition lock to raw
records before any transcript is parsed.

| Family | Role (thesis §3) | Studies | Parser | Partition file |
|---|---|---|---|---|
| `restless4` | counted, primary | bahrami2020four | `families/restless4.py` (director) | Phase III |
| `garcia`, `described` | counted | garcia2023experiential, plonsky2018when | `psych101_v2.parse_garcia/parse_described` (Phase II) | Phase II |
| `two_step` | counted **if validated** | kool2016when exp1–2, kool2017cost exp1–2, zorowitz2023data exp1 | **`families/two_step.py` (new)** | Phase III |
| `horizon_external` | external | sadeghiyeh2020temporal, somerville2017charting, feng2021dynamics | **`families/horizon_external.py` (new; reuses `psych101_v2.parse_horizon`)** | Phase III |
| `bandit2`, `srl3`, `horizon`, `casino`, `wpt` | secondary | Phase II | Phase II parsers | Phase II |
| `spatial_bandit` | secondary (optional) | schulz2020finding exp1–5 | **`families/spatial_bandit.py` (new; K = 8, needs `k_max: 8`)** | Phase III |

Validation command (dev/tune only; writes `reports/phase3/baselines/validation/<family>_<partition>.json`):

```bash
HPE_PSYCH101_P3=… PYTHONPATH=packages python -m hpe.population.families.validate two_step --partition dev
```

The format of each new family was scoped with a template probe that ran through `load_participants` on the
**dev** partition only (letters and numbers masked to `X`/`N`, counts of line templates); no tune or test
transcript was displayed.

## 2. `two_step` — parser specification

Encoding (details in the module docstring of `packages/hpe/population/families/two_step.py`):

* **Designs.** *One-stage* (kool2016when/exp1, kool2017cost/exp1): each day one of two spaceship pairs is shown;
  spaceships fly deterministically to one of two planets whose single alien yields the outcome → one step per
  day, `mode="bandit"`. *Two-stage* (kool2016when/exp2, kool2017cost/exp2, zorowitz2023data/exp1): one spaceship
  pair, common/rare transitions to a planet with two aliens → a spaceship step (`mode="none"`; the planet reached
  appears in the next step's features) and an alien step (`mode="bandit"`).
* **Action indices** are canonical per participant and fixed by information available before the first choice:
  the instruction order of spaceships/planets/aliens (Kool studies), the first presentation order (Zorowitz,
  whose instructions name no labels; red planet = −1, blue = +1). The transcript lists the options in varying
  order on some files (screen side); that order is a feature (`swapped`), never the index.
* **Outcomes → [0, 1]** by fixed affine maps: kool2016when/exp1 `(r + 4) / 9` (treasure +N, antimatter −N,
  nothing 0; observed range on dev −4…+5); kool2017cost/exp1 `found / 9` (found 0…9; the ×5 multiplier is a
  context feature and `received = found × 5` is checked on every line); binary outcomes unchanged
  (kool2016when/exp2, kool2017cost/exp2 0/1 pieces; Zorowitz treasure = 1, junk = 0). A value outside the map's
  range is a parse error, never clipped. The −4…+5 and 0…9 ranges were taken from the dev data; we did not verify
  them against the original papers' method sections.
* **12 context features in [−1, 1]**: stage, design, start pair, current planet, planet reached on the previous
  day, multiplier flag, presentation swapped, day progress (stated number of days; Zorowitz 201 = the length of
  every dev transcript), and 2-dimensional `psych101_v2._code` codes of the two canonical options **salted per
  participant** (labels carry no cross-participant identity).
* **Missed responses** produce no step; a day with a spaceship choice but a missed alien choice keeps the
  spaceship step. Counted in `meta` (below).
* **Checks that raise `ParseError`** (participant excluded, reason recorded): unknown line template, chosen label
  not presented, planet not in the instructed set, aliens not those instructed for the planet (or alien colour ≠
  planet colour), multiplier statement missing, `received ≠ found × multiplier`, outcome outside the fixed range,
  more days than stated.

Unit tests on synthetic transcripts: `tests/population/families/test_families.py` (one-stage, two-stage with
multiplier and missed alien choice, Zorowitz ordering, rejection of out-of-range/wrong-alien/unpresented
choices, per-participant salting).

## 3. `two_step` — validation on dev (and parse rate on tune)

Source: `reports/phase3/baselines/validation/two_step_dev.json`, `two_step_tune.json`.

| Partition | Transcripts | Parsed | Excluded | Exclusion reason |
|---|---|---|---|---|
| dev | 275 | 274 | 1 | `kool2017cost/exp2.csv#43`: outcomes such as "You find −165 pieces of space treasure" (negative counts for a 0/1 outcome — a defect of that transcript); outside the fixed range |
| tune | 139 | 139 | 0 | — |

Dev descriptives per file (steps = choices encoded; two-stage files have two steps per day):

| File | n | design | steps min / median / max | missed spaceship choices per participant (mean) | missed alien choices (mean) | P(action 0) | mean normalised outcome | common-transition fraction (mean over spaceships) |
|---|---|---|---|---|---|---|---|---|
| kool2016when/exp1 | 72 | one-stage | 14 / 122 / 125 | 5.97 | 0 | 0.483 | 0.529 | 1.000 (deterministic) |
| kool2016when/exp2 | 75 | two-stage | 58 / 247 / 250 | 3.71 | 1.39 | 0.527 | 0.517 | 0.704 |
| kool2017cost/exp1 | 37 | one-stage | 176 / 196 / 200 | 4.30 | 0 | 0.543 | 0.524 | 1.000 (deterministic) |
| kool2017cost/exp2 | 34 | two-stage | 382 / 398 / 400 | 2.24 | 0 | 0.511 | 0.510 | 0.699 |
| zorowitz2023data/exp1 | 56 | two-stage | 402 / 402 / 402 | 0 | 0 | 0.503 | 0.456 | 0.697 |

Behavioural signature (participant-level mean probability of repeating the previous day's spaceship; dev):

| File | after reward, common | after reward, rare | after no reward, common | after no reward, rare |
|---|---|---|---|---|
| kool2016when/exp2 | 0.708 | 0.664 | 0.611 | 0.672 |
| kool2017cost/exp2 | 0.728 | 0.675 | 0.632 | 0.678 |
| zorowitz2023data/exp1 | 0.829 | 0.754 | 0.717 | 0.748 |

| One-stage file (same start pair as the previous day) | after a positive outcome | after 0 | after a negative outcome |
|---|---|---|---|
| kool2016when/exp1 | 0.792 | 0.656 | 0.406 |
| kool2017cost/exp1 (outcomes ≥ 0) | 0.678 | 0.392 | — |

Reading: the parsed transitions reproduce the designs (deterministic one-stage; ≈ 0.70 common transitions in every
two-stage file), outcome statistics sit near 0.5 as expected for drifting payoffs, every multiplier line satisfies
`received = found × 5`, and the classic reward × transition interaction of two-step stay probabilities (stay after
rewarded-common > rewarded-rare, after unrewarded-rare > unrewarded-common) appears in all three two-stage files.
A mis-assigned action index or planet would destroy these patterns. The tune partition parses without exclusion
and shows the same patterns (`two_step_tune.json`).

**Amendment A3-1 §11 item 8 — the three conditions for counting `two_step`:**

| Condition | Status | Source |
|---|---|---|
| parser reaches ≥ 95 % parse rate on dev | **met: 274 / 275 = 99.6 %** | `two_step_dev.json` (`parse_rate` 0.99636) |
| encoding-leak check (`core.conformance.check_encoding_leak`, every input slot, threshold 0.98) is clean on dev | **met: no slot flagged** (also clean for restless4, garcia, described, horizon_external, spatial_bandit on dev) | `two_step_dev.json` (`encoding_leak_check: []`); the runner repeats the check per family in `run_info.json` |
| the critic signs off | **open** — not this lane's to give | — |

**Verdict of this lane: the `two_step` parser passes validation on dev** (one transcript excluded for corrupt
outcome values). Whether `two_step` becomes a counted family is the director's decision, to be recorded before any
freeze (thesis §3). Caveats for that decision: (i) the family mixes a one-stage and a two-stage design (one vs two
choices per day); (ii) the 5 two_step participants listed in `exposed_participants.json` are excluded from test
scoring by the core loader; (iii) one dev participant of kool2016when/exp1 has only 14 encoded choices (111 missed
days) — kept, because no behaviour-based exclusion rule was preregistered.

## 4. `horizon_external` — validation on dev

The three studies use the wilson2014 horizon-task vocabulary; the Phase II parser parses every dev and tune
transcript (`reports/phase3/baselines/validation/horizon_external_{dev,tune}.json`):

| File (dev) | n | steps per participant min / median / max | free choices (median) | games | P(action 0) | mean normalised reward |
|---|---|---|---|---|---|---|
| feng2021dynamics/exp1 | 10 | 2400 / 2400 / 2400 | 1120 | 320 | 0.512 | 0.527 |
| sadeghiyeh2020temporal/exp1 | 31 | 390 / 665 / 1170 | 302 | 154 | 0.485 | 0.510 |
| somerville2017charting/exp1 | 31 | 1200 / 1200 / 1200 | 560 | 160 | 0.518 | 0.525 |

Parsed 72/72 (dev) and 37/37 (tune); instructed trials are kept as inputs (`forced=True`) and not scored. Per the
thesis, models for the external-replication analysis are fitted on `horizon` (wilson2014) only; this lane did not
fit anything on `horizon_external`.

## 4b. `spatial_bandit` — validation on dev

Schulz et al. (2020), 8 options, 30 rounds × 10 trials, rewards in the stated range 0–50 → `/50`; action index =
position in the instruction's option list (digits 1–8, identical order in every dev transcript); block = round.
Parsed 214/214 (dev) and 106/106 (tune), 300 choices each
(`reports/phase3/baselines/validation/spatial_bandit_{dev,tune}.json`). Mean normalised reward on dev 0.686–0.749
per file; P(option 1) is 0.28–0.33 in exp1–4 and 0.07 in exp5 (a different reward structure, not a parse
difference: all five files list the options in the same order). No model was fitted on this family (time).

`restless4` (director's parser) was re-validated on dev for completeness: 348/348 parsed, 14–150 choices
(median 148), `reports/phase3/baselines/validation/restless4_dev.json`.

## 5. Independence and contamination

* **Independence.** Counted families come from different publications with disjoint participants
  (bahrami2020four; garcia2023experiential; plonsky2018when; and, if admitted, kool2016when + kool2017cost +
  zorowitz2023data). They all reach us through one compilation (Psych-101): one transcript format, one
  preprocessing pipeline, natural-language rendering by the same authors. Independence is therefore at the level
  of participants and original studies, **not** of data processing. `two_step` itself pools three publications
  (two from the same group, Kool/Cushman/Gershman).
* **Contamination of external models.** Every participant in `prompts_training.jsonl` — including every dev, tune
  and test participant of every Phase III family — was in the training data of Centaur (Binz et al.). Centaur or
  Minitaur scores on these participants would be biased in their favour and are not valid comparators here.
  The comparator pool contains no language model (no LM calls in any model).
* **Internal contamination.** Deviation D3-1 (director): opening lines of 5 test transcripts were displayed before
  the partition freeze; those participants are excluded from confirmatory scoring by the core loader. This lane
  displayed dev transcripts only (template counts and the outcome lines of one excluded dev participant while
  diagnosing its parse failure). `mcpl` and `igt` remain contaminated (Phase I test scoring) and were not used.

## 6. External datasets (search 2026-11-04)

Goal (thesis §3): a dataset outside Psych-101 with sequential choices, ideally with affect or social variables,
under a licence that permits research reuse. Licences were read from the pages themselves on 2026-11-04.

| Candidate | What it contains | Licence as stated on the page | Usable? |
|---|---|---|---|
| **Psych-201**, Hugging Face `marcelbinz/Psych-201` (https://huggingface.co/datasets/marcelbinz/Psych-201) | 208,021 rows (one per participant transcript), parquet, 541 MB; columns include `text`, `study`, `experiment`, `participant`, **`is_psych101`**, `is_psych101_test`, RTs, demographics and **questionnaire scores** (STAI-S/T, PHQ-9/8, GAD-7, BDI-II, OCI, PSWQ, IUS, BIS/BAS, AUDIT, DAST, …), `in_eval` (columns from the datasets-server info endpoint) | dataset card tag: `apache-2.0`; the datasets-server `info` endpoint reports an empty licence string; README empty | **Most promising, not yet used.** Participants with `is_psych101 = false` were not in Psych-101 and hence not in Centaur's Psych-101 training data; questionnaire scores are *trait-level* person covariates (not trial-level affect). Same transcript format as Psych-101 (not independent in processing). Needs a director decision, a new locked partition file, and licence confirmation (card tag vs empty info field) before any use. Not downloaded by this lane. |
| Psych-201 collection repo (https://github.com/linushof/Psych-201, fork of marcelbinz/Psych-201) | per-experiment folders with zipped `prompts.jsonl` and generation scripts; "The deadline for contributions has now passed" | no licence file or statement shown | covered by the HF dataset above |
| Psych-201 paper: Binz et al., "Post-training makes large language models less human-like", arXiv 2605.07632 (v1 8 May 2026, v2 25 May 2026) | introduces Psych-201 (208,021 participants, 25.9 M responses per the search summary) | article CC BY 4.0; no dataset statement on the abstract page | reference only |
| Blain & Rutledge (2020) eLife 57977 — two-armed "car race" bandit (stable/volatile) with momentary happiness ratings every 3–4 trials, 75 participants | data/code at https://github.com/BastienBlain/MSWB_LearningNotReward | no LICENSE file shown on the repository page | **not usable** without a licence (all rights reserved by default) |
| Gillan et al. (2016) two-step + psychiatric questionnaires, OSF `usdgt` | public OSF node | `node_license: null` (OSF API) | not usable without a licence |
| McEllin et al. (2023) partner choice in economic games (social), OSF `yebvt` | public OSF node | `node_license: null` | not usable without a licence |
| Guitart-Masip et al. (2023) reward learning and stress (COVID-19), OSF `ps38n` | public OSF node | `node_license: null` | not usable without a licence |
| OpenCogData index (https://nimh-dsst.github.io/OpenCogData/) | catalogue of public cognitive-task datasets | site: "MIT License DSST/NIMH"; no per-dataset licences | a lead list only |

Conclusion: **no licensed non-Psych-101 dataset with trial-level affect or social variables was verified in this
session.** Gate F remains untestable on licensed data as far as this lane established. The concrete next step is
Psych-201's non-Psych-101 participants (sequential-choice studies with trait questionnaires), subject to the
director's decision and licence confirmation.

## 7. Known gaps

* `spatial_bandit` parsed and validated but no model fitted on it.
* The −4…+5 / 0…9 outcome ranges of the Kool one-stage tasks were not checked against the original papers.
* Validation shows the parser reproduces known design features and behavioural signatures; it cannot prove
  that every Psych-101 rendering matches the original raw data (we have no access to the raw files).
