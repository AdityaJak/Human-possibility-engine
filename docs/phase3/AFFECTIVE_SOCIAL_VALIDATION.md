# Affective and Social Validation (Gate F) — lane `research/affective-social`

_Written 2026-11-04. **Status: EXPLORATORY.** Every number below comes from dev (fitting) and tune
(selection and report) partitions. No test participant was encoded, scored or looked at. Nothing here
passes a gate. No statement in this file says or implies that any model has, simulates or experiences
emotions: "affect" means a person's **reported** happiness rating, used as an observed input._

## 1. Summary

* **Gate F is testable**, for affect, on a licensed dataset this lane found and adapted: the Great Brain
  Experiment (GBE) risky-decision-and-happiness data (Rutledge; Dryad doi:10.5061/dryad.prr4xgxkk, CC0-1.0;
  47,067 participants, 2.7 M choices, 1.1 M momentary happiness ratings). A participant partition was
  committed before any fit (`reports/phase3/affect/partitions_gbe.json`, seed 20261108, dev/tune/test
  18,827 / 9,413 / 18,827).
* For social variables a small licensed dataset exists (Najar, Bonnet, Bahrami & Palminteri 2020, MIT
  licence; 68 participants with observed demonstrator choices). It is adapted and partitioned
  (`reports/phase3/affect/partitions_najar2020.json`, 27 / 14 / 27); with 27 test participants it is
  under-powered for anything but a large effect (§7).
* Exploratory results: see §6 (filled from `reports/phase3/affect/*/summary.json`).
* Psych-101 contains no study with trial-level affect ratings or social information (§2.3).

## 2. Datasets searched

All pages were opened on 2026-11-04. "Licence" quotes the repository's own metadata or licence file.
Rule applied (lane brief): use only data whose licence permits research use and redistribution of derived
statistics; a missing or unclear licence means **not used**, even when the accompanying article is open
access (an article licence does not cover a separately deposited dataset).

### 2.1 Affect (trial-level mood / happiness ratings with sequential choices)

| Dataset | URL opened | Licence as stated | Size | Trial-level variables | N | Usable? |
|---|---|---|---|---|---|---|
| Rutledge, GBE "Risky decision and happiness task" (data behind Rutledge et al. 2014 PNAS; 2016 Curr Biol; 2017 JAMA Psychiatry) | https://datadryad.org/dataset/doi:10.5061/dryad.prr4xgxkk ; mirror https://zenodo.org/records/4421847 | Dryad API `"license": "https://spdx.org/licenses/CC0-1.0.html"`; Zenodo `"license": {"id": "cc-zero"}` | zip 87.5 MB (`Rutledge_risk_and_happiness_task_2023.zip`; MD5 = Zenodo's, SHA-256 = Dryad's listed digest) | per trial: safe amount, gamble win/loss amounts, choice (gamble/safe), outcome, RT; happiness 0–100 before trial 1 and after every 2–3 trials (12 per 30-trial play) | 47,067 (91,058 plays) | **Yes** — used (§4). Dryad's own download endpoint answered `{"error":"Unauthorized, must have current bearer token"}`; the identical file came from the Zenodo mirror |
| Blain & Rutledge 2020 eLife "Momentary subjective well-being depends on learning and not reward" | https://github.com/BastienBlain/MSWB_LearningNotReward | none shown (no LICENSE file, no licence in sidebar) | `.mat` files | 2 bandit tasks × 80 trials; happiness every 3–4 trials | 75 | No — no licence |
| Preregistered replication of Rutledge et al. 2014 (Cognition & Emotion 2021) | https://osf.io/pcua3 (file `9g2zw`, "Data and Analyses.zip") | OSF node licence: none (`node_license: null`) | 236 MB | gamble/safe choices, happiness ratings | not checked further | No — no licence |
| Keren et al. 2021 eLife "The temporal representation of experience in subjective mood" | https://osf.io/vw7sz | OSF node licence: none; the *article* is CC0 per eLife/UCL Discovery | per-subject xlsx/csv (random, structured, adaptive gambling + mood) | gambling choices, mood ratings | online + lab samples | Not used — data-deposit licence unclear (article CC0 does not state the OSF files' terms) |
| Jangraw et al. 2023 Nat Hum Behav "A highly replicable decline in mood during rest and simple tasks" | https://osf.io/km69z (online cohorts); app cohorts = the GBE Dryad record above | OSF node licence: none | `PassageOfTimeDysphoria_Data.zip` 206 MB | mood ratings during rest / gambling blocks | online cohorts (≈ 1,900 adults) | No for the OSF part — no licence. App part: same GBE data (used) |
| Mason et al. 2024 "Rare and extreme outcomes in risky choice" (found via the `multiRL` package) | https://osf.io/hy3q4 | OSF node licence: none | 81 MB csv | risky choices; **no** affect or social variable | 143 | No — irrelevant and unlicensed |

### 2.2 Social (sequential choices with observed choices of others / advice)

| Dataset | URL opened | Licence as stated | Size | Trial-level variables | N | Usable? |
|---|---|---|---|---|---|---|
| Najar, Bonnet, Bahrami & Palminteri 2020 PLoS Biol (imitation in social RL) | https://github.com/hrl-team/mfree_imitation (`data/task1.csv`, `data/task2.csv`) | `LICENSE`: "MIT License / Copyright (c) 2019 hrl-team" (repository-wide) | 0.65 + 0.91 MB | own choice, outcome; demonstrator choice on observational trials (outcome never shown) | 24 + 44 = 68 (experiment 3, n = 302, is not in the repository) | **Yes, small** — used (§4.5). Caveats: MIT is a software licence applied to the whole repository; demonstrations were computer-generated (Bernoulli 0.8 / 0.2) while participants were told they observed another participant (paper, Methods) |
| Toyokawa, Whalen & Laland 2019 Nat Hum Behav (3-armed bandit, groups see choice frequencies) | https://github.com/WataruToyokawa/ToyokawaWhalenLaland2018 | none shown; README: "some are missing due to their size … WT will upload all the data to somewhere open" | csv | own choice, payoff, others' choice frequencies | 699 analysed | No — no licence, data incomplete |
| Toyokawa & Gaissmaier 2022 eLife (risky bandits with social frequency information) | https://github.com/WataruToyokawa/ToyokawaGaissmaier2021 | none shown (article CC-BY 4.0 "except where otherwise noted") | — | own choice, payoff, others' choice counts | 585 | No — no data licence |
| Deffner, Kleinow & McElreath 2020 R Soc Open Sci (4-armed bandit, groups of 4) | https://github.com/DominikDeffner/Dynamic-Social-Learning ; Zenodo https://zenodo.org/records/4034787 | GitHub: none shown; Zenodo metadata `"license": {"id": "other-open"}` (no licence text) | 4.2 MB | own choice, payoff, group members' previous choices | 200 + 37 individual controls | Not used — licence unclear. **Best social candidate if the authors add an explicit licence** |
| Zhang & Gläscher 2020 Sci Adv (social influence task) | https://github.com/lei-zhang/SIT | README/LICENSE: "CC BY-NC 4.0" | `data/behavioral/*.mat` 0.5–5 KB | **only aggregate figure data** (group means/SEMs; 185 per-person parameters); no trial-level choices | 185 | No — trial-level data not deposited |
| Heffner, Son & FeldmanHall 2021 Nat Hum Behav (emotion prediction errors, repeated ultimatum game; affect *and* social) | https://github.com/jpheffne/epe | none shown | csv | offers, accept/reject, expected and experienced emotion ratings | 1,016 | No — no licence |

### 2.3 Psych-101 (the Phase III source)

`prompts_training.jsonl` (sha256 `7a370a94…`; Hugging Face page licence "apache-2.0") holds 76 experiment
files (counted from the `experiment` field only). Instruction headers of the studies outside the Phase II/III
partitions were read from `experiment_heads.json`; no participant choices were printed. None has trial-level
affect ratings or observed choices of real others. Closest, all unsuitable: `wise2019acomputational` (shock-
probability ratings — expectations, not affect), `jansen2021dunningkruger` (self-assessed performance),
`hilbig2014generalized` (four fictitious "expert ratings" used as cues).

### 2.4 Searches that returned nothing usable

Dryad API (`/api/v2/search`, all CC0 by policy): "social learning bandit", "mood ratings reward learning",
"social information multi-armed bandit", "momentary mood gambling", "advice taking learning", "mood learning",
"trust game", "ultimatum game", "observational learning", "conformity" — only the GBE record was relevant.
Zenodo API: "happiness ratings gambling task" (returned the GBE mirror), "mood ratings reinforcement learning",
"momentary mood bandit", "social learning multi-armed bandit experiment", "social information bandit humans
data", "observational learning bandit humans", "conformity bandit experiment data", "advice taking learning
task data" — nothing else with sequential choices plus affect/social variables. Seen only as search snippets
and **not assessed**: OpenNeuro ds004315 (EEG, mood manipulation + probabilistic selection), hBayesDM's
`rdt_happiness` example data, Mesoudi et al. 2015 Proc B (Zenodo 5015579, CC0; arrowhead design task, not a
choice-sequence task).

## 3. What Gate F needs, and how the GBE data meet it

Gate F (thesis §6): on a licensed human dataset where affect or social variables are observed, the
mechanism beats a **matched-complexity control** (same added parameters fed permuted variables) by ≥ 0.02
nats/choice, CI excluding 0, on held-out participants.

* Observed affect: "How happy are you right now?" (0–100) before trial 1 and after every 2–3 trials. The
  rating stored on a trial was given **after** that trial's outcome, so it is an input from the **next**
  trial on; the pre-play rating is an input to trial 1. A trial's own rating never reaches its own row
  (unit test `test_current_rating_never_reaches_its_own_row`).
* Choice: gamble vs safe on described options (gain, mixed and loss trials). This is a decision task with
  outcome feedback, not a learning task: option values are shown, so sequential effects can only come from
  state (recent outcomes, mood, fatigue, person), which is what an affect mechanism would have to capture.
* Held-out participants: 18,827 test participants exist and are locked.

## 4. Methods

### 4.1 Data access and partitions

* Adapter `packages/hpe/population/affect/gbe.py`. The `.mat` (sha256 `b6535132…`) is read with
  `scipy.io.loadmat`; participants outside the requested partitions are discarded before any record is
  encoded; `test` requires a freeze record (`check_partition_access`, same rule as `core.data`).
* Partition: `make_partition` — seeded permutation (`hpe.core.rng.stream(20261108, "phase3_affect_partition",
  "gbe_risk_happiness")`) of the 47,067 participant ids, 40/20/40; written and committed (`c62830b`) before any
  model was fitted; test digest `f19e278e…` is verified on every load.
* Validation on dev+tune: every checked play (first ≤ 10 per participant, 28,240 participants) satisfies the README invariants (safe choice ⇒ outcome =
  safe amount; gamble ⇒ outcome ∈ {win, loss}; pre-play rating present; ratings 0–100); 0 exclusions.
  One correction before fitting: the first value-range check assumed |amount| ≤ 150 and rejected 14,214
  ratio-design plays (README: up to 55 × 4 = 220); the scale was set to 220 after counting option values by
  design version on dev only.
* Sequence = a participant's plays in order. The adapter default is ≤ 10 plays; run `gbe_explore_01` uses
  ≤ 4 plays (≤ 120 trials) for compute (DA-3).

### 4.2 Encoding

Encoding v3 with K = 2 (0 = safe, 1 = gamble), F = 12 features: safe, win, loss, gamble EV, EV − safe (all
/220), trial type (gain/mixed/loss), position in play, play index, design family, gamble range. Previous
outcome block: the safe amount is always observed (it is displayed), the gamble outcome only when chosen.
Block start at each play. Three affect columns are **appended after** the v3 layout and used only by models
that ask for them: latest rating before this choice (/100), "new rating since the last choice", trials since
that rating (/3).

### 4.3 Models (all registered with `owner="affect"`; all pass `check_causal`)

| Model | Role | Inputs | Parameters (hidden 16 / 32) |
|---|---|---|---|
| `affect_gru_base` | comparator | v3 only | see §6 |
| `affect_gru_obs` | **mechanism** (candidate) | v3 + the 3 affect columns | base + 3·3·h |
| `affect_gru_shuf` | **matched-complexity control** | v3 + the 3 columns with each rating replaced by one of the person's *earlier* ratings, drawn at random (first rating kept) | identical to `obs` |
| `affect_gru_aux` | latent-affect variant (candidate) | v3 only; trained with an auxiliary head predicting each observed rating from the hidden state (Rutledge-style "mood as a function of recent outcomes", learned) | base + h + 1 |
| `affect_gru_aux_shuf` | its control | as `aux`, targets permuted within participant | identical to `aux` |

Control design. The brief asked for "the same extra inputs permuted across trials within participant". A
full within-person permutation would feed a choice ratings given *later* (after outcomes that depend on later
choices), i.e. future information. The control therefore uses a **causal** within-person permutation: at the
k-th rating (k ≥ 1) it receives the value of a uniformly drawn earlier rating j < k, carried forward until the
next rating; the timing columns are unchanged (`models.past_shuffle`, unit-tested to depend only on the past).
It keeps the number of parameters, the input distribution and the person's rating level so far, and destroys
the alignment between a rating and the outcomes that preceded it. `obs − shuf` therefore measures the value of
*timely* affect; `shuf − base` measures person-level rating information.

### 4.4 Fitting, selection, statistics

Script `python -m hpe.population.affect.experiment` (own script because `core.runner` loads only
Psych-101). Fit on all dev participants (`core.nets.train`: Adam, 20 % participant validation split for early
stopping, plateau halving, patience 10, ≤ 150 epochs; batch 64; `threads = 1`), grid hidden ∈ {16, 32}, lr 0.01
(2 configurations per model, ≤ 2,700 CPU-s per model). For each model, (configuration, lapse) is chosen by tune
mean NLL over the thesis lapse grid and reported on tune (the runner's exploratory convention; selection
optimism is negligible with 2 × 7 options and 9,413 participants). Contrasts: per-participant paired
difference, 95 % percentile bootstrap (2,000 resamples, `core.evaluate.paired`). Secondary split: choices
immediately after a new rating vs other choices. Causality: `check_causal` on three tune sequences per
selected model.

### 4.5 Social data (Najar et al. 2020)

Adapter `packages/hpe/population/affect/najar.py`. Steps = the participant's own choices (180 or 240);
demonstrations are not steps. Choices are coded good/bad symbol in the CSV; to stop a population model from
learning "always pick label 1", each (participant, pair) gets a seeded label flip. Five social columns per
step summarise demonstrations seen **before** it on the current pair (any, last action one-hot, share choosing
action 1, number since own last choice on the pair). The control block recomputes them after replacing each
demonstration's action with that of a random *earlier* demonstration of the same person (any pair; labels are
flipped independently per pair, so a borrowed action is uninformative). Models: `social_gru_base`,
`social_gru_obs` (mechanism), `social_gru_shuf` (control).

## 5. Deviations and disclosures

| ID | What | Consequence |
|---|---|---|
| DA-0 | Before the GBE partition was written, `nPlays` quantiles over all 47,067 participants and the field shapes of one record were printed (metadata only; no choice, outcome or rating). Option-value ranges by design version were then computed on dev only. | none for scoring |
| DA-1 | Before the Najar partition was written, per-column value counts over both CSVs (incl. the overall share of good-symbol choices and of rewarded trials across all 68 participants) and the first 12 rows of two participants were printed to learn the layout. One of them (`najar2020_imitation#exp1_3`) fell into test. | it is excluded from any confirmatory scoring (`najar.EXPOSED`); the aggregate shares are disclosed here |
| DA-2 | Human data are loaded by lane adapters, not `core.data.load_family` (AGENTS.md §5 rule 1), because the core loader is hard-wired to the Psych-101 subset. The adapters apply the same `check_partition_access` lock before encoding. | core change request (§9) |
| DA-3 | Abandoned attempts (no result of the abandoned run was produced or seen): (a) a 3-epoch timing pilot on 2,000 dev / 500 tune participants for `affect_gru_obs` and `affect_gru_aux` (hidden 32), scratchpad only — tune NLL 0.508 and 0.509 were displayed; (b) `gbe_explore_01a` with ≤ 10 plays per participant, stopped after 13 CPU-min inside the first configuration because padding every batch to ≤ 300 steps made each model cost ≈ 40 CPU-min; relaunched as `gbe_explore_01` with ≤ 4 plays (≤ 120 trials) from commit `703d567`. | the ≤ 4-plays cap drops the later plays of the 5.8 % of dev participants with more than 4 plays (81.6 % of dev plays are kept) |

## 6. Results (exploratory; filled by the director from `reports/phase3/affect/gbe_explore_01/summary.json`)

GBE, all 18,827 dev participants fitted, all 9,413 tune participants scored, ≤ 4 plays (120 choices) each.
Two configurations per model (hidden 16/32, lr 0.01, batch 64), lapse selected on tune.

| model | tune NLL | top-1 | params |
|---|---|---|---|
| `affect_gru_base` (no affect input) | **0.4737** | 0.7735 | 6,978 |
| `affect_gru_shuf` (causal past-shuffle control) | 0.4740 | 0.7732 | 7,266 |
| `affect_gru_aux_shuf` (mood readout, permuted targets) | 0.4749 | 0.7728 | 7,011 |
| `affect_gru_obs` (observed ratings as input) | 0.4750 | 0.7727 | 2,866 |
| `affect_gru_aux` (latent-mood readout trained on observed ratings) | 0.4756 | 0.7721 | 7,011 |

| contrast (first − second; negative favours first) | estimate [95 % CI], n = 9,413 |
|---|---|
| **Gate F contrast: observed − matched-complexity control** | **+0.0010 [+0.0005, +0.0015]** |
| observed − no affect | +0.0013 [+0.0009, +0.0018] |
| control − no affect | +0.0004 [−0.0001, +0.0008] |
| latent-mood readout: observed − permuted targets | +0.0007 [+0.0002, +0.0012] |
| latent-mood readout − plain GRU | +0.0019 [+0.0015, +0.0024] |

Reported happiness, given as an input or as an auxiliary target, makes held-out risky-choice prediction slightly
**worse**, not better. The `affect_gru_shuf` run used the key-seeded shuffle (before the director's
key-independent fix, commit f9bacc3); its conclusion does not depend on that detail. The Najar 2020 social
models were implemented and tested but **not run** (the lane stopped on the account's API spend limit).

## 7. Gate F status

**Not met on tune; no freeze.** The Gate F tune contrast is +0.0010 (wrong sign) at n = 9,413, so the effect, if
any, is far below the 0.02 margin. Gate F remains testable on GBE (a licensed external dataset); social data
(Najar 2020) remain untested.

## 8. Limits

* GBE choices are described risky choices; the dataset tests whether reported mood helps predict *decisions*
  beyond recent outcomes, not learning. Ratings are self-reports on a slider, every 2–3 trials, in an
  unsupervised smartphone app (random cursor start in versions 3–5, fixed at 50 in versions 1–2).
* The baseline GRU already sees every outcome, so any rating effect it can compute from outcomes (Rutledge's
  CR/EV/RPE model) is not credited to the rating; only the residual is.
* A positive result would show that **observed** human affect carries predictive information; it would say
  nothing about whether a model "has" affect, and no such claim is made.
* Najar: 68 participants, computer-generated demonstrations, MIT software licence; test n = 26 after DA-1.

## 9. Registry rows

Recorded as X3-012 in `EXPERIMENT_REGISTRY_V3.md`. Core change requests: none outstanding.
