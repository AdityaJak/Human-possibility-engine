# Phase III Research Results

_Compiled by the Research Director on 2026-11-04 from the exploratory ledger (`EXPERIMENT_REGISTRY_V3.md`) and
the committed result directories named there. **Every number below is exploratory**: fitted on development
participants, selected and reported on tuning participants (selection-optimistic) or on development seeds of the
synthetic suite. **No test partition was opened** and no confirmatory run was performed, because no candidate met
a preregistered stop rule. The independent critic recomputed the central contrasts from the committed
per-participant scores and refitted seeds (`INDEPENDENT_SCIENTIFIC_AUDIT.md`, results review)._

## 1. The question and the answer

**Question (H3).** Does a model that learns across people *and* explicitly adapts an individual representation
predict held-out human choices better than strong population-only and adaptive models?

**Answer on the tested data: no, not at the preregistered effect size.** On single-session learning tasks a
population sequence model already infers the person from context. Explicit person-level adaptation added at most
0.009 nats/choice to a single model and 0.001–0.003 to a seed ensemble — inside the seed noise of single fits —
while the largest gains in the programme came from **variance reduction (seed ensembles, ≈ 0.02)** and
**architecture (a causal transformer on garcia)**.

## 2. Primary family (restless 4-armed bandit, Bahrami & Navajas 2020; 348 dev / 174 tune people)

| model (tune NLL; chance 1.386) | NLL | role |
|---|---|---|
| Bayesian-head adapter, 5-seed ensemble (X3-007) | **0.7096** | comparator (matched-ensemble control) |
| `pop_gru_ens`, 5 seeds (X3-003) | 0.7102 | strongest comparator of the preregistered pool |
| F3 arbiter, 5-seed ensemble (X3-007) | 0.7113 | candidate (matched-ensemble control) |
| F3 episodic memory + learned arbitration (X3-005) | 0.7237 | best single candidate |
| causal transformer `pop_tf` | 0.7263 | comparator |
| `pop_lstm` | 0.7268 | comparator |
| best conventional adapter (`gru_hb_mix` / `bayes_head`) | 0.7270 / 0.7275 | comparator (Gate B2) |
| F1 individual latent | 0.7280 | candidate |
| `pop_gru` (24 configurations) | 0.7325 | comparator |
| Phase II numpy GRU | 0.7751 | reference |
| F2 world model + individual policy | 0.8306 | candidate |
| hierarchical Bayesian cognitive model `hb_cog` | 0.8322 | comparator |

Key contrasts (critic-reproduced, 20,000 resamples; negative favours the first model):

| contrast | estimate [95 % CI] |
|---|---|
| F3 − strongest comparator (`pop_gru_ens`) | **+0.0135 [+0.0082, +0.0183]** |
| F3 − `pop_gru` (24 configs) | −0.0088 [−0.0156, −0.0024] |
| F3 ensemble − `pop_gru` ensemble (matched) | −0.0014 [−0.0048, +0.0019] |
| F3 ensemble − its own frozen version | −0.0009 [−0.0056, +0.0025] |
| F3 ensemble − donor (mismatched person) | −0.0069 [−0.0110, −0.0037] |

## 3. Other families

| family | strongest comparator (tune NLL) | best candidate (tune NLL) | candidate − strongest comparator |
|---|---|---|---|
| garcia (Garcia et al. 2023; 68 tune) | `pop_tf` 0.4339 | F3 0.4348 | +0.0009 |
| described (Plonsky et al. 2018; 41 tune) | `pop_gru` 0.3550 | F1 0.3568 | +0.0018 [−0.0008, +0.0042] |
| **external: Great Brain Experiment risky choice** (CC0; 1,000 tune) | `pop_gru_ens` 0.4826 | F3 0.4934 | +0.0108 [+0.0093, +0.0125] |

On the external dataset, where individuals have stable risk preferences, explicit adaptation versus the same model
with adaptation frozen was −0.0004 to −0.0020: the in-context GRU already captures stable preferences within 120
choices. On garcia the largest *conventional* adaptation gain of the programme appeared (full online fine-tuning
−0.013 versus its frozen backbone), still behind the transformer. `two_step` was never counted (critic sign-off
missing) and the described-gambles comparator pool was not run.

## 4. Gate status (all from tune / development evidence; no confirmatory run)

| gate | tune / development evidence | stop rule | status |
|---|---|---|---|
| A — learned capability | own-history adaptation beats a length-matched donor's in restless4, garcia (F3 −0.0069 ensembled; adapters similar); A1 (outcome-permuted training) **not run** | — | not evaluated (no joint freeze) |
| B — individual adaptation | B1 nominal tune pass for F3 vs its twin (−0.015 / −0.019) but twin biased (critic N-3); **B2 fails on the primary** (−0.0034 vs 0.005 margin); matched-ensemble gain −0.0009 | not met | **not passed** |
| C — human behavioural prediction | best candidate behind or level with the strongest comparator in every counted family | not met | **not passed** |
| D — cross-domain transfer | pretraining on other families helps a 16-person target sample (restless4: pretrained `pop_gru` 0.8498 vs specialist 0.8781), but the meta-learned candidate is no better than plain pretraining (0.8522) — see `TRANSFER_LEARNING_RESULTS.md` | not met | **not passed** |
| E — continual learning | F4 earns the most reward (0.829 of oracle) but shows no savings on return (−0.024 vs replay +0.033) | not met | **not passed** |
| F — affect / social | observed happiness ratings worsen choice prediction slightly (+0.0010 [+0.0005, +0.0015], n = 9,413); social data not run | not met | **not passed** |
| G — causal decision support | requires a prospective randomised study (`CAUSAL_DECISION_RESEARCH.md`) | — | **not started** |

## 5. What was learned (reported with the same prominence as failures)

1. **Seed variance is as large as the effects under study.** Single fits of one GRU configuration differ by SD
   0.003 (restless4) to 0.009 (garcia; range 0.031). Any future confirmatory comparison must ensemble or
   seed-bootstrap every gate-bearing model (critic N-4).
2. **Ensembling beats personalisation.** A 5-seed population ensemble gained 0.022 over a single GRU on the
   primary family — more than any adaptation mechanism.
3. **In-context adaptation is strong.** Explicit person states add ≤ 0.009 to a single recurrent model and
   ≈ 0.001–0.003 to an ensemble on single-session tasks, including the external dataset.
4. **Where person information *is* valuable: across sessions.** When the recurrent state is reset between plays,
   earlier-session history is worth +0.0145 to +0.0195 nats/choice on the external dataset (X3-013). That is the
   setting of a real decision-support product (people return in separate sessions) and is the pivot question.
5. **Cross-family pretraining is a useful population prior at small samples** (restless4, k = 16), but
   first-order meta-learning added nothing on top of it.
6. **Mechanisms that did not survive:** F1's trait/context split (meta-training equals a bolt-on latent), F2's
   world-model/policy separation (0.03–0.14 worse), F4's in-context continual learner (no retention), affect
   inputs (no gain).

## 6. Compute

≈ 10 CPU-hours recorded across Phase III result artifacts (model runs 8.7 h including the affect lane; Gate E
0.44 h; Gate D and the session-headroom run in their logs), out of a 60 CPU-hour budget; 0 GPU; 0 language-model
calls in any model. Four of five research lanes were stopped by the account's API spend limit; the director
resumed their definitions unchanged and committed their results.

## 7. Artifacts

Registry rows X3-001 … X3-014; result directories under `reports/phase3/{director,baselines,candidates,continual,affect,critic}/`;
reports `POPULATION_ADAPTATION_ARCHITECTURES.md`, `ADAPTATION_BENCHMARKS.md`, `HUMAN_DATA_VALIDATION.md`,
`TRANSFER_LEARNING_RESULTS.md`, `AFFECTIVE_SOCIAL_VALIDATION.md`, `CAUSAL_DECISION_RESEARCH.md`,
`INDEPENDENT_SCIENTIFIC_AUDIT.md`, `BREAKTHROUGH_VERDICT.md`.
