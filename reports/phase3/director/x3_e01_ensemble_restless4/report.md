# x3_e01_ensemble_restless4 — Matched-ensemble control — 5-member ensembles of pop_gru, the F3 arbiter and the Bayesian-head adapter, restless4 tune

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: director; preregistration hash `b47f8d7f50a4207f`; fit dev / select tune / report tune; seed 20261109; threads 1
* Hypothesis: (Exploratory.) If individual adaptation adds value beyond variance reduction, the F3 and bayes_head ensembles beat pop_gru_ens by about the single-model gain (0.004–0.009 nats/choice).

## restless4 (counted_primary); best comparator on tune: `bayes_head_ens`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_head_ens | comparator | 0.7096 [0.6697, 0.7511] | 174 | 0.733 | 0.070 | 0.863 / 0.813 / 0.633 / 0.526 | 40020 | 0.0 | — | 105.8 | ok |
| f3_ens | candidate | 0.7113 [0.6711, 0.7516] | 174 | 0.732 | 0.071 | 0.860 / 0.813 / 0.637 / 0.532 | 40050 | 0.0 | — | 174.6 | ok |
| f3_ens[frozen] | candidate | 0.7122 [0.6701, 0.7538] | 174 | 0.733 | 0.076 | 0.864 / 0.814 / 0.635 / 0.532 | 40050 | 0.0 | — | 174.6 | ok |
| pop_gru_ens | comparator | 0.7127 [0.6720, 0.7559] | 174 | 0.731 | 0.074 | 0.863 / 0.814 / 0.638 / 0.532 | 40020 | 0.0 | — | 116.7 | ok |
| bayes_head_ens[frozen] | comparator | 0.7132 [0.6714, 0.7568] | 174 | 0.732 | 0.076 | 0.866 / 0.820 / 0.634 / 0.528 | 40020 | 0.0 | — | 105.8 | ok |
| f3_ens[mismatched] | candidate | 0.7182 [0.6797, 0.7587] | 174 | 0.731 | 0.078 | 0.868 / 0.820 / 0.643 / 0.538 | 40050 | 0.0 | — | 174.6 | ok |

| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | sd | n |
|---|---|---|---|---|---|---|
| f3_ens − bayes_head_ens | vs_best_comparator | +0.0017 | [-0.0017, +0.0050] | 0.8360 | 0.0226 | 174 |
| f3_ens − f3_ens[frozen] | vs_frozen | -0.0009 | [-0.0056, +0.0025] | 0.3495 | 0.0276 | 174 |
| f3_ens − f3_ens[mismatched] | vs_mismatched | -0.0069 | [-0.0110, -0.0037] | 0.0005 | 0.0257 | 174 |
| f3_ens − pop_gru_ens | vs_comparator | -0.0014 | [-0.0048, +0.0018] | 0.2065 | 0.0224 | 174 |

## Candidate vs best comparator across counted families (Holm)

| candidate | family | estimate | 95% CI | Holm p | meets 0.02 margin |
|---|---|---|---|---|---|
| f3_ens | restless4 | +0.0017 | [-0.0017, +0.0050] | 0.8360 | False |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
