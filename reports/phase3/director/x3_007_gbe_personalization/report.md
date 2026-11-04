# x3_007_gbe_personalization — External dataset (Great Brain Experiment risky choice) — population GRU, seed ensemble, F3, F1 bolt-on latent, Bayesian head

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: director; preregistration hash `fcb9996567e9138f`; fit dev / select tune / report tune; seed 20261110; threads 2
* Hypothesis: (Exploratory.) With stable individual risk preferences and repeated plays, explicit individual adaptation adds more over the population GRU than on the Psych-101 learning tasks (where it added 0.004-0.009 nats/choice).

## gbe_risk (external); best comparator on tune: `pop_gru_ens`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_gru_ens | comparator | 0.4826 [0.4739, 0.4911] | 1000 | 0.770 | 0.114 | 0.547 / 0.488 / 0.461 / 0.427 | 34890 | 0.0 | — | 707.1 | ok |
| pop_gru | comparator | 0.4892 [0.4806, 0.4981] | 1000 | 0.766 | 0.115 | 0.548 / 0.496 / 0.470 / 0.436 | 2722 | 0.02 | hidden | 411.8 | ok |
| f3_arbiter[twin] | candidate | 0.4909 [0.4822, 0.4995] | 1000 | 0.765 | 0.116 | 0.549 / 0.499 / 0.470 / 0.439 | 6978 | 0.0 | bandwidth | 137.7 | ok |
| f3_arbiter | candidate | 0.4934 [0.4851, 0.5015] | 1000 | 0.764 | 0.117 | 0.552 / 0.499 / 0.472 / 0.445 | 6984 | 0.0 | bandwidth | 137.7 | ok |
| bayes_head | comparator | 0.4935 [0.4853, 0.5021] | 1000 | 0.763 | 0.115 | 0.553 / 0.502 / 0.472 / 0.440 | 6978 | 0.01 | — | 141.5 | ok |
| f1_bolt | candidate | 0.4936 [0.4855, 0.5019] | 1000 | 0.764 | 0.118 | 0.552 / 0.499 / 0.472 / 0.444 | 7434 | 0.0 | — | 0.0 | ok |
| f3_arbiter[frozen] | candidate | 0.4940 [0.4853, 0.5027] | 1000 | 0.764 | 0.118 | 0.552 / 0.499 / 0.473 / 0.445 | 6984 | 0.0 | bandwidth | 137.7 | ok |
| f1_bolt[frozen] | candidate | 0.4940 [0.4851, 0.5022] | 1000 | 0.764 | 0.118 | 0.552 / 0.499 / 0.473 / 0.445 | 7434 | 0.0 | — | 0.0 | ok |
| f1_bolt[mismatched] | candidate | 0.4940 [0.4854, 0.5026] | 1000 | 0.763 | 0.117 | 0.552 / 0.499 / 0.473 / 0.445 | 7434 | 0.0 | — | 0.0 | ok |
| bayes_head[frozen] | comparator | 0.4955 [0.4871, 0.5039] | 1000 | 0.761 | 0.117 | 0.553 / 0.503 / 0.474 / 0.445 | 6978 | 0.01 | — | 141.5 | ok |
| f3_arbiter[mismatched] | candidate | 0.4959 [0.4881, 0.5042] | 1000 | 0.762 | 0.119 | 0.554 / 0.501 / 0.474 / 0.448 | 6984 | 0.0 | bandwidth | 137.7 | ok |
| bayes_head[mismatched] | comparator | 0.4979 [0.4897, 0.5061] | 1000 | 0.760 | 0.117 | 0.553 / 0.505 / 0.476 / 0.451 | 6978 | 0.01 | — | 141.5 | ok |

| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | sd | n |
|---|---|---|---|---|---|---|
| f1_bolt − pop_gru_ens | vs_best_comparator | +0.0111 | [+0.0094, +0.0127] | 1.0000 | 0.0272 | 1000 |
| f1_bolt − f1_bolt[frozen] | vs_frozen | -0.0004 | [-0.0005, -0.0003] | 0.0005 | 0.0021 | 1000 |
| f1_bolt − f1_bolt[mismatched] | vs_mismatched | -0.0004 | [-0.0006, -0.0002] | 0.0005 | 0.0028 | 1000 |
| f1_bolt − bayes_head | vs_comparator | +0.0001 | [-0.0020, +0.0021] | 0.5455 | 0.0339 | 1000 |
| f1_bolt − pop_gru | vs_comparator | +0.0044 | [+0.0023, +0.0065] | 1.0000 | 0.0345 | 1000 |
| f3_arbiter − pop_gru_ens | vs_best_comparator | +0.0108 | [+0.0093, +0.0125] | 1.0000 | 0.0265 | 1000 |
| f3_arbiter − f3_arbiter[twin] | vs_twin | +0.0025 | [+0.0003, +0.0047] | 0.9885 | 0.0359 | 1000 |
| f3_arbiter − f3_arbiter[frozen] | vs_frozen | -0.0006 | [-0.0010, -0.0002] | 0.0020 | 0.0067 | 1000 |
| f3_arbiter − f3_arbiter[mismatched] | vs_mismatched | -0.0025 | [-0.0031, -0.0018] | 0.0005 | 0.0103 | 1000 |
| f3_arbiter − bayes_head | vs_comparator | -0.0001 | [-0.0022, +0.0018] | 0.4325 | 0.0332 | 1000 |
| f3_arbiter − pop_gru | vs_comparator | +0.0042 | [+0.0021, +0.0062] | 1.0000 | 0.0340 | 1000 |

## Candidate vs best comparator across counted families (Holm)

| candidate | family | estimate | 95% CI | Holm p | meets 0.02 margin |
|---|---|---|---|---|---|
| f1_bolt | gbe_risk | +0.0111 | [+0.0094, +0.0127] | — | False |
| f3_arbiter | gbe_risk | +0.0108 | [+0.0093, +0.0125] | — | False |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
