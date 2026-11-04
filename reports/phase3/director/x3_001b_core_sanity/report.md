# x3_001b_core_sanity — Torch population GRU (encoding v3, trainer v2) vs the Phase II numpy GRU, tune partition

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: director; preregistration hash `0cb23791d9ac871f`; fit dev / select tune / report tune; seed 20261107; threads 2
* Hypothesis: (Exploratory.) pop_gru with the v3 encoding is at least as good as the Phase II ref_gru on tune.

## bandit2 (secondary); best comparator on tune: `pop_gru`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_gru | comparator | 0.3600 [0.2931, 0.4352] | 16 | 0.831 | 0.054 | 0.438 / 0.326 / 0.316 / 0.361 | 2284 | 0.0 | hidden, lr, dropout | 377.6 | ok |

## described (counted); best comparator on tune: `pop_gru`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_gru | comparator | 0.3556 [0.3135, 0.3968] | 41 | 0.839 | 0.032 | 0.385 / 0.364 / 0.357 / 0.316 | 8004 | 0.0 | hidden, lr, dropout | 1296.0 | ok |

## garcia (counted); best comparator on tune: `pop_gru`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_gru | comparator | 0.4415 [0.4070, 0.4763] | 68 | 0.772 | 0.060 | 0.535 / 0.500 / 0.400 / 0.331 | 2284 | 0.0 | hidden, lr, dropout | 211.6 | ok |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
