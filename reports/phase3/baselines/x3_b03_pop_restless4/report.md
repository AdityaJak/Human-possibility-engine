# x3_b03_pop_restless4 — Population-only comparators (pop_gru, pop_lstm 24 configs; pop_gru_ens; pop_tf; ref_gru_p2) on restless4, tune partition

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: science/validation-v3; preregistration hash `a2413b681c1a0b16`; fit dev / select tune / report tune; seed 20261108; threads 2
* Hypothesis: (Exploratory.) Tune NLL of the population-only pool members under equal caps on restless4.

## restless4 (counted_primary); best comparator on tune: `pop_gru_ens`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_gru_ens | comparator | 0.7102 [0.6691, 0.7534] | 174 | 0.732 | 0.074 | 0.864 / 0.810 / 0.635 / 0.528 | 40020 | 0.0 | hidden, lr, dropout | 2669.8 | ok |
| pop_tf | comparator | 0.7263 [0.6828, 0.7685] | 174 | 0.726 | 0.081 | 0.878 / 0.829 / 0.655 / 0.539 | 19108 | 0.02 | window, lr, dropout | 937.7 | ok |
| pop_lstm | comparator | 0.7268 [0.6826, 0.7698] | 174 | 0.728 | 0.079 | 0.883 / 0.828 / 0.652 / 0.540 | 4292 | 0.01 | hidden, lr, dropout | 432.6 | ok |
| pop_gru | comparator | 0.7325 [0.6865, 0.7767] | 174 | 0.724 | 0.079 | 0.894 / 0.836 / 0.653 / 0.543 | 8004 | 0.01 | lr, dropout | 2657.8 | ok |
| ref_gru_p2 | comparator | 0.7751 [0.7304, 0.8199] | 174 | 0.706 | 0.084 | 0.926 / 0.884 / 0.706 / 0.580 | 1312 | 0.01 | hidden, lr | 167.5 | ok |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
