# x3_b03_pop_garcia — Population-only comparators (pop_gru, pop_lstm 24 configs; pop_gru_ens; pop_tf; ref_gru_p2) on garcia, tune partition

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: science/validation-v3; preregistration hash `4538e9d8d4897e46`; fit dev / select tune / report tune; seed 20261108; threads 2
* Hypothesis: (Exploratory.) Tune NLL of the population-only pool members under equal caps on garcia.

## garcia (counted); best comparator on tune: `pop_tf`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_tf | comparator | 0.4339 [0.4005, 0.4685] | 68 | 0.776 | 0.061 | 0.527 / 0.501 / 0.394 / 0.314 | 19108 | 0.02 | window, lr, dropout | 293.9 | ok |
| ref_gru_p2 | comparator | 0.4403 [0.4071, 0.4730] | 68 | 0.771 | 0.064 | 0.546 / 0.505 / 0.395 / 0.316 | 666 | 0.0 | hidden, lr | 83.8 | ok |
| pop_gru_ens | comparator | 0.4403 [0.4043, 0.4751] | 68 | 0.776 | 0.061 | 0.537 / 0.498 / 0.394 / 0.332 | 40020 | 0.0 | hidden, lr, dropout | 1145.9 | ok |
| pop_gru | comparator | 0.4509 [0.4154, 0.4865] | 68 | 0.772 | 0.066 | 0.541 / 0.501 / 0.410 / 0.351 | 3236 | 0.02 | hidden, lr | 912.6 | ok |
| pop_lstm | comparator | 0.4596 [0.4236, 0.4961] | 68 | 0.768 | 0.066 | 0.556 / 0.520 / 0.411 / 0.351 | 91652 | 0.02 | hidden, lr | 151.6 | ok |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
