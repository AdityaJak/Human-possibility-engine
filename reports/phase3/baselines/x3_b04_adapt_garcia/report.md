# x3_b04_adapt_garcia — Adaptive comparators on the selected pop_gru backbone (fine-tuning, Bayesian head, exemplar mixture, Reptile, CAVIA, hierarchical Bayesian cognitive model), garcia tune

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: science/validation-v3; preregistration hash `9f37491d2ee62155`; fit dev / select tune / report tune; seed 20261108; threads 1
* Hypothesis: (Exploratory.) Tune NLL of conventional individual-adaptation methods on the strongest population backbone; defines the Gate B2 comparator per family.

## garcia (counted); best comparator on tune: `ft_full`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ft_full | comparator | 0.4380 [0.4054, 0.4718] | 68 | 0.777 | 0.054 | 0.542 / 0.497 / 0.398 / 0.315 | 3236 | 0.02 | n, lr, l2 | 0.1 | ok |
| bayes_head | comparator | 0.4392 [0.4049, 0.4732] | 68 | 0.771 | 0.058 | 0.542 / 0.497 / 0.400 / 0.317 | 3236 | 0.01 | m, predictive | 1.7 | ok |
| ft_head | comparator | 0.4448 [0.4098, 0.4788] | 68 | 0.768 | 0.058 | 0.548 / 0.498 / 0.402 / 0.331 | 3236 | 0.02 | m, n, lr | 18.8 | ok |
| reptile_gru | comparator | 0.4454 [0.4101, 0.4813] | 68 | 0.774 | 0.063 | 0.557 / 0.502 / 0.405 / 0.317 | 3236 | 0.02 | lr, n | 472.5 | ok |
| exemplar_mix | comparator | 0.4489 [0.4139, 0.4825] | 68 | 0.773 | 0.062 | 0.541 / 0.501 / 0.407 / 0.347 | 3236 | 0.0 | bw | 2.7 | ok |
| cavia_gru | comparator | 0.4501 [0.4128, 0.4859] | 68 | 0.772 | 0.063 | 0.541 / 0.501 / 0.409 / 0.349 | 3444 | 0.02 | c_dim, inner_lr | 77.6 | ok |
| gru_hb_mix | comparator | 0.4505 [0.4159, 0.4845] | 68 | 0.773 | 0.065 | 0.540 / 0.500 / 0.408 / 0.354 | 3836 | 0.01 | w, c | 1.0 | ok |
| cavia_gru[mismatched] | comparator | 0.4506 [0.4161, 0.4860] | 68 | 0.772 | 0.065 | 0.541 / 0.501 / 0.409 / 0.351 | 3444 | 0.02 | c_dim, inner_lr | 77.6 | ok |
| cavia_gru[frozen] | comparator | 0.4509 [0.4148, 0.4863] | 68 | 0.772 | 0.066 | 0.541 / 0.501 / 0.410 / 0.351 | 3444 | 0.02 | c_dim, inner_lr | 77.6 | ok |
| ft_full[frozen] | comparator | 0.4509 [0.4139, 0.4863] | 68 | 0.772 | 0.066 | 0.541 / 0.501 / 0.410 / 0.351 | 3236 | 0.02 | n, lr, l2 | 0.1 | ok |
| ft_head[frozen] | comparator | 0.4509 [0.4147, 0.4879] | 68 | 0.772 | 0.066 | 0.541 / 0.501 / 0.410 / 0.351 | 3236 | 0.02 | m, n, lr | 18.8 | ok |
| bayes_head[frozen] | comparator | 0.4511 [0.4146, 0.4887] | 68 | 0.772 | 0.066 | 0.542 / 0.503 / 0.410 / 0.350 | 3236 | 0.01 | m, predictive | 1.7 | ok |
| gru_hb_mix[frozen] | comparator | 0.4516 [0.4177, 0.4865] | 68 | 0.773 | 0.065 | 0.541 / 0.501 / 0.410 / 0.355 | 3836 | 0.01 | w, c | 1.0 | ok |
| exemplar_mix[frozen] | comparator | 0.4517 [0.4159, 0.4878] | 68 | 0.772 | 0.065 | 0.543 / 0.504 / 0.410 / 0.350 | 3236 | 0.0 | bw | 2.7 | ok |
| gru_hb_mix[mismatched] | comparator | 0.4517 [0.4177, 0.4880] | 68 | 0.773 | 0.066 | 0.541 / 0.501 / 0.409 / 0.355 | 3836 | 0.01 | w, c | 1.0 | ok |
| exemplar_mix[mismatched] | comparator | 0.4529 [0.4185, 0.4866] | 68 | 0.773 | 0.065 | 0.542 / 0.503 / 0.411 / 0.356 | 3236 | 0.0 | bw | 2.7 | ok |
| bayes_head[mismatched] | comparator | 0.4576 [0.4201, 0.4935] | 68 | 0.771 | 0.067 | 0.543 / 0.509 / 0.414 / 0.364 | 3236 | 0.01 | m, predictive | 1.7 | ok |
| ft_full[mismatched] | comparator | 0.4578 [0.4197, 0.4947] | 68 | 0.771 | 0.068 | 0.544 / 0.512 / 0.415 / 0.360 | 3236 | 0.02 | n, lr, l2 | 0.1 | ok |
| ft_head[mismatched] | comparator | 0.4590 [0.4232, 0.4962] | 68 | 0.770 | 0.071 | 0.551 / 0.514 / 0.415 / 0.356 | 3236 | 0.02 | m, n, lr | 18.8 | ok |
| reptile_gru[frozen] | comparator | 0.4596 [0.4210, 0.5009] | 68 | 0.771 | 0.078 | 0.552 / 0.509 / 0.419 / 0.358 | 3236 | 0.02 | lr, n | 472.5 | ok |
| reptile_gru[mismatched] | comparator | 0.4707 [0.4319, 0.5092] | 68 | 0.770 | 0.076 | 0.558 / 0.531 / 0.427 / 0.366 | 3236 | 0.02 | lr, n | 472.5 | ok |
| hb_cog | comparator | 0.5803 [0.5557, 0.6042] | 68 | 0.652 | 0.076 | 0.535 / 0.503 / 0.581 / 0.702 | 599 | 0.05 | tau | 3.8 | ok |
| hb_cog[frozen] | comparator | 0.6190 [0.5912, 0.6483] | 68 | 0.643 | 0.126 | 0.581 / 0.557 / 0.626 / 0.712 | 599 | 0.05 | tau | 3.8 | ok |
| hb_cog[mismatched] | comparator | 0.6418 [0.6138, 0.6712] | 68 | 0.646 | 0.123 | 0.605 / 0.611 / 0.633 / 0.719 | 599 | 0.05 | tau | 3.8 | ok |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
