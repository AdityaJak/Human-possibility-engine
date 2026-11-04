# x3_b04_adapt_restless4 — Adaptive comparators on the selected pop_gru backbone (fine-tuning, Bayesian head, exemplar mixture, Reptile, CAVIA, hierarchical Bayesian cognitive model), restless4 tune

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: science/validation-v3; preregistration hash `ee0c47db7ccd817f`; fit dev / select tune / report tune; seed 20261108; threads 1
* Hypothesis: (Exploratory.) Tune NLL of conventional individual-adaptation methods on the strongest population backbone; defines the Gate B2 comparator per family.

## restless4 (counted_primary); best comparator on tune: `gru_hb_mix`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gru_hb_mix | comparator | 0.7270 [0.6846, 0.7684] | 174 | 0.725 | 0.078 | 0.885 / 0.830 / 0.649 / 0.539 | 8604 | 0.0 | c | 1.2 | ok |
| bayes_head | comparator | 0.7275 [0.6841, 0.7717] | 174 | 0.726 | 0.073 | 0.890 / 0.826 / 0.648 / 0.540 | 8004 | 0.01 | prior_var, m, predictive | 6.1 | ok |
| ft_head | comparator | 0.7285 [0.6861, 0.7705] | 174 | 0.726 | 0.075 | 0.892 / 0.830 / 0.649 / 0.539 | 8004 | 0.01 | m, n, lr | 27.9 | ok |
| ft_full | comparator | 0.7297 [0.6885, 0.7744] | 174 | 0.724 | 0.076 | 0.893 / 0.831 / 0.650 / 0.540 | 8004 | 0.01 | n, lr, l2 | 0.1 | ok |
| reptile_gru | comparator | 0.7306 [0.6854, 0.7744] | 174 | 0.726 | 0.078 | 0.893 / 0.831 / 0.653 / 0.540 | 8004 | 0.02 | lr, n | 347.6 | ok |
| cavia_gru | comparator | 0.7313 [0.6860, 0.7767] | 174 | 0.724 | 0.079 | 0.894 / 0.833 / 0.652 / 0.542 | 8804 | 0.01 | c_dim, inner_lr | 155.8 | ok |
| exemplar_mix | comparator | 0.7324 [0.6897, 0.7737] | 174 | 0.724 | 0.078 | 0.894 / 0.833 / 0.655 / 0.543 | 8004 | 0.0 | w, bw | 2.3 | ok |
| bayes_head[frozen] | comparator | 0.7325 [0.6897, 0.7753] | 174 | 0.724 | 0.079 | 0.894 / 0.836 / 0.653 / 0.543 | 8004 | 0.01 | prior_var, m, predictive | 6.1 | ok |
| cavia_gru[frozen] | comparator | 0.7325 [0.6884, 0.7757] | 174 | 0.724 | 0.079 | 0.894 / 0.836 / 0.653 / 0.543 | 8804 | 0.01 | c_dim, inner_lr | 155.8 | ok |
| ft_full[frozen] | comparator | 0.7325 [0.6881, 0.7773] | 174 | 0.724 | 0.079 | 0.894 / 0.836 / 0.653 / 0.543 | 8004 | 0.01 | n, lr, l2 | 0.1 | ok |
| ft_head[frozen] | comparator | 0.7325 [0.6896, 0.7762] | 174 | 0.724 | 0.079 | 0.894 / 0.836 / 0.653 / 0.543 | 8004 | 0.01 | m, n, lr | 27.9 | ok |
| cavia_gru[mismatched] | comparator | 0.7328 [0.6890, 0.7768] | 174 | 0.723 | 0.077 | 0.894 / 0.836 / 0.653 / 0.543 | 8804 | 0.01 | c_dim, inner_lr | 155.8 | ok |
| exemplar_mix[frozen] | comparator | 0.7329 [0.6888, 0.7786] | 174 | 0.724 | 0.078 | 0.895 / 0.836 / 0.653 / 0.543 | 8004 | 0.0 | w, bw | 2.3 | ok |
| gru_hb_mix[frozen] | comparator | 0.7333 [0.6919, 0.7750] | 174 | 0.724 | 0.083 | 0.889 / 0.835 / 0.658 / 0.547 | 8604 | 0.0 | c | 1.2 | ok |
| gru_hb_mix[mismatched] | comparator | 0.7335 [0.6897, 0.7772] | 174 | 0.724 | 0.082 | 0.889 / 0.836 / 0.657 / 0.547 | 8604 | 0.0 | c | 1.2 | ok |
| ft_head[mismatched] | comparator | 0.7336 [0.6905, 0.7756] | 174 | 0.723 | 0.080 | 0.895 / 0.837 / 0.654 / 0.544 | 8004 | 0.01 | m, n, lr | 27.9 | ok |
| ft_full[mismatched] | comparator | 0.7337 [0.6905, 0.7773] | 174 | 0.722 | 0.079 | 0.894 / 0.838 / 0.655 / 0.543 | 8004 | 0.01 | n, lr, l2 | 0.1 | ok |
| reptile_gru[frozen] | comparator | 0.7350 [0.6903, 0.7824] | 174 | 0.723 | 0.083 | 0.895 / 0.839 / 0.657 / 0.544 | 8004 | 0.02 | lr, n | 347.6 | ok |
| exemplar_mix[mismatched] | comparator | 0.7352 [0.6936, 0.7799] | 174 | 0.724 | 0.080 | 0.896 / 0.838 / 0.657 / 0.546 | 8004 | 0.0 | w, bw | 2.3 | ok |
| reptile_gru[mismatched] | comparator | 0.7364 [0.6920, 0.7810] | 174 | 0.724 | 0.084 | 0.896 / 0.842 / 0.659 / 0.545 | 8004 | 0.02 | lr, n | 347.6 | ok |
| bayes_head[mismatched] | comparator | 0.7399 [0.6960, 0.7827] | 174 | 0.721 | 0.083 | 0.896 / 0.842 / 0.663 / 0.554 | 8004 | 0.01 | prior_var, m, predictive | 6.1 | ok |
| hb_cog | comparator | 0.8322 [0.7886, 0.8741] | 174 | 0.697 | 0.115 | 0.995 / 0.931 / 0.760 / 0.639 | 599 | 0.0 | c, tau | 7.2 | ok |
| hb_cog[frozen] | comparator | 0.9350 [0.8842, 0.9888] | 174 | 0.671 | 0.165 | 1.077 / 1.017 / 0.883 / 0.759 | 599 | 0.0 | c, tau | 7.2 | ok |
| hb_cog[mismatched] | comparator | 0.9981 [0.9440, 1.0550] | 174 | 0.643 | 0.174 | 1.095 / 1.094 / 0.960 / 0.840 | 599 | 0.0 | c, tau | 7.2 | ok |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
