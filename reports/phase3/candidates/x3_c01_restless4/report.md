# x3_c01_restless4 — Design-team candidates (F3 episodic arbiter, F2 world model + policy, F1 individual latent) vs same-size pop_gru, restless4

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: research/adaptation; preregistration hash `5cc5e37f05bee729`; fit dev / select tune / report tune; seed 20261108; threads 2
* Hypothesis: (Exploratory.) Explicit individual adaptation (F3 episodic memory + learned gate, F2 individual policy posterior, F1 online latent) improves tune NLL over a same-size pop_gru twin and over the same population network with its recurrent state running.

## restless4 (counted_primary); best comparator on tune: `pop_gru`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f3_arbiter | candidate | 0.7237 [0.6810, 0.7663] | 174 | 0.727 | 0.074 | 0.871 / 0.829 / 0.650 / 0.540 | 8010 | 0.0 | hidden, dropout, bandwidth | 902.8 | ok |
| f1_latent | candidate | 0.7280 [0.6845, 0.7715] | 174 | 0.727 | 0.078 | 0.879 / 0.832 / 0.654 / 0.542 | 8724 | 0.02 | hidden, dropout | 489.3 | ok |
| f1_bolt | reference | 0.7280 [0.6836, 0.7728] | 174 | 0.728 | 0.079 | 0.880 / 0.833 / 0.651 / 0.543 | 8724 | 0.02 | hidden, dropout | 0.0 | ok |
| f3_fixedmix | reference | 0.7285 [0.6848, 0.7739] | 174 | 0.727 | 0.079 | 0.877 / 0.833 / 0.655 / 0.545 | 8005 | 0.0 | hidden, dropout | 16.2 | ok |
| f1_bolt[frozen] | reference | 0.7293 [0.6844, 0.7739] | 174 | 0.727 | 0.082 | 0.881 / 0.836 / 0.653 / 0.543 | 8724 | 0.02 | hidden, dropout | 0.0 | ok |
| f1_latent[frozen] | candidate | 0.7293 [0.6840, 0.7729] | 174 | 0.727 | 0.082 | 0.881 / 0.836 / 0.653 / 0.543 | 8724 | 0.02 | hidden, dropout | 489.3 | ok |
| f3_gruepi | reference | 0.7293 [0.6843, 0.7719] | 174 | 0.727 | 0.082 | 0.881 / 0.836 / 0.653 / 0.543 | 16488 | 0.02 | hidden, dropout | 139.4 | ok |
| f3_arbiter[frozen] | candidate | 0.7302 [0.6878, 0.7754] | 174 | 0.727 | 0.079 | 0.883 / 0.836 / 0.653 / 0.544 | 8010 | 0.0 | hidden, dropout, bandwidth | 902.8 | ok |
| f3_fixedmix[frozen] | reference | 0.7302 [0.6869, 0.7758] | 174 | 0.727 | 0.079 | 0.883 / 0.836 / 0.653 / 0.544 | 8005 | 0.0 | hidden, dropout | 16.2 | ok |
| f3_fixedshare[frozen] | reference | 0.7302 [0.6855, 0.7784] | 174 | 0.727 | 0.079 | 0.883 / 0.836 / 0.653 / 0.544 | 8004 | 0.0 | hidden, dropout | 0.1 | ok |
| f1_latent[twin] | candidate | 0.7304 [0.6848, 0.7734] | 174 | 0.726 | 0.080 | 0.885 / 0.830 / 0.661 / 0.542 | 8004 | 0.02 | hidden, dropout | 489.3 | ok |
| f1_latent[mismatched] | candidate | 0.7310 [0.6862, 0.7755] | 174 | 0.726 | 0.082 | 0.880 / 0.838 / 0.655 / 0.546 | 8724 | 0.02 | hidden, dropout | 489.3 | ok |
| f3_arbiter[mismatched] | candidate | 0.7314 [0.6889, 0.7766] | 174 | 0.725 | 0.081 | 0.883 / 0.838 / 0.655 / 0.545 | 8010 | 0.0 | hidden, dropout, bandwidth | 902.8 | ok |
| pop_gru | comparator | 0.7322 [0.6858, 0.7770] | 174 | 0.726 | 0.080 | 0.889 / 0.834 / 0.662 / 0.540 | 8004 | 0.01 | hidden, dropout | 320.0 | ok |
| f2_world_policy[twin] | candidate | 0.7364 [0.6922, 0.7790] | 174 | 0.723 | 0.077 | 0.893 / 0.843 / 0.655 / 0.549 | 8004 | 0.0 | hidden, dropout, prior_scale | 1404.7 | ok |
| f3_fixedshare | reference | 0.7368 [0.6945, 0.7801] | 174 | 0.720 | 0.071 | 0.890 / 0.835 / 0.664 / 0.554 | 8004 | 0.0 | hidden, dropout | 0.1 | ok |
| f3_arbiter[twin] | candidate | 0.7387 [0.6948, 0.7810] | 174 | 0.719 | 0.080 | 0.880 / 0.849 / 0.667 / 0.554 | 8004 | 0.0 | hidden, dropout, bandwidth | 902.8 | ok |
| f2_delta | reference | 0.8034 [0.7591, 0.8486] | 174 | 0.701 | 0.089 | 0.971 / 0.905 / 0.717 / 0.616 | 36 | 0.0 | delta_alpha | 3.4 | ok |
| f2_world_policy | candidate | 0.8306 [0.7879, 0.8737] | 174 | 0.697 | 0.083 | 1.010 / 0.926 / 0.746 / 0.635 | 8171 | 0.0 | hidden, dropout, prior_scale | 1404.7 | ok |
| f2_delta[frozen] | reference | 0.8701 [0.8186, 0.9216] | 174 | 0.675 | 0.113 | 1.029 / 0.970 / 0.785 / 0.692 | 36 | 0.0 | delta_alpha | 3.4 | ok |
| f2_world_policy[frozen] | candidate | 0.8791 [0.8279, 0.9312] | 174 | 0.673 | 0.103 | 1.049 / 0.973 / 0.797 / 0.693 | 8171 | 0.0 | hidden, dropout, prior_scale | 1404.7 | ok |
| f2_world_policy[mismatched] | candidate | 1.0950 [1.0011, 1.2011] | 174 | 0.612 | 0.145 | 1.186 / 1.188 / 1.039 / 0.964 | 8171 | 0.0 | hidden, dropout, prior_scale | 1404.7 | ok |

| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | sd | n |
|---|---|---|---|---|---|---|
| f1_latent − pop_gru | vs_best_comparator | -0.0043 | [-0.0099, +0.0013] | 0.0670 | 0.0376 | 174 |
| f1_latent − f1_latent[twin] | vs_twin | -0.0025 | [-0.0075, +0.0028] | 0.1610 | 0.0346 | 174 |
| f1_latent − f1_latent[frozen] | vs_frozen | -0.0013 | [-0.0030, +0.0002] | 0.0450 | 0.0106 | 174 |
| f1_latent − f1_latent[mismatched] | vs_mismatched | -0.0030 | [-0.0049, -0.0013] | 0.0005 | 0.0123 | 174 |
| f2_world_policy − pop_gru | vs_best_comparator | +0.0984 | [+0.0855, +0.1114] | 1.0000 | 0.0882 | 174 |
| f2_world_policy − f2_world_policy[twin] | vs_twin | +0.0943 | [+0.0804, +0.1068] | 1.0000 | 0.0882 | 174 |
| f2_world_policy − f2_world_policy[frozen] | vs_frozen | -0.0484 | [-0.0677, -0.0329] | 0.0005 | 0.1161 | 174 |
| f2_world_policy − f2_world_policy[mismatched] | vs_mismatched | -0.2644 | [-0.3744, -0.1737] | 0.0005 | 0.6687 | 174 |
| f3_arbiter − pop_gru | vs_best_comparator | -0.0086 | [-0.0153, -0.0024] | 0.0020 | 0.0432 | 174 |
| f3_arbiter − f3_arbiter[twin] | vs_twin | -0.0150 | [-0.0210, -0.0096] | 0.0005 | 0.0395 | 174 |
| f3_arbiter − f3_arbiter[frozen] | vs_frozen | -0.0065 | [-0.0120, -0.0024] | 0.0005 | 0.0335 | 174 |
| f3_arbiter − f3_arbiter[mismatched] | vs_mismatched | -0.0077 | [-0.0120, -0.0044] | 0.0005 | 0.0249 | 174 |

## Candidate vs best comparator across counted families (Holm)

| candidate | family | estimate | 95% CI | Holm p | meets 0.02 margin |
|---|---|---|---|---|---|
| f1_latent | restless4 | -0.0043 | [-0.0099, +0.0013] | 0.0670 | False |
| f2_world_policy | restless4 | +0.0984 | [+0.0855, +0.1114] | 1.0000 | False |
| f3_arbiter | restless4 | -0.0086 | [-0.0153, -0.0024] | 0.0020 | False |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
