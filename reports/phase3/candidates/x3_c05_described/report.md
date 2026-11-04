# x3_c05_described — Design-team candidates (F3 episodic arbiter, F2 world model + policy, F1 individual latent) vs same-size pop_gru, described

> **EXPLORATORY (dev/tune; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL per free choice (nats). Not evidence about any individual's future.

* Lane: research/adaptation; preregistration hash `27889afd834b0d32`; fit dev / select tune / report tune; seed 20261108; threads 2
* Hypothesis: (Exploratory.) Explicit individual adaptation (F3 episodic memory + learned gate, F2 individual policy posterior, F1 online latent) improves tune NLL over a same-size pop_gru twin and over the same population network with its recurrent state running.

## described (counted); best comparator on tune: `pop_gru`

| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pop_gru | comparator | 0.3550 [0.3139, 0.3990] | 41 | 0.840 | 0.028 | 0.388 / 0.361 / 0.356 / 0.315 | 8004 | 0.0 | hidden, dropout | 460.9 | ok |
| f2_world_policy[twin] | candidate | 0.3563 [0.3101, 0.3992] | 41 | 0.839 | 0.029 | 0.390 / 0.361 / 0.358 / 0.316 | 8004 | 0.0 | hidden, dropout, prior_scale | 1187.9 | ok |
| f1_latent | candidate | 0.3568 [0.3147, 0.3966] | 41 | 0.836 | 0.028 | 0.389 / 0.363 / 0.356 / 0.319 | 8724 | 0.0 | hidden, dropout | 137.4 | ok |
| f3_arbiter[frozen] | candidate | 0.3571 [0.3133, 0.3991] | 41 | 0.836 | 0.029 | 0.389 / 0.364 / 0.357 / 0.318 | 8010 | 0.0 | hidden, dropout, bandwidth | 418.4 | ok |
| f3_fixedmix[frozen] | reference | 0.3571 [0.3148, 0.3991] | 41 | 0.836 | 0.029 | 0.389 / 0.364 / 0.357 / 0.318 | 8005 | 0.0 | hidden, dropout | 27.2 | ok |
| f1_bolt[frozen] | reference | 0.3571 [0.3156, 0.3975] | 41 | 0.836 | 0.029 | 0.389 / 0.364 / 0.357 / 0.318 | 8724 | 0.0 | hidden, dropout | 0.1 | ok |
| f1_latent[frozen] | candidate | 0.3571 [0.3172, 0.4005] | 41 | 0.836 | 0.029 | 0.389 / 0.364 / 0.357 / 0.318 | 8724 | 0.0 | hidden, dropout | 137.4 | ok |
| f3_gruepi | reference | 0.3571 [0.3144, 0.3993] | 41 | 0.836 | 0.029 | 0.389 / 0.364 / 0.357 / 0.318 | 16488 | 0.0 | hidden, dropout | 187.1 | ok |
| f3_arbiter | candidate | 0.3571 [0.3139, 0.3974] | 41 | 0.836 | 0.029 | 0.389 / 0.364 / 0.357 / 0.318 | 8010 | 0.0 | hidden, dropout, bandwidth | 418.4 | ok |
| f3_arbiter[mismatched] | candidate | 0.3572 [0.3138, 0.3993] | 41 | 0.836 | 0.029 | 0.389 / 0.365 / 0.357 / 0.318 | 8010 | 0.0 | hidden, dropout, bandwidth | 418.4 | ok |
| f3_fixedmix | reference | 0.3572 [0.3148, 0.3988] | 41 | 0.836 | 0.029 | 0.389 / 0.365 / 0.357 / 0.318 | 8005 | 0.0 | hidden, dropout | 27.2 | ok |
| f1_bolt | reference | 0.3575 [0.3165, 0.3993] | 41 | 0.837 | 0.028 | 0.390 / 0.365 / 0.356 / 0.319 | 8724 | 0.0 | hidden, dropout | 0.1 | ok |
| f1_latent[mismatched] | candidate | 0.3580 [0.3167, 0.4005] | 41 | 0.836 | 0.030 | 0.389 / 0.366 / 0.359 / 0.318 | 8724 | 0.0 | hidden, dropout | 137.4 | ok |
| f3_fixedshare[frozen] | reference | 0.3586 [0.3163, 0.4013] | 41 | 0.838 | 0.030 | 0.390 / 0.363 / 0.364 / 0.317 | 8004 | 0.0 | hidden, dropout | 0.1 | ok |
| f3_arbiter[twin] | candidate | 0.3599 [0.3162, 0.4019] | 41 | 0.837 | 0.029 | 0.394 / 0.366 / 0.361 / 0.318 | 8004 | 0.0 | hidden, dropout, bandwidth | 418.4 | ok |
| f1_latent[twin] | candidate | 0.3602 [0.3176, 0.3994] | 41 | 0.836 | 0.032 | 0.393 / 0.368 / 0.361 / 0.319 | 8004 | 0.0 | hidden, dropout | 137.4 | ok |
| f3_fixedshare | reference | 0.3674 [0.3243, 0.4085] | 41 | 0.836 | 0.041 | 0.397 / 0.373 / 0.371 / 0.330 | 8004 | 0.0 | hidden, dropout | 0.1 | ok |
| f2_delta | reference | 0.3806 [0.3381, 0.4230] | 41 | 0.830 | 0.030 | 0.411 / 0.388 / 0.378 / 0.345 | 21 | 0.0 | delta_alpha | 4.0 | ok |
| f2_world_policy | candidate | 0.3811 [0.3390, 0.4239] | 41 | 0.829 | 0.031 | 0.412 / 0.387 / 0.380 / 0.346 | 8156 | 0.0 | hidden, dropout, prior_scale | 1187.9 | ok |
| f2_world_policy[frozen] | candidate | 0.3953 [0.3558, 0.4363] | 41 | 0.824 | 0.055 | 0.421 / 0.403 / 0.397 / 0.360 | 8156 | 0.0 | hidden, dropout, prior_scale | 1187.9 | ok |
| f2_delta[frozen] | reference | 0.3965 [0.3563, 0.4372] | 41 | 0.825 | 0.059 | 0.422 / 0.405 / 0.399 / 0.361 | 21 | 0.0 | delta_alpha | 4.0 | ok |
| f2_world_policy[mismatched] | candidate | 0.4154 [0.3724, 0.4579] | 41 | 0.822 | 0.071 | 0.435 / 0.426 / 0.421 / 0.380 | 8156 | 0.0 | hidden, dropout, prior_scale | 1187.9 | ok |

| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | sd | n |
|---|---|---|---|---|---|---|
| f1_latent − pop_gru | vs_best_comparator | +0.0018 | [-0.0008, +0.0042] | 0.9245 | 0.0082 | 41 |
| f1_latent − f1_latent[twin] | vs_twin | -0.0034 | [-0.0055, -0.0016] | 0.0005 | 0.0065 | 41 |
| f1_latent − f1_latent[frozen] | vs_frozen | -0.0003 | [-0.0008, +0.0002] | 0.1645 | 0.0018 | 41 |
| f1_latent − f1_latent[mismatched] | vs_mismatched | -0.0012 | [-0.0021, -0.0003] | 0.0015 | 0.0030 | 41 |
| f2_world_policy − pop_gru | vs_best_comparator | +0.0261 | [+0.0218, +0.0304] | 1.0000 | 0.0144 | 41 |
| f2_world_policy − f2_world_policy[twin] | vs_twin | +0.0248 | [+0.0205, +0.0294] | 1.0000 | 0.0146 | 41 |
| f2_world_policy − f2_world_policy[frozen] | vs_frozen | -0.0142 | [-0.0191, -0.0095] | 0.0005 | 0.0160 | 41 |
| f2_world_policy − f2_world_policy[mismatched] | vs_mismatched | -0.0343 | [-0.0459, -0.0234] | 0.0005 | 0.0377 | 41 |
| f3_arbiter − pop_gru | vs_best_comparator | +0.0022 | [-0.0005, +0.0047] | 0.9425 | 0.0086 | 41 |
| f3_arbiter − f3_arbiter[twin] | vs_twin | -0.0027 | [-0.0046, -0.0008] | 0.0010 | 0.0060 | 41 |
| f3_arbiter − f3_arbiter[frozen] | vs_frozen | +0.0001 | [+0.0000, +0.0001] | 0.9935 | 0.0001 | 41 |
| f3_arbiter − f3_arbiter[mismatched] | vs_mismatched | -0.0001 | [-0.0001, -0.0000] | 0.0105 | 0.0002 | 41 |

## Candidate vs best comparator across counted families (Holm)

| candidate | family | estimate | 95% CI | Holm p | meets 0.02 margin |
|---|---|---|---|---|---|
| f1_latent | described | +0.0018 | [-0.0008, +0.0042] | 0.9245 | False |
| f2_world_policy | described | +0.0261 | [+0.0218, +0.0304] | 1.0000 | False |
| f3_arbiter | described | +0.0022 | [-0.0005, +0.0047] | 0.9425 | False |

_Exploratory: the report partition is also the selection partition. No gate is evaluated here._
