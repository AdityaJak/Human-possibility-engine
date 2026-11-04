# p2_human_dev_pilot — Phase II dev pilot — all registered policy classes and stronger references on seven Psych-101 learning families

> **EXPLORATORY (dev/tune partitions; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants per partition; teacher-forced per-choice NLL. Not a validation of the simulator and not evidence about any individual's future.

* Preregistration hash: `ee15cd6f2150f602`; partitions `reports/phase2/partitions/psych101.json` (test sha256 `7dc2e7798f57d3fa…`)
* Fit on **dev**, select on **tune**, report on **tune**; code `a7f9558727fa14f8b10d01755301d65b8f7753de+dirty`; wall 3796.1 s; CPU 3743.7 s; LLM calls 0
* Registered policy classes: developmental, memory_only, plain_rl, random, static_persona

## Hypothesis and falsification

(Exploratory, dev/tune only.) Establishes the equal-tuning protocol and the strongest reference models for Gate D. No confirmatory claim follows from this run.

Falsification: Not applicable to a dev pilot. The confirmatory preregistration (Gate D) will require the candidate to beat the best reference by >= 0.02 nats/choice (paired 95% CI excluding 0) on the frozen test partition in >= 2 of 3 preregistered families and not be worse by > 0.02 in any.

## bandit2 (exploratory) — gershman2018deconstructing/exp1.csv, gershman2018deconstructing/exp2.csv; n = {'dev': 32, 'tune': 16}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.3894 [0.3304, 0.4635] | -0.3037 | 0.825 | 0.063 | 0.462 / 0.356 / 0.351 / 0.389 | 666 | `{'hidden': 12.0, 'lr': 0.01, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.02 | hidden, lr | 35.04 |
| ref_pvl_delta | ok | 0.5339 [0.4740, 0.5909] | -0.1593 | 0.699 | 0.105 | 0.553 / 0.539 / 0.504 / 0.539 | 4 | `{'A': 0.4, 'alpha': 0.9, 'c': 1.0, 'lam': 1.0}` | 0.0 | A, alpha, c, lam | 6.4 |
| developmental | ok | 0.5618 [0.5188, 0.6090] | -0.1314 | 0.729 | 0.084 | 0.604 / 0.557 / 0.509 / 0.576 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 1.0}` | 0.2 | exploration_base, lr_encoder, lr_head | 7.41 |
| ref_exemplar | ok | 0.6103 [0.5747, 0.6472] | -0.0828 | 0.662 | 0.099 | 0.641 / 0.611 / 0.557 / 0.632 | 16 | `{'beta': 6.0, 'c': 0.5, 'decay': 0.9, 'lr_att': 0.0}` | 0.1 | beta, c, decay | 8.82 |
| ref_bias | ok | 0.6807 [0.6584, 0.7026] | -0.0125 | 0.583 | 0.176 | 0.687 / 0.679 / 0.681 / 0.676 | 1 | `{}` | 0.0 | — | 0.66 |
| plain_rl | ok | 0.6823 [0.6612, 0.7018] | -0.0108 | 0.569 | 0.123 | 0.684 / 0.690 / 0.659 / 0.696 | — | `{'lr': 0.05}` | 0.3 | — | 2.83 |
| random | ok | 0.6931 [0.6931, 0.6931] | -0.0000 | 0.500 | 0.000 | 0.693 / 0.693 / 0.693 / 0.693 | — | `{}` | 0.0 | — | 0.81 |
| memory_only | ok | 0.7752 [0.6941, 0.8599] | +0.0820 | 0.599 | 0.197 | 0.787 / 0.786 / 0.672 / 0.855 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 8.47 |
| static_persona | ok | 0.8236 [0.7115, 0.9716] | +0.1305 | 0.542 | 0.236 | 0.855 / 0.864 / 0.761 / 0.815 | — | `{}` | 0.3 | — | 0.85 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_bayes_rule | unavailable | — | — | — | — | — | — | ValueError: ref_bayes_rule needs full (corrective) feedback | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.1723 | [+0.1152, +0.2263] | 0.0005 | 1.57 | 16 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.1723 | [+0.1178, +0.2228] | 0.0005 | 1.57 | 16 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.2912 | [-0.3447, -0.2253] | 0.0005 | -2.27 | 16 |

Robustness — report NLL by experiment file:

| model | gershman2018deconstructing/exp1.csv | gershman2018deconstructing/exp2.csv |
|---|---|---|
| ref_gru | 0.407 | 0.372 |
| ref_pvl_delta | 0.633 | 0.435 |
| developmental | 0.577 | 0.547 |
| ref_exemplar | 0.637 | 0.583 |
| ref_bias | 0.650 | 0.711 |
| plain_rl | 0.682 | 0.683 |
| random | 0.693 | 0.693 |
| memory_only | 0.760 | 0.790 |
| static_persona | 0.828 | 0.819 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.538 | 0.552 | 0.600 |
| memory_only | 0.774 | 0.780 | 0.772 |
| plain_rl | 0.679 | 0.701 | 0.667 |
| random | 0.693 | 0.693 | 0.693 |
| ref_bias | 0.628 | 0.694 | 0.730 |
| ref_exemplar | 0.615 | 0.581 | 0.634 |
| ref_gru | 0.303 | 0.350 | 0.533 |
| ref_pvl_delta | 0.597 | 0.429 | 0.563 |
| static_persona | 0.847 | 0.838 | 0.781 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 32,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 16,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## casino (exploratory) — lefebvre2017behavioural/exp1.csv, lefebvre2017behavioural/exp2.csv; n = {'dev': 31, 'tune': 16}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.5562 [0.4849, 0.6308] | -0.1370 | 0.702 | 0.111 | 0.696 / 0.550 / 0.502 / 0.476 | 666 | `{'hidden': 8.0, 'lr': 0.01, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.1 | hidden, lr | 22.2 |
| developmental | ok | 0.6226 [0.5979, 0.6513] | -0.0705 | 0.651 | 0.124 | 0.670 / 0.634 / 0.597 / 0.590 | — | `{'exploration_base': 0.1, 'lr_encoder': 0.06, 'lr_head': 0.1}` | 0.0 | exploration_base, lr_encoder, lr_head | 4.51 |
| plain_rl | ok | 0.6233 [0.5895, 0.6590] | -0.0698 | 0.691 | 0.139 | 0.657 / 0.625 / 0.606 / 0.606 | — | `{'lr': 0.05}` | 0.0 | — | 1.88 |
| memory_only | ok | 0.6369 [0.5397, 0.7401] | -0.0562 | 0.693 | 0.143 | 0.620 / 0.628 / 0.649 / 0.650 | — | `{'beta': 3.0, 'memory_capacity': 256.0}` | 0.3 | — | 5.3 |
| ref_exemplar | ok | 0.6548 [0.6278, 0.6896] | -0.0383 | 0.644 | 0.149 | 0.703 / 0.642 / 0.639 / 0.635 | 16 | `{'beta': 6.0, 'c': 2.0, 'decay': 0.98, 'lr_att': 0.0}` | 0.1 | beta, c, decay | 3.49 |
| ref_pvl_delta | ok | 0.6791 [0.6541, 0.7003] | -0.0141 | 0.584 | 0.096 | 0.678 / 0.674 / 0.680 / 0.685 | 4 | `{'A': 0.1, 'alpha': 0.5, 'c': 1.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 4.79 |
| random | ok | 0.6931 [0.6931, 0.6931] | +0.0000 | 0.500 | 0.000 | 0.693 / 0.693 / 0.693 / 0.693 | — | `{}` | 0.0 | — | 0.44 |
| ref_bias | ok | 0.6950 [0.6901, 0.7008] | +0.0019 | 0.480 | 0.125 | 0.694 / 0.696 / 0.695 / 0.695 | 1 | `{}` | 0.3 | — | 0.34 |
| static_persona | ok | 0.9507 [0.8095, 1.0867] | +0.2576 | 0.455 | 0.358 | 0.882 / 0.945 / 0.960 / 1.015 | — | `{}` | 0.3 | — | 0.68 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_bayes_rule | unavailable | — | — | — | — | — | — | ValueError: ref_bayes_rule needs full (corrective) feedback | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.0665 | [+0.0055, +0.1235] | 0.032 | 0.54 | 16 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.0665 | [+0.0060, +0.1250] | 0.03 | 0.54 | 16 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.1389 | [-0.2062, -0.0631] | 0.0005 | -0.91 | 16 |

Robustness — report NLL by experiment file:

| model | lefebvre2017behavioural/exp1.csv | lefebvre2017behavioural/exp2.csv |
|---|---|---|
| ref_gru | 0.575 | 0.532 |
| developmental | 0.614 | 0.634 |
| plain_rl | 0.628 | 0.617 |
| memory_only | 0.629 | 0.647 |
| ref_exemplar | 0.654 | 0.656 |
| ref_pvl_delta | 0.698 | 0.655 |
| random | 0.693 | 0.693 |
| ref_bias | 0.694 | 0.696 |
| static_persona | 0.792 | 1.155 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.612 | 0.626 | 0.632 |
| memory_only | 0.589 | 0.684 | 0.647 |
| plain_rl | 0.594 | 0.661 | 0.621 |
| random | 0.693 | 0.693 | 0.693 |
| ref_bias | 0.685 | 0.695 | 0.707 |
| ref_exemplar | 0.632 | 0.675 | 0.662 |
| ref_gru | 0.538 | 0.618 | 0.516 |
| ref_pvl_delta | 0.671 | 0.711 | 0.656 |
| static_persona | 1.040 | 0.695 | 1.099 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 31,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 16,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## described (exploratory) — plonsky2018when/exp1.csv; n = {'dev': 85, 'tune': 41}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.3720 [0.3314, 0.4116] | -0.3212 | 0.836 | 0.041 | 0.401 / 0.380 / 0.375 / 0.332 | 666 | `{'hidden': 12.0, 'lr': 0.003, 'max_epochs': 10.0, 'max_steps_per_epoch': 60000.0, 'patience': 3.0, 'window': 64.0}` | 0.02 | hidden, lr | 224.0 |
| ref_pvl_delta | ok | 0.6505 [0.6363, 0.6655] | -0.0427 | 0.615 | 0.073 | 0.647 / 0.642 / 0.658 / 0.655 | 4 | `{'A': 0.1, 'alpha': 0.9, 'c': 3.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 42.23 |
| developmental | ok | 0.6552 [0.6406, 0.6701] | -0.0379 | 0.623 | 0.062 | 0.661 / 0.642 / 0.666 / 0.653 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 0.35}` | 0.3 | exploration_base, lr_encoder | 50.1 |
| plain_rl | ok | 0.6636 [0.6493, 0.6790] | -0.0295 | 0.611 | 0.070 | 0.656 / 0.661 / 0.674 / 0.664 | — | `{'lr': 0.05}` | 0.3 | — | 22.13 |
| ref_bayes_rule | ok | 0.6690 [0.6563, 0.6825] | -0.0242 | 0.601 | 0.077 | 0.641 / 0.675 / 0.667 / 0.692 | 15 | `{'sigma': 0.5, 'tau2': 3.0}` | 0.3 | sigma, tau2 | 62.25 |
| ref_bias | ok | 0.6781 [0.6728, 0.6834] | -0.0150 | 0.587 | 0.046 | 0.676 / 0.677 / 0.678 / 0.682 | 1 | `{}` | 0.0 | — | 4.53 |
| ref_exemplar | ok | 0.6799 [0.6766, 0.6834] | -0.0132 | 0.598 | 0.078 | 0.680 / 0.677 / 0.683 / 0.680 | 16 | `{'beta': 6.0, 'c': 2.0, 'decay': 0.98, 'lr_att': 0.05}` | 0.0 | beta, c, decay, lr_att | 304.53 |
| random | ok | 0.6931 [0.6931, 0.6931] | +0.0000 | 0.500 | 0.000 | 0.693 / 0.693 / 0.693 / 0.693 | — | `{}` | 0.0 | — | 4.49 |
| memory_only | ok | 0.7757 [0.7447, 0.8053] | +0.0826 | 0.613 | 0.200 | 0.749 / 0.792 / 0.796 / 0.766 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 110.49 |
| static_persona | ok | 0.8841 [0.8377, 0.9294] | +0.1909 | 0.506 | 0.276 | 0.885 / 0.895 / 0.910 / 0.847 | — | `{}` | 0.3 | — | 7.03 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.2832 | [+0.2367, +0.3285] | 0.0005 | 1.86 | 41 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.2832 | [+0.2365, +0.3310] | 0.0005 | 1.86 | 41 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.3061 | [-0.3467, -0.2629] | 0.0005 | -2.16 | 41 |

Robustness — report NLL by experiment file:

| model | plonsky2018when/exp1.csv |
|---|---|
| ref_gru | 0.372 |
| ref_pvl_delta | 0.650 |
| developmental | 0.655 |
| plain_rl | 0.664 |
| ref_bayes_rule | 0.669 |
| ref_bias | 0.678 |
| ref_exemplar | 0.680 |
| random | 0.693 |
| memory_only | 0.776 |
| static_persona | 0.884 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.629 | 0.657 | 0.680 |
| memory_only | 0.720 | 0.810 | 0.800 |
| plain_rl | 0.640 | 0.667 | 0.684 |
| random | 0.693 | 0.693 | 0.693 |
| ref_bayes_rule | 0.652 | 0.665 | 0.689 |
| ref_bias | 0.660 | 0.678 | 0.696 |
| ref_exemplar | 0.675 | 0.683 | 0.682 |
| ref_gru | 0.359 | 0.386 | 0.373 |
| ref_pvl_delta | 0.647 | 0.648 | 0.656 |
| static_persona | 0.871 | 0.921 | 0.863 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 85,
  "excluded": [
   "plonsky2018when/exp1.csv#94: no scorable trials"
  ],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 41,
  "excluded": [
   "plonsky2018when/exp1.csv#86: no scorable trials",
   "plonsky2018when/exp1.csv#89: no scorable trials"
  ],
  "ambiguous_choices_skipped": 0
 }
}
```

## garcia (exploratory) — garcia2023experiential/exp1.csv, garcia2023experiential/exp2.csv, garcia2023experiential/exp3.csv, garcia2023experiential/exp4.csv; n = {'dev': 138, 'tune': 68}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.4467 [0.4109, 0.4824] | -0.2464 | 0.771 | 0.065 | 0.546 / 0.505 / 0.397 / 0.338 | 666 | `{'hidden': 12.0, 'lr': 0.003, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.02 | hidden, lr | 178.57 |
| ref_bayes_rule | ok | 0.6120 [0.5869, 0.6397] | -0.0811 | 0.650 | 0.115 | 0.587 / 0.559 / 0.610 / 0.692 | 15 | `{'sigma': 0.5, 'tau2': 0.3}` | 0.1 | sigma, tau2 | 38.6 |
| ref_pvl_delta | ok | 0.6175 [0.5916, 0.6435] | -0.0757 | 0.637 | 0.086 | 0.603 / 0.563 / 0.611 / 0.693 | 4 | `{'A': 0.4, 'alpha': 0.5, 'c': 1.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 32.25 |
| plain_rl | ok | 0.6258 [0.6049, 0.6484] | -0.0673 | 0.644 | 0.107 | 0.600 / 0.576 / 0.622 / 0.705 | — | `{'lr': 0.01}` | 0.3 | lr | 10.08 |
| developmental | ok | 0.6405 [0.6124, 0.6687] | -0.0527 | 0.646 | 0.135 | 0.588 / 0.564 / 0.645 / 0.765 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 0.1}` | 0.3 | exploration_base, lr_encoder, lr_head | 24.64 |
| ref_exemplar | ok | 0.6427 [0.6257, 0.6618] | -0.0505 | 0.645 | 0.115 | 0.587 / 0.579 / 0.658 / 0.747 | 16 | `{'beta': 2.0, 'c': 2.0, 'decay': 0.98, 'lr_att': 0.0}` | 0.3 | beta, c, decay, lr_att | 73.31 |
| memory_only | ok | 0.6511 [0.6133, 0.6890] | -0.0420 | 0.645 | 0.138 | 0.641 / 0.618 / 0.637 / 0.708 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 36.44 |
| ref_bias | ok | 0.6515 [0.6365, 0.6673] | -0.0416 | 0.643 | 0.088 | 0.616 / 0.600 / 0.658 / 0.732 | 1 | `{}` | 0.0 | — | 2.67 |
| random | ok | 0.6931 [0.6931, 0.6931] | +0.0000 | 0.500 | 0.000 | 0.693 / 0.693 / 0.693 / 0.693 | — | `{}` | 0.0 | — | 3.16 |
| static_persona | ok | 0.9017 [0.8449, 0.9565] | +0.2085 | 0.516 | 0.296 | 0.886 / 0.867 / 0.925 / 0.929 | — | `{}` | 0.3 | — | 4.02 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.1937 | [+0.1666, +0.2192] | 0.0005 | 1.72 | 68 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.1937 | [+0.1665, +0.2192] | 0.0005 | 1.72 | 68 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.2048 | [-0.2333, -0.1770] | 0.0005 | -1.70 | 68 |

Robustness — report NLL by experiment file:

| model | garcia2023experiential/exp1.csv | garcia2023experiential/exp2.csv | garcia2023experiential/exp3.csv | garcia2023experiential/exp4.csv |
|---|---|---|---|---|
| ref_gru | 0.553 | 0.449 | 0.396 | 0.390 |
| ref_bayes_rule | 0.656 | 0.645 | 0.587 | 0.561 |
| ref_pvl_delta | 0.673 | 0.625 | 0.600 | 0.572 |
| plain_rl | 0.661 | 0.654 | 0.608 | 0.581 |
| developmental | 0.676 | 0.677 | 0.622 | 0.586 |
| ref_exemplar | 0.669 | 0.660 | 0.642 | 0.600 |
| memory_only | 0.707 | 0.691 | 0.601 | 0.605 |
| ref_bias | 0.673 | 0.662 | 0.642 | 0.629 |
| random | 0.693 | 0.693 | 0.693 | 0.693 |
| static_persona | 0.898 | 0.932 | 0.845 | 0.932 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.515 | 0.639 | 0.767 |
| memory_only | 0.479 | 0.658 | 0.817 |
| plain_rl | 0.537 | 0.625 | 0.716 |
| random | 0.693 | 0.693 | 0.693 |
| ref_bayes_rule | 0.504 | 0.611 | 0.721 |
| ref_bias | 0.580 | 0.653 | 0.722 |
| ref_exemplar | 0.562 | 0.638 | 0.728 |
| ref_gru | 0.353 | 0.447 | 0.540 |
| ref_pvl_delta | 0.523 | 0.601 | 0.728 |
| static_persona | 0.837 | 0.918 | 0.951 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 138,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 68,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## horizon (exploratory) — wilson2014humans/exp1.csv, wilson2014humans/exp2.csv, wilson2014humans/exp3.csv, wilson2014humans/exp4.csv, wilson2014humans/exp5.csv, waltz2020differential/exp1.csv; n = {'dev': 103, 'tune': 51}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.4589 [0.4152, 0.5005] | -0.2342 | 0.783 | 0.057 | 0.480 / 0.467 / 0.441 / 0.447 | 666 | `{'hidden': 12.0, 'lr': 0.003, 'max_epochs': 8.0, 'max_steps_per_epoch': 60000.0, 'patience': 3.0, 'window': 64.0}` | 0.02 | hidden, lr | 296.72 |
| ref_pvl_delta | ok | 0.5294 [0.4853, 0.5762] | -0.1638 | 0.765 | 0.103 | 0.548 / 0.531 / 0.512 / 0.526 | 4 | `{'A': 0.4, 'alpha': 0.9, 'c': 3.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 92.22 |
| developmental | ok | 0.6276 [0.6146, 0.6413] | -0.0655 | 0.652 | 0.056 | 0.662 / 0.618 / 0.618 / 0.612 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 1.0}` | 0.1 | exploration_base, lr_encoder, lr_head | 101.67 |
| ref_exemplar | ok | 0.6318 [0.6243, 0.6401] | -0.0613 | 0.680 | 0.114 | 0.636 / 0.632 / 0.629 / 0.630 | 16 | `{'beta': 6.0, 'c': 2.0, 'decay': 0.9, 'lr_att': 0.0}` | 0.0 | beta, c, decay | 447.1 |
| plain_rl | ok | 0.6879 [0.6855, 0.6903] | -0.0053 | 0.557 | 0.038 | 0.689 / 0.683 / 0.685 / 0.695 | — | `{'lr': 0.01}` | 0.3 | lr | 52.28 |
| ref_bias | ok | 0.6912 [0.6905, 0.6919] | -0.0020 | 0.540 | 0.042 | 0.692 / 0.689 / 0.690 / 0.693 | 1 | `{}` | 0.0 | — | 7.64 |
| random | ok | 0.6931 [0.6931, 0.6931] | +0.0000 | 0.500 | 0.000 | 0.693 / 0.693 / 0.693 / 0.693 | — | `{}` | 0.0 | — | 7.76 |
| memory_only | ok | 0.7851 [0.7685, 0.8025] | +0.0920 | 0.574 | 0.189 | 0.792 / 0.755 / 0.774 / 0.820 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 279.98 |
| static_persona | ok | 0.8531 [0.8117, 0.8946] | +0.1600 | 0.504 | 0.222 | 0.834 / 0.866 / 0.868 / 0.845 | — | `{}` | 0.3 | — | 9.9 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_bayes_rule | unavailable | — | — | — | — | — | — | ValueError: ref_bayes_rule needs full (corrective) feedback | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.1687 | [+0.1338, +0.2054] | 0.0005 | 1.29 | 51 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.1687 | [+0.1334, +0.2007] | 0.0005 | 1.29 | 51 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.2323 | [-0.2758, -0.1887] | 0.0005 | -1.46 | 51 |

Robustness — report NLL by experiment file:

| model | wilson2014humans/exp1.csv | wilson2014humans/exp2.csv | wilson2014humans/exp3.csv | wilson2014humans/exp4.csv | wilson2014humans/exp5.csv | waltz2020differential/exp1.csv |
|---|---|---|---|---|---|---|
| ref_gru | 0.410 | 0.348 | 0.459 | 0.584 | 0.421 | 0.267 |
| ref_pvl_delta | 0.466 | 0.387 | 0.513 | 0.652 | 0.477 | 0.406 |
| developmental | 0.614 | 0.586 | 0.626 | 0.652 | 0.593 | 0.631 |
| ref_exemplar | 0.616 | 0.620 | 0.640 | 0.646 | 0.609 | 0.621 |
| plain_rl | 0.695 | 0.695 | 0.691 | 0.687 | 0.679 | 0.689 |
| ref_bias | 0.695 | 0.694 | 0.692 | 0.690 | 0.690 | 0.689 |
| random | 0.693 | 0.693 | 0.693 | 0.693 | 0.693 | 0.693 |
| memory_only | 0.819 | 0.853 | 0.800 | 0.779 | 0.727 | 0.800 |
| static_persona | 0.851 | 0.762 | 0.857 | 0.817 | 0.926 | 0.861 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.603 | 0.656 | 0.624 |
| memory_only | 0.748 | 0.790 | 0.818 |
| plain_rl | 0.679 | 0.690 | 0.695 |
| random | 0.693 | 0.693 | 0.693 |
| ref_bias | 0.688 | 0.691 | 0.694 |
| ref_exemplar | 0.615 | 0.647 | 0.633 |
| ref_gru | 0.372 | 0.558 | 0.447 |
| ref_pvl_delta | 0.454 | 0.634 | 0.500 |
| static_persona | 0.891 | 0.827 | 0.841 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 103,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 51,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## igt (secondary) — steingroever2015data/exp1.csv, steingroever2015data/exp2.csv, steingroever2015data/exp3.csv; n = {'dev': 204, 'tune': 103}

Chance NLL = 1.3863. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 1.1001 [1.0372, 1.1636] | -0.2862 | 0.520 | 0.099 | 1.197 / 1.119 / 1.086 / 0.998 | 780 | `{'hidden': 12.0, 'igt_reward_scale': 250.0, 'lr': 0.003, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.0 | hidden, lr | 140.68 |
| developmental | ok | 1.1831 [1.1280, 1.2377] | -0.2032 | 0.455 | 0.111 | 1.263 / 1.209 / 1.166 / 1.093 | — | `{'exploration_base': 0.3, 'igt_reward_scale': 250.0, 'lr_encoder': 0.06, 'lr_head': 0.35}` | 0.0 | exploration_base, lr_encoder | 60.62 |
| memory_only | ok | 1.2840 [1.2410, 1.3215] | -0.1023 | 0.431 | 0.151 | 1.297 / 1.287 / 1.296 / 1.256 | — | `{'beta': 6.0, 'igt_reward_scale': 500.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 59.8 |
| ref_instance_based | ok | 1.2890 [1.2572, 1.3157] | -0.0973 | 0.396 | 0.137 | 1.320 / 1.314 / 1.301 / 1.221 | 3 | `{'beta': 6.0, 'decay': 0.5, 'igt_reward_scale': 500.0}` | 0.2 | decay | 49.52 |
| ref_pvl_delta | ok | 1.2973 [1.2672, 1.3245] | -0.0890 | 0.400 | 0.143 | 1.285 / 1.286 / 1.311 / 1.308 | 4 | `{'A': 0.1, 'alpha': 0.5, 'c': 2.0, 'igt_reward_scale': 500.0, 'lam': 1.0}` | 0.3 | A, alpha, lam | 28.05 |
| ref_exemplar | ok | 1.3110 [1.2979, 1.3238] | -0.0753 | 0.461 | 0.177 | 1.311 / 1.317 / 1.317 / 1.299 | 16 | `{'beta': 6.0, 'c': 0.5, 'decay': 0.9, 'igt_reward_scale': 250.0, 'lr_att': 0.0}` | 0.0 | beta, c, decay | 50.47 |
| ref_delta_rule | ok | 1.3142 [1.2883, 1.3369] | -0.0721 | 0.381 | 0.149 | 1.298 / 1.313 / 1.332 / 1.314 | 2 | `{'alpha': 0.4, 'beta': 6.0, 'igt_reward_scale': 250.0}` | 0.2 | alpha | 45.69 |
| plain_rl | ok | 1.3369 [1.3204, 1.3516] | -0.0494 | 0.395 | 0.141 | 1.350 / 1.335 / 1.331 / 1.331 | — | `{'igt_reward_scale': 500.0, 'lr': 0.05}` | 0.0 | — | 23.84 |
| ref_bias | ok | 1.3595 [1.3497, 1.3694] | -0.0268 | 0.306 | 0.119 | 1.376 / 1.358 / 1.356 / 1.348 | 3 | `{'igt_reward_scale': 250.0}` | 0.2 | — | 4.97 |
| random | ok | 1.3863 [1.3863, 1.3863] | +0.0000 | 0.250 | 0.000 | 1.386 / 1.386 / 1.386 / 1.386 | — | `{'igt_reward_scale': 250.0}` | 0.0 | — | 4.6 |
| static_persona | ok | 1.8336 [1.7629, 1.8976] | +0.4473 | 0.248 | 0.413 | 1.821 / 1.830 / 1.837 / 1.847 | — | `{'igt_reward_scale': 250.0}` | 0.3 | — | 5.86 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_bayes_rule | unavailable | — | — | — | — | — | — | ValueError: ref_bayes_rule needs full (corrective) feedback | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.0830 | [+0.0644, +0.1021] | 0.0005 | 0.84 | 103 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.0830 | [+0.0645, +0.1023] | 0.0005 | 0.84 | 103 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.2594 | [-0.3196, -0.1993] | 0.0005 | -0.83 | 103 |

Robustness — report NLL by experiment file:

| model | steingroever2015data/exp1.csv | steingroever2015data/exp2.csv | steingroever2015data/exp3.csv |
|---|---|---|---|
| ref_gru | 1.113 | 1.255 | 1.015 |
| developmental | 1.201 | 1.275 | 1.086 |
| memory_only | 1.300 | 1.270 | 1.214 |
| ref_instance_based | 1.297 | 1.353 | 1.244 |
| ref_pvl_delta | 1.311 | 1.286 | 1.237 |
| ref_exemplar | 1.313 | 1.327 | 1.301 |
| ref_delta_rule | 1.322 | 1.328 | 1.277 |
| plain_rl | 1.345 | 1.339 | 1.299 |
| ref_bias | 1.361 | 1.321 | 1.360 |
| random | 1.386 | 1.386 | 1.386 |
| static_persona | 1.803 | 2.179 | 1.915 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 1.069 | 1.235 | 1.249 |
| memory_only | 1.189 | 1.318 | 1.347 |
| plain_rl | 1.299 | 1.340 | 1.374 |
| random | 1.386 | 1.386 | 1.386 |
| ref_bias | 1.302 | 1.365 | 1.413 |
| ref_delta_rule | 1.254 | 1.312 | 1.379 |
| ref_exemplar | 1.275 | 1.313 | 1.345 |
| ref_gru | 0.992 | 1.155 | 1.156 |
| ref_instance_based | 1.243 | 1.300 | 1.325 |
| ref_pvl_delta | 1.194 | 1.319 | 1.382 |
| static_persona | 1.883 | 1.820 | 1.796 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 204,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 103,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## mcpl (primary) — collsiöö2023MCPL/exp1.csv, collsiöö2023MCPL/exp2.csv, collsiöö2023MCPL/exp3.csv; n = {'dev': 93, 'tune': 47}

Chance NLL = 2.1972. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 1.2481 [1.0960, 1.3970] | -0.9491 | 0.599 | 0.124 | 1.686 / 1.279 / 1.067 / 0.955 | 1065 | `{'hidden': 12.0, 'lr': 0.003, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.02 | hidden, lr | 128.86 |
| memory_only | ok | 1.4259 [1.2684, 1.5874] | -0.7713 | 0.597 | 0.225 | 1.899 / 1.490 / 1.220 / 1.090 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.1 | beta, memory_capacity | 32.8 |
| ref_instance_based | ok | 1.4414 [1.3175, 1.5731] | -0.7559 | 0.598 | 0.239 | 1.852 / 1.469 / 1.254 / 1.187 | 3 | `{'h': 0.5, 'kappa': 0.05, 's': 0.5}` | 0.1 | h, s | 13.1 |
| ref_population_static | ok | 1.5684 [1.4691, 1.6835] | -0.6288 | 0.398 | 0.147 | 1.876 / 1.605 / 1.422 / 1.367 | 200 | `{'smoothing': 1.0}` | 0.02 | — | 3.89 |
| developmental | ok | 1.6927 [1.6060, 1.7870] | -0.5045 | 0.393 | 0.146 | 2.075 / 1.777 / 1.517 / 1.399 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 0.35}` | 0.2 | exploration_base, lr_encoder | 20.4 |
| ref_bayes_rule | ok | 1.7306 [1.5536, 1.9049] | -0.4666 | 0.488 | 0.253 | 1.955 / 1.741 / 1.623 / 1.601 | 15 | `{'sigma': 0.5, 'tau2': 0.3}` | 0.3 | sigma, tau2 | 26.37 |
| ref_delta_rule | ok | 1.7358 [1.5864, 1.8832] | -0.4614 | 0.490 | 0.223 | 1.932 / 1.743 / 1.644 / 1.623 | 2 | `{'alpha': 0.03, 'sigma': 0.75}` | 0.3 | sigma | 27.57 |
| plain_rl | ok | 1.7609 [1.7044, 1.8269] | -0.4363 | 0.321 | 0.111 | 2.044 / 1.837 / 1.614 / 1.544 | — | `{'lr': 0.05}` | 0.02 | — | 9.08 |
| ref_exemplar | ok | 1.8369 [1.7771, 1.8987] | -0.3603 | 0.388 | 0.168 | 2.141 / 1.848 / 1.692 / 1.662 | 16 | `{'beta': 6.0, 'c': 2.0, 'decay': 1.0, 'lr_att': 0.05}` | 0.05 | beta, c, decay, lr_att | 73.93 |
| ref_bias | ok | 1.9506 [1.9167, 1.9869] | -0.2466 | 0.288 | 0.077 | 1.987 / 1.953 / 1.907 / 1.955 | 8 | `{'smoothing': 1.0}` | 0.02 | — | 1.73 |
| ref_pvl_delta | ok | 2.1305 [2.0977, 2.1628] | -0.0668 | 0.222 | 0.068 | 2.209 / 2.158 / 2.081 / 2.073 | 4 | `{'A': 0.1, 'alpha': 0.5, 'c': 1.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 19.96 |
| random | ok | 2.1972 [2.1972, 2.1972] | -0.0000 | 0.111 | 0.000 | 2.197 / 2.197 / 2.197 / 2.197 | — | `{}` | 0.0 | — | 1.96 |
| static_persona | ok | 2.8624 [2.8193, 2.9059] | +0.6652 | 0.105 | 0.465 | 2.837 / 2.856 / 2.893 / 2.864 | — | `{}` | 0.3 | — | 2.75 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.4446 | [+0.3550, +0.5361] | 0.0005 | 1.44 | 47 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.4446 | [+0.3618, +0.5307] | 0.0005 | 1.44 | 47 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.7025 | [-0.8523, -0.5463] | 0.0005 | -1.29 | 47 |

Robustness — report NLL by experiment file:

| model | collsiöö2023MCPL/exp1.csv | collsiöö2023MCPL/exp2.csv | collsiöö2023MCPL/exp3.csv |
|---|---|---|---|
| ref_gru | 1.101 | 1.021 | 1.552 |
| memory_only | 1.265 | 1.199 | 1.740 |
| ref_instance_based | 1.295 | 1.269 | 1.699 |
| ref_population_static | 1.457 | 1.390 | 1.804 |
| developmental | 1.601 | 1.581 | 1.857 |
| ref_bayes_rule | 1.468 | 1.403 | 2.208 |
| ref_delta_rule | 1.535 | 1.482 | 2.104 |
| plain_rl | 1.746 | 1.700 | 1.823 |
| ref_exemplar | 1.837 | 1.806 | 1.863 |
| ref_bias | 1.984 | 1.958 | 1.919 |
| ref_pvl_delta | 2.139 | 2.136 | 2.119 |
| random | 2.197 | 2.197 | 2.197 |
| static_persona | 2.883 | 2.840 | 2.865 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 1.777 | 1.580 | 1.714 |
| memory_only | 1.717 | 1.274 | 1.277 |
| plain_rl | 1.722 | 1.676 | 1.879 |
| random | 2.197 | 2.197 | 2.197 |
| ref_bayes_rule | 2.047 | 1.666 | 1.475 |
| ref_bias | 1.827 | 1.957 | 2.068 |
| ref_delta_rule | 1.944 | 1.690 | 1.571 |
| ref_exemplar | 1.772 | 1.765 | 1.969 |
| ref_gru | 1.488 | 1.133 | 1.116 |
| ref_instance_based | 1.615 | 1.348 | 1.355 |
| ref_population_static | 1.659 | 1.494 | 1.548 |
| ref_pvl_delta | 2.180 | 2.084 | 2.124 |
| static_persona | 2.815 | 2.889 | 2.885 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 93,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 47,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## srl3 (exploratory) — gershman2020reward/exp1.csv; n = {'dev': 30, 'tune': 15}

Chance NLL = 1.0986. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.7411 [0.6980, 0.7853] | -0.3575 | 0.716 | 0.056 | 0.754 / 0.748 / 0.726 / 0.736 | 723 | `{'hidden': 12.0, 'lr': 0.01, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.1 | hidden, lr | 104.51 |
| developmental | ok | 0.8544 [0.8217, 0.8893] | -0.2442 | 0.636 | 0.048 | 0.891 / 0.860 / 0.805 / 0.862 | — | `{'exploration_base': 0.1, 'lr_encoder': 0.06, 'lr_head': 0.35}` | 0.1 | exploration_base, lr_encoder | 13.75 |
| ref_exemplar | ok | 0.9480 [0.9357, 0.9613] | -0.1506 | 0.564 | 0.083 | 0.953 / 0.952 / 0.978 / 0.909 | 16 | `{'beta': 6.0, 'c': 2.0, 'decay': 0.9, 'lr_att': 0.0}` | 0.0 | beta, c, decay | 33.87 |
| memory_only | ok | 1.0017 [0.9675, 1.0391] | -0.0969 | 0.596 | 0.160 | 0.999 / 1.025 / 0.928 / 1.055 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 29.14 |
| ref_pvl_delta | ok | 1.0396 [1.0285, 1.0513] | -0.0590 | 0.406 | 0.039 | 1.041 / 1.053 / 1.062 / 1.002 | 4 | `{'A': 0.1, 'alpha': 0.5, 'c': 1.0, 'lam': 1.0}` | 0.0 | A, alpha, c, lam | 13.06 |
| plain_rl | ok | 1.0780 [1.0653, 1.0916] | -0.0206 | 0.437 | 0.100 | 1.073 / 1.083 / 1.069 / 1.087 | — | `{'lr': 0.01}` | 0.0 | lr | 7.02 |
| ref_bias | ok | 1.0986 [1.0978, 1.0993] | -0.0000 | 0.325 | 0.029 | 1.098 / 1.099 / 1.100 / 1.097 | 2 | `{}` | 0.3 | — | 0.98 |
| random | ok | 1.0986 [1.0986, 1.0986] | -0.0000 | 0.333 | 0.000 | 1.099 / 1.099 / 1.099 / 1.099 | — | `{}` | 0.0 | — | 1.19 |
| static_persona | ok | 1.4763 [1.3952, 1.5553] | +0.3777 | 0.327 | 0.385 | 1.474 / 1.500 / 1.464 / 1.467 | — | `{}` | 0.3 | — | 1.43 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_bayes_rule | unavailable | — | — | — | — | — | — | ValueError: ref_bayes_rule needs full (corrective) feedback | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.1133 | [+0.0838, +0.1404] | 0.0005 | 1.95 | 15 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.1133 | [+0.0828, +0.1399] | 0.0005 | 1.95 | 15 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.3575 | [-0.4004, -0.3124] | 0.0005 | -4.04 | 15 |

Robustness — report NLL by experiment file:

| model | gershman2020reward/exp1.csv |
|---|---|
| ref_gru | 0.741 |
| developmental | 0.854 |
| ref_exemplar | 0.948 |
| memory_only | 1.002 |
| ref_pvl_delta | 1.040 |
| plain_rl | 1.078 |
| ref_bias | 1.099 |
| random | 1.099 |
| static_persona | 1.476 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.866 | 0.815 | 0.882 |
| memory_only | 1.048 | 0.950 | 1.007 |
| plain_rl | 1.065 | 1.075 | 1.094 |
| random | 1.099 | 1.099 | 1.099 |
| ref_bias | 1.097 | 1.099 | 1.100 |
| ref_exemplar | 0.952 | 0.931 | 0.961 |
| ref_gru | 0.755 | 0.690 | 0.778 |
| ref_pvl_delta | 1.040 | 1.036 | 1.043 |
| static_persona | 1.351 | 1.569 | 1.509 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 30,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 15,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## wpt (exploratory) — speekenbrink2008learning/exp1.csv; n = {'dev': 9, 'tune': 5}

Chance NLL = 0.6931. Best reference: `ref_bayes_rule`; best non-developmental: `plain_rl`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| developmental | ok | 0.5659 [0.4717, 0.6995] | -0.1273 | 0.714 | 0.084 | 0.564 / 0.618 / 0.515 / 0.566 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 0.35}` | 0.2 | exploration_base, lr_encoder | 2.1 |
| plain_rl | ok | 0.5691 [0.4678, 0.6991] | -0.1241 | 0.723 | 0.107 | 0.567 / 0.642 / 0.512 / 0.555 | — | `{'lr': 0.2}` | 0.3 | — | 0.87 |
| ref_bayes_rule | ok | 0.5733 [0.4818, 0.6933] | -0.1199 | 0.720 | 0.128 | 0.553 / 0.666 / 0.523 / 0.551 | 15 | `{'sigma': 0.5, 'tau2': 3.0}` | 0.02 | sigma, tau2 | 2.43 |
| ref_exemplar | ok | 0.5754 [0.4889, 0.7049] | -0.1177 | 0.727 | 0.107 | 0.588 / 0.637 / 0.526 / 0.550 | 16 | `{'beta': 2.0, 'c': 2.0, 'decay': 0.98, 'lr_att': 0.05}` | 0.02 | beta, c, decay, lr_att | 6.78 |
| memory_only | ok | 0.5946 [0.4568, 0.7710] | -0.0986 | 0.719 | 0.095 | 0.569 / 0.715 / 0.538 / 0.557 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 3.09 |
| ref_gru | ok | 0.5995 [0.5230, 0.7100] | -0.0937 | 0.699 | 0.128 | 0.621 / 0.650 / 0.562 / 0.565 | 666 | `{'hidden': 8.0, 'lr': 0.01, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.0 | hidden, lr | 9.13 |
| ref_bias | ok | 0.6928 [0.6922, 0.6933] | -0.0004 | 0.523 | 0.036 | 0.694 / 0.692 / 0.692 / 0.693 | 1 | `{}` | 0.0 | — | 0.15 |
| random | ok | 0.6931 [0.6931, 0.6931] | -0.0000 | 0.500 | 0.000 | 0.693 / 0.693 / 0.693 / 0.693 | — | `{}` | 0.0 | — | 0.2 |
| ref_pvl_delta | ok | 0.7059 [0.6936, 0.7194] | +0.0128 | 0.519 | 0.075 | 0.689 / 0.700 / 0.723 / 0.712 | 4 | `{'A': 0.1, 'alpha': 0.5, 'c': 1.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 2.08 |
| static_persona | ok | 1.0480 [0.8934, 1.2170] | +0.3549 | 0.404 | 0.382 | 1.076 / 1.038 / 1.053 / 1.025 | — | `{}` | 0.3 | — | 0.23 |
| hybrid_exemplar | unavailable | — | — | — | — | — | — | policy class 'hybrid_exemplar' not registered in this build (KeyError: "no polic | — | — | 0.0 |
| ref_delta_rule | unavailable | — | — | — | — | — | — | ValueError: ref_delta_rule (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_instance_based | unavailable | — | — | — | — | — | — | ValueError: ref_instance_based (Phase I) is defined for MCPL and IGT only | — | — | 0.0 |
| ref_population_static | unavailable | — | — | — | — | — | — | ValueError: ref_population_static is defined for MCPL only | — | — | 0.0 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_bayes_rule) | -0.0074 | [-0.0118, -0.0015] | 0.014 | -1.08 | 5 |
| developmental_vs_best_non_developmental (developmental − plain_rl) | -0.0032 | [-0.0235, +0.0113] | 0.768 | -0.14 | 5 |
| best_reference_vs_ref_bias (ref_bayes_rule − ref_bias) | -0.1195 | [-0.2104, +0.0007] | 0.056 | -0.86 | 5 |

Robustness — report NLL by experiment file:

| model | speekenbrink2008learning/exp1.csv |
|---|---|
| developmental | 0.566 |
| plain_rl | 0.569 |
| ref_bayes_rule | 0.573 |
| ref_exemplar | 0.575 |
| memory_only | 0.595 |
| ref_gru | 0.599 |
| ref_bias | 0.693 |
| random | 0.693 |
| ref_pvl_delta | 0.706 |
| static_persona | 1.048 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.647 | 0.444 | 0.546 |
| memory_only | 0.707 | 0.407 | 0.576 |
| plain_rl | 0.661 | 0.448 | 0.538 |
| random | 0.693 | 0.693 | 0.693 |
| ref_bayes_rule | 0.649 | 0.458 | 0.556 |
| ref_bias | 0.692 | 0.693 | 0.693 |
| ref_exemplar | 0.653 | 0.486 | 0.543 |
| ref_gru | 0.668 | 0.511 | 0.575 |
| ref_pvl_delta | 0.714 | 0.703 | 0.699 |
| static_persona | 1.086 | 1.353 | 0.858 |

Parse validation (partitions used): 

```json
{
 "dev": {
  "participants": 9,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 },
 "tune": {
  "participants": 5,
  "excluded": [],
  "ambiguous_choices_skipped": 0
 }
}
```

## Validity notes

* Report partition = tune, on which hyper-parameters were also selected; numbers are selection-optimistic and exploratory.
* Every Psych-101 MCPL/IGT participant was fitted on or scored in Phase I; the other seven families are pristine.
* Registry policy classes and reference models get grids of equal size caps (<= 24) and the same lapse grid; fits at grid edges are flagged.
* Encodings (12 context features, reward normalisation to [0, 1]) are modelling choices shared by all models; block resets are visible only via obs.meta, which registry classes do not read.
* Teacher-forced one-step-ahead likelihood measures predictive fit to choice sequences, not generative realism.

Regenerate from logs: `python -m hpe.evaluation.human analyze2 reports/phase2/human/p2_human_dev_pilot`
