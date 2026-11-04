# p2_human_dev_pilot_hybrid — Phase II dev pilot B — Candidate A (hybrid_exemplar) vs strongest references on the three pristine families (exploratory; fit dev / select tune / report tune)

> **EXPLORATORY (dev/tune partitions; selection-optimistic).** Real human data (Psych-101, Apache-2.0). Held-out participants per partition; teacher-forced per-choice NLL. Not a validation of the simulator and not evidence about any individual's future.

* Preregistration hash: `26b5d2f941079eba`; partitions `reports/phase2/partitions/psych101.json` (test sha256 `7dc2e7798f57d3fa…`)
* Fit on **dev**, select on **tune**, report on **tune**; code `38d2de49c14d31cd91468149cb8d3b384420dbee`; wall 1615.1 s; CPU 1593.0 s; LLM calls 0
* Registered policy classes: age_script_only, developmental, hybrid_exemplar, memory_only, plain_rl, random, static_persona

## Hypothesis and falsification

(Exploratory, dev/tune only.) Does Candidate A approach or beat the population-fitted GRU and the per-participant references on the pristine families named as Gate D primary/secondary? Informs the freeze decision; proves nothing.

Falsification: Not applicable to a dev pilot. The confirmatory preregistration (Gate D) will require the candidate to beat the best reference by >= 0.02 nats/choice (paired 95% CI excluding 0) on the frozen test partition in >= 2 of 3 preregistered families and not be worse by > 0.02 in any.

## bandit2 (exploratory) — gershman2018deconstructing/exp1.csv, gershman2018deconstructing/exp2.csv; n = {'dev': 32, 'tune': 16}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.3894 [0.3304, 0.4635] | -0.3037 | 0.825 | 0.063 | 0.462 / 0.356 / 0.351 / 0.389 | 666 | `{'hidden': 12.0, 'lr': 0.01, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.02 | hidden, lr | 36.75 |
| ref_pvl_delta | ok | 0.5339 [0.4740, 0.5909] | -0.1593 | 0.699 | 0.105 | 0.553 / 0.539 / 0.504 / 0.539 | 4 | `{'A': 0.4, 'alpha': 0.9, 'c': 1.0, 'lam': 1.0}` | 0.0 | A, alpha, c, lam | 6.6 |
| developmental | ok | 0.5618 [0.5188, 0.6090] | -0.1314 | 0.729 | 0.084 | 0.604 / 0.557 / 0.509 / 0.576 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 1.0}` | 0.2 | exploration_base, lr_encoder, lr_head | 6.67 |
| hybrid_exemplar | ok | 0.5854 [0.5429, 0.6315] | -0.1078 | 0.699 | 0.084 | 0.641 / 0.547 / 0.528 / 0.625 | — | `{'exemplar_beta': 1.0, 'gate_lr': 0.2, 'kernel_width': 1.0, 'recency_decay': 0.9}` | 0.05 | exemplar_beta, gate_lr, kernel_width, recency_decay | 26.46 |
| ref_exemplar | ok | 0.6103 [0.5747, 0.6472] | -0.0828 | 0.662 | 0.099 | 0.641 / 0.611 / 0.557 / 0.632 | 16 | `{'beta': 6.0, 'c': 0.5, 'decay': 0.9, 'lr_att': 0.0}` | 0.1 | beta, c, decay | 9.09 |
| ref_bias | ok | 0.6807 [0.6584, 0.7026] | -0.0125 | 0.583 | 0.176 | 0.687 / 0.679 / 0.681 / 0.676 | 1 | `{}` | 0.0 | — | 0.63 |
| plain_rl | ok | 0.6823 [0.6612, 0.7018] | -0.0108 | 0.569 | 0.123 | 0.684 / 0.690 / 0.659 / 0.696 | — | `{'lr': 0.05}` | 0.3 | — | 2.82 |
| memory_only | ok | 0.7752 [0.6941, 0.8599] | +0.0820 | 0.599 | 0.197 | 0.787 / 0.786 / 0.672 / 0.855 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 8.68 |
| ref_bayes_rule | unavailable | — | — | — | — | — | — | ValueError: ref_bayes_rule needs full (corrective) feedback | — | — | 0.0 |

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
| hybrid_exemplar | 0.603 | 0.568 |
| ref_exemplar | 0.637 | 0.583 |
| ref_bias | 0.650 | 0.711 |
| plain_rl | 0.682 | 0.683 |
| memory_only | 0.760 | 0.790 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.538 | 0.552 | 0.600 |
| hybrid_exemplar | 0.573 | 0.570 | 0.616 |
| memory_only | 0.774 | 0.780 | 0.772 |
| plain_rl | 0.679 | 0.701 | 0.667 |
| ref_bias | 0.628 | 0.694 | 0.730 |
| ref_exemplar | 0.615 | 0.581 | 0.634 |
| ref_gru | 0.303 | 0.350 | 0.533 |
| ref_pvl_delta | 0.597 | 0.429 | 0.563 |

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

## described (exploratory) — plonsky2018when/exp1.csv; n = {'dev': 85, 'tune': 41}

Chance NLL = 0.6931. Best reference: `ref_gru`; best non-developmental: `ref_gru`.

| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_gru | ok | 0.3720 [0.3314, 0.4116] | -0.3212 | 0.836 | 0.041 | 0.401 / 0.380 / 0.375 / 0.332 | 666 | `{'hidden': 12.0, 'lr': 0.003, 'max_epochs': 10.0, 'max_steps_per_epoch': 60000.0, 'patience': 3.0, 'window': 64.0}` | 0.02 | hidden, lr | 245.25 |
| hybrid_exemplar | ok | 0.6414 [0.6255, 0.6588] | -0.0517 | 0.627 | 0.070 | 0.647 / 0.630 / 0.658 / 0.630 | — | `{'exemplar_beta': 1.0, 'gate_lr': 0.05, 'kernel_width': 0.5, 'recency_decay': 0.9}` | 0.3 | exemplar_beta, gate_lr, kernel_width, recency_decay | 203.64 |
| ref_pvl_delta | ok | 0.6505 [0.6363, 0.6655] | -0.0427 | 0.615 | 0.073 | 0.647 / 0.642 / 0.658 / 0.655 | 4 | `{'A': 0.1, 'alpha': 0.9, 'c': 3.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 50.36 |
| developmental | ok | 0.6552 [0.6406, 0.6701] | -0.0379 | 0.623 | 0.062 | 0.661 / 0.642 / 0.666 / 0.653 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 0.35}` | 0.3 | exploration_base, lr_encoder | 44.32 |
| plain_rl | ok | 0.6636 [0.6493, 0.6790] | -0.0295 | 0.611 | 0.070 | 0.656 / 0.661 / 0.674 / 0.664 | — | `{'lr': 0.05}` | 0.3 | — | 20.35 |
| ref_bayes_rule | ok | 0.6690 [0.6563, 0.6825] | -0.0242 | 0.601 | 0.077 | 0.641 / 0.675 / 0.667 / 0.692 | 15 | `{'sigma': 0.5, 'tau2': 3.0}` | 0.3 | sigma, tau2 | 60.78 |
| ref_bias | ok | 0.6781 [0.6728, 0.6834] | -0.0150 | 0.587 | 0.046 | 0.676 / 0.677 / 0.678 / 0.682 | 1 | `{}` | 0.0 | — | 3.89 |
| ref_exemplar | ok | 0.6799 [0.6766, 0.6834] | -0.0132 | 0.598 | 0.078 | 0.680 / 0.677 / 0.683 / 0.680 | 16 | `{'beta': 6.0, 'c': 2.0, 'decay': 0.98, 'lr_att': 0.05}` | 0.0 | beta, c, decay, lr_att | 332.89 |
| memory_only | ok | 0.7757 [0.7447, 0.8053] | +0.0826 | 0.613 | 0.200 | 0.749 / 0.792 / 0.796 / 0.766 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 89.09 |

| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |
|---|---|---|---|---|---|
| developmental_vs_best_reference (developmental − ref_gru) | +0.2832 | [+0.2367, +0.3285] | 0.0005 | 1.86 | 41 |
| developmental_vs_best_non_developmental (developmental − ref_gru) | +0.2832 | [+0.2365, +0.3310] | 0.0005 | 1.86 | 41 |
| best_reference_vs_ref_bias (ref_gru − ref_bias) | -0.3061 | [-0.3467, -0.2629] | 0.0005 | -2.16 | 41 |

Robustness — report NLL by experiment file:

| model | plonsky2018when/exp1.csv |
|---|---|
| ref_gru | 0.372 |
| hybrid_exemplar | 0.641 |
| ref_pvl_delta | 0.650 |
| developmental | 0.655 |
| plain_rl | 0.664 |
| ref_bayes_rule | 0.669 |
| ref_bias | 0.678 |
| ref_exemplar | 0.680 |
| memory_only | 0.776 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.629 | 0.657 | 0.680 |
| hybrid_exemplar | 0.613 | 0.647 | 0.665 |
| memory_only | 0.720 | 0.810 | 0.800 |
| plain_rl | 0.640 | 0.667 | 0.684 |
| ref_bayes_rule | 0.652 | 0.665 | 0.689 |
| ref_bias | 0.660 | 0.678 | 0.696 |
| ref_exemplar | 0.675 | 0.683 | 0.682 |
| ref_gru | 0.359 | 0.386 | 0.373 |
| ref_pvl_delta | 0.647 | 0.648 | 0.656 |

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
| ref_gru | ok | 0.4467 [0.4109, 0.4824] | -0.2464 | 0.771 | 0.065 | 0.546 / 0.505 / 0.397 / 0.338 | 666 | `{'hidden': 12.0, 'lr': 0.003, 'max_epochs': 15.0, 'patience': 4.0, 'window': 64.0}` | 0.02 | hidden, lr | 148.12 |
| ref_bayes_rule | ok | 0.6120 [0.5869, 0.6397] | -0.0811 | 0.650 | 0.115 | 0.587 / 0.559 / 0.610 / 0.692 | 15 | `{'sigma': 0.5, 'tau2': 0.3}` | 0.1 | sigma, tau2 | 31.2 |
| ref_pvl_delta | ok | 0.6175 [0.5916, 0.6435] | -0.0757 | 0.637 | 0.086 | 0.603 / 0.563 / 0.611 / 0.693 | 4 | `{'A': 0.4, 'alpha': 0.5, 'c': 1.0, 'lam': 1.0}` | 0.3 | A, alpha, c, lam | 29.58 |
| hybrid_exemplar | ok | 0.6247 [0.5929, 0.6577] | -0.0685 | 0.644 | 0.142 | 0.594 / 0.561 / 0.625 / 0.719 | — | `{'exemplar_beta': 1.0, 'gate_lr': 0.05, 'kernel_width': 0.5, 'recency_decay': 0.99}` | 0.3 | exemplar_beta, gate_lr, kernel_width, recency_decay | 82.05 |
| plain_rl | ok | 0.6258 [0.6049, 0.6484] | -0.0673 | 0.644 | 0.107 | 0.600 / 0.576 / 0.622 / 0.705 | — | `{'lr': 0.01}` | 0.3 | lr | 12.02 |
| developmental | ok | 0.6405 [0.6124, 0.6687] | -0.0527 | 0.646 | 0.135 | 0.588 / 0.564 / 0.645 / 0.765 | — | `{'exploration_base': 0.3, 'lr_encoder': 0.06, 'lr_head': 0.1}` | 0.3 | exploration_base, lr_encoder, lr_head | 22.37 |
| ref_exemplar | ok | 0.6427 [0.6257, 0.6618] | -0.0505 | 0.645 | 0.115 | 0.587 / 0.579 / 0.658 / 0.747 | 16 | `{'beta': 2.0, 'c': 2.0, 'decay': 0.98, 'lr_att': 0.0}` | 0.3 | beta, c, decay, lr_att | 69.95 |
| memory_only | ok | 0.6511 [0.6133, 0.6890] | -0.0420 | 0.645 | 0.138 | 0.641 / 0.618 / 0.637 / 0.708 | — | `{'beta': 6.0, 'memory_capacity': 64.0}` | 0.3 | beta, memory_capacity | 34.9 |
| ref_bias | ok | 0.6515 [0.6365, 0.6673] | -0.0416 | 0.643 | 0.088 | 0.616 / 0.600 / 0.658 / 0.732 | 1 | `{}` | 0.0 | — | 2.48 |

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
| hybrid_exemplar | 0.671 | 0.662 | 0.602 | 0.564 |
| plain_rl | 0.661 | 0.654 | 0.608 | 0.581 |
| developmental | 0.676 | 0.677 | 0.622 | 0.586 |
| ref_exemplar | 0.669 | 0.660 | 0.642 | 0.600 |
| memory_only | 0.707 | 0.691 | 0.601 | 0.605 |
| ref_bias | 0.673 | 0.662 | 0.642 | 0.629 |

Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):

| model | most predictable | middle | least predictable |
|---|---|---|---|
| developmental | 0.515 | 0.639 | 0.767 |
| hybrid_exemplar | 0.490 | 0.623 | 0.760 |
| memory_only | 0.479 | 0.658 | 0.817 |
| plain_rl | 0.537 | 0.625 | 0.716 |
| ref_bayes_rule | 0.504 | 0.611 | 0.721 |
| ref_bias | 0.580 | 0.653 | 0.722 |
| ref_exemplar | 0.562 | 0.638 | 0.728 |
| ref_gru | 0.353 | 0.447 | 0.540 |
| ref_pvl_delta | 0.523 | 0.601 | 0.728 |

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

## Validity notes

* Report partition = tune, on which hyper-parameters were also selected; numbers are selection-optimistic and exploratory.
* Every Psych-101 MCPL/IGT participant was fitted on or scored in Phase I; the other seven families are pristine.
* Registry policy classes and reference models get grids of equal size caps (<= 24) and the same lapse grid; fits at grid edges are flagged.
* Encodings (12 context features, reward normalisation to [0, 1]) are modelling choices shared by all models; block resets are visible only via obs.meta, which registry classes do not read.
* Teacher-forced one-step-ahead likelihood measures predictive fit to choice sequences, not generative realism.

Regenerate from logs: `python -m hpe.evaluation.human analyze2 reports/phase2/human/p2_human_dev_pilot_hybrid`
