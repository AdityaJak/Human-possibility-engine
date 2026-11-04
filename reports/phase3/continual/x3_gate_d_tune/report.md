# x3_gate_d_tune — Gate D leave-one-family-out transfer (history-only inputs), k in {0, 16, 64}, tune partition

> **EXPLORATORY (tune partition; selection-optimistic).** Real human data (Psych-101, Apache-2.0) via `load_family` (dev/tune only). History-only inputs. No gate is evaluated here.

* Lane research/continual-transfer; definition hash `9fd08848aa540405`; seed 20261109; threads 1
* Hypothesis: (Exploratory.) A GRU pretrained on other task families and refined by family-level first-order meta-learning (F4-style), then fine-tuned on k = 16 target participants, predicts held-out target participants better than a specialist trained on those 16 and than a pooled-pretrained pop_gru fine-tuned on them.
* Sources: bandit2, srl3, horizon, casino, wpt, garcia, described, restless4 (dev; the target excluded). Fine-tuning settings (shared): lr [0.003, 0.01] × epochs [5, 20, 60]; 5 draws per k
* hb_cog: pending: the evaluation lane's hb_cog is not merged into the director branch

## described (tune n = 41, dev n = 85; uniform-choice NLL 0.6931)

| k | model | tune NLL [95% CI] | selected setting | lapse | per-draw mean NLL | params | CPU s (model, target) | causal |
|---|---|---|---|---|---|---|---|---|
| 0 | pretrained_pop_gru | 0.4984 [0.4701, 0.5232] | zero-shot | 0.0 | — | 8004 | 734.9 | ok |
| 0 | meta_reptile | 0.4991 [0.4727, 0.5245] | zero-shot | 0.0 | — | 8004 | 694.4 | ok |
| 16 | pretrained_pop_gru | 0.3702 [0.3246, 0.4154] | lr=0.003,epochs=60 | 0.0 | 0.370, 0.372, 0.369, 0.371, 0.369 | 8004 | 734.9 | ok |
| 16 | meta_reptile | 0.3704 [0.3265, 0.4156] | lr=0.003,epochs=60 | 0.0 | 0.372, 0.372, 0.369, 0.369, 0.370 | 8004 | 694.4 | ok |
| 16 | specialist | 0.3739 [0.3325, 0.4184] | lr=0.01,epochs=60 | 0.0 | 0.369, 0.378, 0.370, 0.376, 0.375 | 8004 | 738.4 | ok |
| 64 | meta_reptile | 0.3617 [0.3159, 0.4049] | lr=0.01,epochs=60 | 0.0 | 0.363, 0.361, 0.362, 0.362, 0.361 | 8004 | 694.4 | ok |
| 64 | pretrained_pop_gru | 0.3618 [0.3172, 0.4079] | lr=0.01,epochs=60 | 0.0 | 0.362, 0.361, 0.363, 0.360, 0.362 | 8004 | 734.9 | ok |
| 64 | specialist | 0.3632 [0.3178, 0.4073] | lr=0.01,epochs=60 | 0.0 | 0.364, 0.362, 0.363, 0.364, 0.362 | 8004 | 738.4 | ok |

| k | contrast (negative favours first) | estimate | 95% CI | p (max boot, t) | sd | n |
|---|---|---|---|---|---|---|
| 0 | meta_reptile − pretrained_pop_gru | +0.0007 | [-0.0033, +0.0046] | 0.6440 | 0.0131 | 41 |
| 16 | meta_reptile − specialist | -0.0034 | [-0.0057, -0.0011] | 0.0030 | 0.0076 | 41 |
| 16 | meta_reptile − pretrained_pop_gru | +0.0002 | [-0.0007, +0.0010] | 0.6720 | 0.0029 | 41 |
| 16 | pretrained_pop_gru − specialist | -0.0036 | [-0.0059, -0.0014] | 0.0023 | 0.0077 | 41 |
| 64 | meta_reptile − specialist | -0.0015 | [-0.0029, -0.0000] | 0.0259 | 0.0048 | 41 |
| 64 | meta_reptile − pretrained_pop_gru | -0.0001 | [-0.0013, +0.0011] | 0.4543 | 0.0039 | 41 |
| 64 | pretrained_pop_gru − specialist | -0.0014 | [-0.0030, +0.0002] | 0.0447 | 0.0052 | 41 |

## garcia (tune n = 68, dev n = 138; uniform-choice NLL 0.6931)

| k | model | tune NLL [95% CI] | selected setting | lapse | per-draw mean NLL | params | CPU s (model, target) | causal |
|---|---|---|---|---|---|---|---|---|
| 0 | pretrained_pop_gru | 0.5906 [0.5611, 0.6204] | zero-shot | 0.2 | — | 8004 | 174.0 | ok |
| 0 | meta_reptile | 0.5963 [0.5682, 0.6231] | zero-shot | 0.2 | — | 8004 | 171.4 | ok |
| 16 | pretrained_pop_gru | 0.5684 [0.5389, 0.5990] | lr=0.01,epochs=20 | 0.05 | 0.565, 0.575, 0.563, 0.568, 0.571 | 8004 | 174.0 | ok |
| 16 | specialist | 0.5711 [0.5416, 0.5982] | lr=0.003,epochs=60 | 0.02 | 0.568, 0.577, 0.569, 0.569, 0.572 | 8004 | 176.8 | ok |
| 16 | meta_reptile | 0.5712 [0.5417, 0.6012] | lr=0.003,epochs=20 | 0.05 | 0.565, 0.573, 0.567, 0.573, 0.579 | 8004 | 171.4 | ok |
| 64 | pretrained_pop_gru | 0.5594 [0.5285, 0.5873] | lr=0.003,epochs=20 | 0.02 | 0.558, 0.560, 0.558, 0.558, 0.562 | 8004 | 174.0 | ok |
| 64 | meta_reptile | 0.5605 [0.5315, 0.5890] | lr=0.003,epochs=20 | 0.02 | 0.559, 0.562, 0.558, 0.559, 0.564 | 8004 | 171.4 | ok |
| 64 | specialist | 0.5628 [0.5332, 0.5912] | lr=0.003,epochs=60 | 0.01 | 0.562, 0.565, 0.561, 0.563, 0.564 | 8004 | 176.8 | ok |

| k | contrast (negative favours first) | estimate | 95% CI | p (max boot, t) | sd | n |
|---|---|---|---|---|---|---|
| 0 | meta_reptile − pretrained_pop_gru | +0.0058 | [+0.0029, +0.0093] | 1.0000 | 0.0140 | 68 |
| 16 | meta_reptile − specialist | +0.0000 | [-0.0044, +0.0042] | 0.5190 | 0.0180 | 68 |
| 16 | meta_reptile − pretrained_pop_gru | +0.0027 | [+0.0007, +0.0050] | 0.9960 | 0.0093 | 68 |
| 16 | pretrained_pop_gru − specialist | -0.0027 | [-0.0072, +0.0018] | 0.1228 | 0.0188 | 68 |
| 64 | meta_reptile − specialist | -0.0023 | [-0.0068, +0.0011] | 0.1390 | 0.0171 | 68 |
| 64 | meta_reptile − pretrained_pop_gru | +0.0011 | [-0.0006, +0.0033] | 0.8535 | 0.0086 | 68 |
| 64 | pretrained_pop_gru − specialist | -0.0034 | [-0.0096, +0.0010] | 0.1206 | 0.0234 | 68 |

## restless4 (tune n = 174, dev n = 348; uniform-choice NLL 1.3863)

| k | model | tune NLL [95% CI] | selected setting | lapse | per-draw mean NLL | params | CPU s (model, target) | causal |
|---|---|---|---|---|---|---|---|---|
| 0 | pretrained_pop_gru | 1.3128 [1.3014, 1.3238] | zero-shot | 0.02 | — | 8004 | 138.3 | ok |
| 0 | meta_reptile | 1.3343 [1.3240, 1.3435] | zero-shot | 0.05 | — | 8004 | 135.9 | ok |
| 16 | pretrained_pop_gru | 0.8498 [0.8076, 0.8926] | lr=0.01,epochs=20 | 0.01 | 0.862, 0.853, 0.836, 0.846, 0.852 | 8004 | 138.3 | ok |
| 16 | meta_reptile | 0.8522 [0.8080, 0.8986] | lr=0.01,epochs=20 | 0.01 | 0.870, 0.855, 0.839, 0.842, 0.855 | 8004 | 135.9 | ok |
| 16 | specialist | 0.8781 [0.8351, 0.9210] | lr=0.01,epochs=60 | 0.05 | 0.911, 0.866, 0.869, 0.867, 0.877 | 8004 | 134.0 | ok |
| 64 | pretrained_pop_gru | 0.7923 [0.7455, 0.8407] | lr=0.01,epochs=20 | 0.01 | 0.804, 0.794, 0.786, 0.789, 0.789 | 8004 | 138.3 | ok |
| 64 | meta_reptile | 0.7933 [0.7482, 0.8369] | lr=0.01,epochs=20 | 0.01 | 0.808, 0.796, 0.793, 0.782, 0.788 | 8004 | 135.9 | ok |
| 64 | specialist | 0.8115 [0.7677, 0.8572] | lr=0.003,epochs=60 | 0.01 | 0.821, 0.813, 0.804, 0.804, 0.815 | 8004 | 134.0 | ok |

| k | contrast (negative favours first) | estimate | 95% CI | p (max boot, t) | sd | n |
|---|---|---|---|---|---|---|
| 0 | meta_reptile − pretrained_pop_gru | +0.0216 | [+0.0153, +0.0282] | 1.0000 | 0.0434 | 174 |
| 16 | meta_reptile − specialist | -0.0260 | [-0.0329, -0.0175] | 0.0005 | 0.0511 | 174 |
| 16 | meta_reptile − pretrained_pop_gru | +0.0024 | [+0.0002, +0.0051] | 0.9880 | 0.0167 | 174 |
| 16 | pretrained_pop_gru − specialist | -0.0283 | [-0.0344, -0.0223] | 0.0005 | 0.0409 | 174 |
| 64 | meta_reptile − specialist | -0.0183 | [-0.0238, -0.0129] | 0.0005 | 0.0377 | 174 |
| 64 | meta_reptile − pretrained_pop_gru | +0.0009 | [-0.0011, +0.0032] | 0.8135 | 0.0147 | 174 |
| 64 | pretrained_pop_gru − specialist | -0.0192 | [-0.0248, -0.0140] | 0.0005 | 0.0358 | 174 |

## Tune stop rule (A3-2): candidate `meta_reptile` at k = 16, margin 0.02

| comparator | target | estimate | Holm p | meets |
|---|---|---|---|---|
| specialist | restless4 | -0.0260 | 0.0015 | True |
| specialist | garcia | 0.0000 | 0.5190 | False |
| specialist | described | -0.0034 | 0.0060 | False |
| pretrained_pop_gru | restless4 | 0.0024 | 1.0000 | False |
| pretrained_pop_gru | garcia | 0.0027 | 1.0000 | False |
| pretrained_pop_gru | described | 0.0002 | 1.0000 | False |
| hb_cog | — | — | — | pending (not merged); the rule cannot be met without it |

Targets meeting both available comparisons: 0; **tune stop rule met: False** (hb_cog pending). Exploratory; no gate is evaluated.

