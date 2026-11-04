# Phase III — Transfer and Continual Learning Results (Gates D and E, plus session headroom)

_Written by the director from the lane's code and outputs after the lane stopped (API spend limit). All numbers below
are copied from committed reports; nothing here is confirmatory. No test partition was opened._

## 1. Gate D — leave-one-family-out transfer (X3-014, exploratory, tune partition)

Question: does a GRU pretrained on **other** Psych-101 task families, optionally refined by family-level first-order
meta-learning (F4-style Reptile), predict new people in a held-out family better after fine-tuning on k = 16 people
than (a) a specialist trained on those 16 people only, and (b) a pooled-pretrained `pop_gru` fine-tuned the same way?
History-only inputs; the same fine-tuning grid (lr ∈ {0.003, 0.01} × epochs ∈ {5, 20, 60}) for every model, five
draws of the k people. Report: `reports/phase3/continual/x3_gate_d_tune/report.md`.

Tune NLL (nats/choice; lower is better):

| target (tune n) | k | pretrained `pop_gru` | `meta_reptile` (candidate) | specialist |
|---|---|---|---|---|
| restless4 (174) | 0 | **1.3128** | 1.3343 | — |
| restless4 | 16 | **0.8498** | 0.8522 | 0.8781 |
| restless4 | 64 | **0.7923** | 0.7933 | 0.8115 |
| garcia (68) | 0 | **0.5906** | 0.5963 | — |
| garcia | 16 | **0.5684** | 0.5712 | 0.5711 |
| garcia | 64 | **0.5594** | 0.5605 | 0.5628 |
| described (41) | 0 | **0.4984** | 0.4991 | — |
| described | 16 | **0.3702** | 0.3704 | 0.3739 |
| described | 64 | 0.3618 | **0.3617** | 0.3632 |

Key contrasts at k = 16 (negative favours the first model; paired over tune people; 95% bootstrap CI):

| target | candidate − specialist | candidate − pretrained | pretrained − specialist |
|---|---|---|---|
| restless4 | −0.0260 [−0.0329, −0.0175] | +0.0024 [+0.0002, +0.0051] | −0.0283 [−0.0344, −0.0223] |
| garcia | +0.0000 [−0.0044, +0.0042] | +0.0027 [+0.0007, +0.0050] | −0.0027 [−0.0072, +0.0018] |
| described | −0.0034 [−0.0057, −0.0011] | +0.0002 [−0.0007, +0.0010] | −0.0036 [−0.0059, −0.0014] |

**Findings.**
1. Cross-family pretraining helps a small target sample in one of three families by a margin that matters
   (restless4: −0.028 nats/choice vs a specialist), helps slightly in `described` (−0.004) and not reliably in
   `garcia`.
2. The meta-learning step adds nothing over plain pooled pretraining: it is slightly *worse* than the pretrained
   `pop_gru` in all three families at k = 16 (+0.0002 to +0.0027), and worse zero-shot.
3. Tune stop rule (amendment A3-2, margin 0.02 against every comparator in ≥ 2 of 3 targets, Holm): **not met** —
   0 targets meet both available comparisons; the hierarchical-Bayesian comparator (`hb_cog`) was not merged, so the
   rule could not have been met in any case.

## 2. Gate E — continual learning without forgetting (X3-011, development seeds)

Synthetic switching environments (contexts recur), report `reports/phase3/continual/x3_gate_e_dev/report.md`.
Reward relative to the oracle: F4 meta-learned GRU **0.829**, Bayes filter 0.814, online fine-tuning 0.740, replay
0.708, EWC 0.687. Savings on returning contexts: F4 −0.024, fine-tuning −0.003, EWC −0.003, replay +0.033, Bayes
filter +0.047. F4 adapts fastest but does **not** remember returning contexts (savings −0.021 vs fine-tuning and
−0.057 vs replay); the Gate E stop rule is not met. Confirmatory seeds 3000–3049 remain untouched.

## 3. Session headroom (X3-013, GBE risky choice)

The largest person-specific effect measured in Phase III is not an adaptation mechanism but **continuity**: a GRU
that loses its state between sessions is worse by +0.0195 [+0.0146, +0.0243] nats/choice than the same weights
carrying state across sessions (n = 318 people). This bounds what a persistent person-level representation could add
in session-based deployments and is the pivot question recorded in `BREAKTHROUGH_VERDICT.md`.

## 4. What this means

Transfer that works in Phase III is **plain pooled pretraining**, not the meta-learned or explicitly adaptive
mechanisms proposed as candidates. Continual learning with savings was not achieved by the candidate. These are
exploratory tune-partition results with selection optimism; they support the NO-GO verdict for H3 and do not
support any claim of a new transfer mechanism.
