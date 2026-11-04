# Scientific Limitations — Phase II (as of 2026-11-04)

Every item names the evidence that establishes it. Nothing here is softened by later text.

## 1. No confirmatory result exists for any gate

All Phase II architecture results (X-003 … X-010) are exploratory: dev/tune partitions, selection-
optimistic by construction. The only test-partition evaluations are the causal lane's C-001 (not
supported) and C-002 (passed, with recorded protocol deviations), and neither bears on Gates A–F.

## 2. The strongest baseline is a population-fitted sequence model, and we lose to it

A GRU with ≤ 2k parameters fitted on other participants beats every per-participant learner we have
in 8 of 9 Psych-101 families, by +0.07 to +0.45 nats (X-009). The Phase I "IGT best of 8" result
does not survive this comparison. Candidate A beats the exemplar reference on MCPL (dev, −0.040,
X-003) but its standing against the GRU on the pristine families is the subject of X-010.

## 3. The developmental architectures do not learn non-linear compositional structure

No architecture — Phase I learner, plain RL, memory-only, or Candidate A — exceeds the linear-probe
ceiling on the candidate-free `composition` family (0.672) or 0.68 on `conjunction`; none learns the
parity primitive (X-004, X-008). Gate B is therefore not evaluable: the designed hard families are
unsolved by everyone. This is the central negative finding of Phase II on H1.

## 4. Participant contact and the "independent population" requirement

Every MCPL and IGT participant was touched in Phase I, and Phase I artifacts hold per-participant
scores for 63 MCPL and 137 IGT participants now in the Phase II test partition (critic audit). The
protection is procedural and code-enforced, not informational; MCPL/IGT can only be secondary
families. Only one dataset family (Psych-101 / Psych-201 share a source) has adapters; the second,
independent dataset Gate D requires has been identified (OpenNeuro candidates with CC0/PDDL licences)
but not adapted. Gate D's "≥ 2 datasets" clause is unmet.

## 5. Small families and low power

`horizon_task/exp2` (test n = 3) and `probability_learning` (test n = 9) cannot count toward any gate
(minimum-n rule). HeartSteps has 37 participants (15 in test): power for moderators of 0.20 is
50–74 % (X-005); C-001's null on heterogeneity is partly a power statement. Synthetic society effects
of a few thousandths of accuracy read as zero with 5 seeds (social lane).

## 6. Published foundation-model figures are not comparable, and we could not run the models

Centaur-70B and Minitaur-8B cannot run in this environment (weights 140 GB / 16 GB bf16; CPU-only,
15.7 GB RAM). Their published per-experiment NLLs use a different split, scoring unit (token-level
over text), information (full transcript) and no lapse mixing (`STATE_OF_THE_ART_BENCHMARK.md` §2–3).
"Simile" has no public documentation to compare against. **No state-of-the-art claim is made.**

## 7. The social and midlife mechanisms are repaired or diagnosed, not validated

Social v2 repairs friendship formation and makes demonstration information worth ≈ +0.021 accuracy,
with the information-vs-reward control attributing it correctly; but connected − isolated is still
≈ 0 because family teaching substitutes for peers in this world (social lane). Nothing about human
social learning is established. Midlife adaptation remains unmeasured (uptake metric absent in
Phase I); intergenerational transmission is undetectable at the task ceiling.

## 8. Affect is untested

Candidate D was never executed (`AFFECTIVE_MODEL_REPORT.md`). H3's affective half has no evidence.

## 9. Causal validity is an estimator property, not a product property

The WCLS estimator is valid by simulation and suggestions raise the proximal outcome on held-out
HeartSteps participants; a tailored policy learned on 22 participants adds nothing over always-send
(C-002). Gate E (prospective improvement from model-guided selection) has not started and cannot be
inferred retrospectively. Nothing transfers from HeartSteps to the synthetic society or to Future Self.

## 10. Protocol debts carried forward

Phase I tuning overlapped evaluation seeds and selected lapse on fitting participants (fixed in Phase
II rules); the causal lane ran its own test confirmations (critic re-run pending); grids were unequal
in the cognitive lane's dev work (equal-size re-tune required before any freeze); the age-script-only
null cannot differ from static persona on accuracy (judged on log-loss); the deployed site's
publication waits on a repository setting.
