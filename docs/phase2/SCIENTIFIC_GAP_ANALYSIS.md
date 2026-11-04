# Scientific Gap Analysis — why the central hypothesis did not pass

_Stage 1 output. Every number is from a committed artifact; interpretations are labelled as such.
Reproduction log: `reports/phase2/stage1_reproduction.json` (every committed result re-analysed from raw
logs, byte-identical)._

## Gap 1 — The environment cannot discriminate architectures (H1 failed)

**Evidence.** Task rules are linear readouts of a 3-d latent projection of 12 binary features
(ADR-0003). Plain per-family linear RL reaches 0.837 held-out accuracy; the developmental learner
0.849 (+0.012, CI [−0.008, +0.031]); the learner needs ~9 more episodes to reach criterion and is
worse by 0.046 under label noise 0.15.
**Interpretation.** A raw-feature linear learner is near-optimal within a family, so no representation
learning can pay off inside a family; the learner's 4-unit encoder bottleneck and plasticity schedule
cost sample efficiency without buying anything. Removing the plasticity schedule *improved* accuracy
(+0.012).
**Phase II response.** New task families where the Bayes-optimal within-family solution is not linear
in raw features: compositional rules, conjunctions, context-dependent mappings, non-stationary rules
with regime changes, delayed and partial feedback. Controls where specialists *should* win.

## Gap 2 — The social pathway never engages (Exp 3 failed; social ablation −0.004)

**Evidence.** Developmental agents average 0.018 friends over 15 years versus 0.49 for memory-only
agents under identical world rules; connected − isolated = −0.004; misinformation has no detectable
effect; learning-lab social ablation −0.004 [−0.013, +0.004].
**Interpretation.** Two independent faults: (a) the activity bandit's values-weighted reward steers
agents away from socialising, so demonstrations never arrive; (b) even in the lab, where demonstrations
are injected, trust-weighted imitation adds nothing — the demonstration channel is a small share of
input and skepticism down-weights correct teachers when the learner is confidently wrong.
**Phase II response (Candidate E).** Instrumented relationship formation (meetings, ties, trust
updates, demonstrations received, knowledge actually transferred), with mechanisms that make social
information *useful* (teaching reduces uncertainty faster than solo practice on hard items) rather
than a reward that encodes the desired result.

## Gap 3 — Ceiling and missing manipulation checks (Exp 4 failed, Exp 5 inconclusive, Exp 2 check failed)

**Evidence.** Adults plateau at 0.80–0.85; midlife education −0.003 [−0.015, +0.007]; cross-fostered
children all reach 0.83; Exp 2's manipulation check read a flag that is false after eligibility ends.
**Interpretation.** With linear rules every agent converges to the same plateau, leaving no variance
for late education or household environment to explain; uptake was never measured.
**Phase II response.** Non-stationary families (rules change at life events), uptake metrics
preregistered for every intervention, power analyses before running.

## Gap 4 — Humans learn cue–criterion mappings like exemplar models; our learner does not (Exp 7 primary failed)

**Evidence.** MCPL held-out NLL: instance-based 1.505, our memory-only baseline 1.529, population-static
1.602, **developmental 1.734**, plain RL 1.770, chance 2.197. Developmental's fitted lapse sits at the
grid edge (0.3), i.e. its predictive distribution is so miscalibrated that a 30% uniform mixture is the
best fix available.
**Interpretation.** MCPL is a 25-configuration → 9-level mapping with probabilistic feedback that
stops partway; humans behave as if retrieving similar past cases. Our learner's replay memory is used
only for consolidation, never for prediction, so it cannot express exemplar-like generalisation.
**Diagnosis on dev/tune participants (X-002).** The loss is concentrated in the *most predictable*
participants: top tercile by reference-model accuracy, gap +0.488 nats; bottom tercile +0.031;
r = +0.77 between the gap and predictability. Where humans apply a consistent, similarity-graded
cue→criterion mapping, the learner's 9-way categorical head cannot express the ordinal, graded
structure of the response scale, while exemplar retrieval does. The fault is structural, not a
tuning matter.
**Phase II response (Candidate A).** Hybrid parametric + exemplar prediction with (i) an ordinal /
metric output structure (graded similarity between neighbouring criterion levels), (ii) exemplar
retrieval by learned similarity, (iii) learned, uncertainty-driven arbitration; compared against the
instance-based model, memory-only, plain RL and a GRU sequence model under **equal tuning budgets**
(Phase I gave registry classes only lapse fitting — a protocol unfairness that Phase II removes).

## Gap 5 — The IGT advantage is unexplained (Exp 7 secondary passed)

**Evidence.** IGT held-out NLL: developmental 1.171 (best), instance-based 1.281, memory-only 1.286,
delta rule 1.304, plain RL 1.332; stable across three splits.
**Interpretation (hypothesis).** The learner's uncertainty-driven exploration and mastery estimates
produce exploration patterns closer to human deck sampling than a softmax delta rule. Unverified.
**Diagnosis (X-002).** The advantage is broad — 64% of dev/tune participants, every predictability
tercile, largest for the least predictable (−0.166) — consistent with better modelling of noisy,
exploratory deck sampling rather than of a few highly regular participants.
**Phase II response.** Ablate exploration/mastery components on the IGT **dev** partition; compare to a
PVL-Delta reference model; keep IGT as an anchor family and add a second bandit family (Psych-101
two-armed bandits, horizon task) to test whether the advantage generalises (Gate D/F).

## Gap 6 — Human validation is too narrow

Only two task families from one dataset; HeartSteps validates a causal pipeline, not an agent.
Psych-101 contains further learning tasks not yet used (two-armed bandits, horizon task, probability
learning, reversal learning). **Phase II response.** ≥ 3 families from ≥ 2 datasets, frozen partitions,
modern sequence baseline, calibration and subgroup reporting (`EXPERIMENT_REGISTRY.md`).

## Gap 7 — Methodological debts

Tuning seed 7 overlapped evaluation seeds (audit M-4; verdicts unchanged without it); 403/460 run
manifests carry `+dirty` code versions (M-3); preregistration hashes include `schema_version`; no power
analyses; tuning scripts not committed. **Phase II response.** Separate dev/tune/test partitions for both
seeds and participants; tuning scripts committed with their logs; clean-tree requirement for
confirmatory runs; registry-enforced budgets.

## Gap 8 — Affect is present in the world but untested

Stress/energy/social/motivation proxies drive activity rewards; no affective mechanism is evaluated
against human data; no model of how affect modulates learning rate, exploration or memory.
**Phase II response (Candidate D).** Literature-grounded affective state (e.g. reward-prediction-error
driven mood, uncertainty-driven arousal) with matched-complexity controls, tested on human tasks where
affect effects are documented.

## Gap 9 — Engineering limits on science

10k agents: 66% of a 1-year run is one-time prehistory; the job market scales superlinearly;
per-agent histories live in process memory (5.1 GB at 10k × 1 y). Not blocking Phase II science
(100–1,000 agents suffice) but blocking 10k multi-year studies.
