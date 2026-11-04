# Research Thesis v2 — Phase II

_Status: charter, written before any Phase II experiment. Thresholds below are fixed now and may not be
loosened after data are seen. Phase I results are the baseline (`RESEARCH_RESULTS.md`) and are never
re-labelled._

## 1. The refined scientific question

Can developmental experience produce reusable computational representations of cognition, learning,
affective responses and social adaptation that predict real human decisions and responses to
interventions better than strong non-developmental models?

"Developmental" means: capabilities that change through an agent's own interaction history under
persistent state, not through age labels, scripted maturation, prompt length or accumulated text.

## 2. What Phase I established (the baseline we build on)

| Finding | Evidence | Consequence for Phase II |
|---|---|---|
| Learning in the synthetic world is real and generalises to unseen symbols | ablations: freezing learning −0.41, withholding feedback −0.27 | Gate A machinery exists; re-verify per candidate |
| Developmental ≫ static persona (+0.41) and memory-only (+0.08) | H1 arm table | Weak baselines are no longer informative; compare only against the strongest |
| Developmental ≈ plain RL within a family (+0.012, CI incl. 0); worse under label noise | H1 primary + sensitivity | Linear task rules make raw-feature learners near-optimal; the environment must change |
| Shared representation transfers (+0.09 AUC) and transfers negatively when structure differs (−0.09) | Exp 6 | Gate C machinery exists; keep the orthogonal control |
| Social pathway inert: 0.018 friends in 15 years; demonstrations contribute −0.004 | Exp 3, ablations | Mechanism must be repaired before H3 can be tested |
| No adult adaptation / no transmission detectable at a 0.80–0.85 plateau | Exp 4, Exp 5 | Ceiling effects; need non-stationary, higher-ceiling tasks and uptake metrics |
| On real human MCPL choices the learner loses to exemplar models by +0.23 nats; on IGT it is the best of 8 | Exp 7 | Hybrid parametric/exemplar is the first candidate; IGT advantage is the anchor to explain |
| Causal estimator recovers a randomised human effect | Exp 7b | Causal pipeline usable; no agent claim |

## 3. Hypotheses and their falsifiers

**H1 — Developmental learning advantage.** On Phase II task families with non-linear, compositional
and non-stationary structure, a developmental architecture learns more efficiently or generalises
better than equally resourced plain RL, exemplar-memory and sequence-model baselines.
*Falsified if* no candidate beats the strongest baseline by the Gate B margin on the confirmatory
synthetic seeds.

**H2 — Experience-dependent development.** Persistent interaction history plus structured
developmental mechanisms produce learning effects not explained by age labels, scripted maturation or
textual memory.
*Falsified if* removing the mechanisms (while keeping age scripts) leaves held-out performance unchanged
within 0.02, or if an age-script-only control matches the developmental arm.

**H3 — Integrated affective and social adaptation.** Computational affect and social-relationship
state improve prediction of human behaviour where those factors are relevant.
*Falsified if* affect/social variants do not reduce held-out human NLL by ≥ 0.02 nats on tasks where
the literature predicts affect/social effects, or if they do so only by adding free parameters that a
matched-complexity control also exploits.

**H4 — Cross-domain generalisation.** A shared developmental representation transfers across distinct
task families and environments more efficiently than independent specialists.
*Falsified if* pretraining on family set A does not reduce few-shot loss on unseen family B by the
Gate F margin, or if the orthogonal-structure control shows the same "benefit".

## 4. Gates with thresholds (fixed now)

| Gate | Test | Pass threshold | Partition |
|---|---|---|---|
| A — genuine learning | per candidate: parameters change only with experience; held-out gain vs frozen copy | ≥ 0.10 accuracy, CI excl. 0, 10 seeds | synthetic dev → confirmatory seeds 500–509 |
| B — strong-baseline advantage | candidate − strongest of {plain RL, exemplar kNN, GRU sequence model, Bayesian rule learner}, chosen post hoc (conservative) | ≥ 0.05 absolute (or ≥ 10% relative) held-out accuracy, paired 95% CI excl. 0, in ≥ 2 of 3 v2 families; not worse by > 0.02 in any family | synthetic confirmatory seeds 500–509, families frozen in the registry |
| C — transfer | few-shot AUC on an unseen family, shared vs orthogonal structure | ≥ 0.03 AUC advantage where shared; negative transfer quantified where orthogonal | as B |
| D — human correspondence | held-out participants, ≥ 3 task families from ≥ 2 datasets | candidate NLL ≤ best fitted cognitive/exemplar/sequence model − 0.02 nats/choice, CI excl. 0, in ≥ 2 of 3 families; not worse by > 0.02 in any; ECE reported | frozen human test partitions (registry) |
| E — causal decision improvement | prospective study | out of scope for this phase (needs participants and ethics review); never inferred from D | — |
| F — cross-domain generalisation | representation trained on family set A evaluated few-shot on domain B (synthetic→human bandits, bandits→cue learning) | ≥ 0.02 nats/choice better than training from scratch, CI excl. 0 | frozen human test partitions |

A gate passed on synthetic data never substitutes for D. A model that wins one human task and loses
another is reported as exactly that.

## 5. Stop rules

Abandon a candidate when, on the **dev** partition, it fails to beat the strongest baseline after its
compute budget (registry) is spent, or when its advantage disappears under matched-complexity controls.
Pivot the programme if no candidate passes Gate D on at least one human family beyond IGT.
