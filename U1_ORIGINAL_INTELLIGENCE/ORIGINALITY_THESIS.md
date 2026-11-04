# U1 — Originality Thesis

_Research track U1_ORIGINAL_INTELLIGENCE, opened 2026-11-04 by the Research Director. Separate from the Phase I–III
code (`packages/hpe`), which is preserved unchanged. This document records the questions, the candidate principles
generated, the novelty review of each, and why one was selected. It was written **before** any U1 experiment was
run._

## 0. Stance

The mandate asks for a computational principle of intelligence that is missing from existing learning algorithms.
A principle proposed in one working session by one research group is very unlikely to be new; the realistic
contributions are (a) a precise formulation that makes an old idea testable in a new way, (b) an executable system,
and (c) an experiment that could tell the idea apart from strong alternatives. Novelty is judged in
`PRIOR_ART_AND_NOVELTY.md` with five explicit levels: new implementation, new combination, new algorithm, new
empirical finding, new principle. Nothing in this track is called a new principle unless the evidence and the prior
art both allow it.

## 1. The fundamental questions and what is already known

| question | working answer from the literature | what remains open |
|---|---|---|
| What causes an inexperienced system to acquire reusable competence? | Structure shared across experiences that the learner can represent compactly (hierarchical Bayes, "learning to learn": Kemp, Goodman & Tenenbaum 2010; meta-learning: Wang et al. 2016, Duan et al. 2016) | Which representation of that structure survives changes the learner never saw |
| How can concepts form without being given? | Clustering by similarity, by predictive equivalence (bisimulation: Givan, Dean & Greig 2003; MDP homomorphisms: Ravindran & Barto 2002), or by latent causes (Gershman, Blei & Niv 2010) | Whether *interventional* equivalence (same consequences when acted on) is what makes concepts transferable |
| How to separate general principles from incidental correlations? | Invariance across environments (Peters, Bühlmann & Meinshausen 2016; Arjovsky et al. 2019), interventions (Pearl; Eberhardt) | Whether an agent's own experiments suffice, online, inside one short episode |
| What makes a capability transfer to a different situation? | Shared latent parameters (hidden-parameter MDPs: Doshi-Velez & Konidaris 2016), object/relational structure (Diuk, Cohen & Littman 2008; Kansky et al. 2017) | Transfer when *every* surface feature is new and familiar parts are recombined |
| How can knowledge accumulate without retraining? | Non-parametric / modular knowledge with sufficient statistics (Dirichlet-process models; schemas: Drescher 1991) | Whether accumulated concepts keep earlier competence intact while new ones are added |
| Can a learner recognise its limits and design experiments? | Bayesian experimental design (Lindley 1956; MacKay 1992), information-directed exploration (Russo & Van Roy 2014), knows-what-it-knows (Li, Littman & Walsh 2008) | Whether self-designed experiments are what makes identification fast in new situations |
| How do complex capabilities grow from simpler ones? | Composition of learned components (program/library learning: Ellis et al. 2021; options) | — (not addressed by the selected test) |

## 2. Candidate principles generated

Six candidates, deliberately different in kind. For each: the claim, the closest prior art (detail and citations in
`PRIOR_ART_AND_NOVELTY.md`), and a novelty assessment *before* any experiment.

**P1 — Interventional concept formation with active identification.** Concepts are equivalence classes of entities
whose consequences under intervention are indistinguishable; knowledge is stored at the concept level; in a new
situation the learner identifies concepts by choosing the interventions that best discriminate between candidate
assignments, treats perceptual cues only as a revisable prior whose validity it tracks, and opens a new concept
when the consequences contradict every existing one. Closest: hierarchical Bayesian causal schemas (Kemp et al.
2010), latent-cause models (Gershman et al. 2010), bisimulation / MDP homomorphisms, Bayesian experimental design.
_Assessment: every component exists; the combination as an acting agent compared with meta-learned recurrent
agents under controlled shifts is not, to our knowledge, standard. At best a new combination; a new empirical
finding is possible._

**P2 — Invariance certification by self-designed interventions.** A predictive relation becomes reusable knowledge
only after the agent's own interventions show it is invariant across contexts the agent deliberately varies;
others stay context-local. Closest: invariant causal prediction (Peters et al. 2016), IRM (Arjovsky et al. 2019),
active ICP (Gamella & Heinze-Deml 2020). _Assessment: close to active ICP; low novelty._

**P3 — Contradiction-driven refinement.** Knowledge units (condition → consequence) are refined only when a reliable
unit fails, by searching for the minimal distinguishing condition. Closest: Drescher's schema mechanism (marginal
attribution, 1991), ripple-down rules (Compton & Jansen 1990), XCS (Wilson 1995), incremental decision trees.
_Assessment: essentially Drescher's mechanism; not new._

**P4 — Composition-closed growth.** New capabilities arise only as compositions of verified ones, with a library
that grows by compression. Closest: DreamCoder (Ellis et al. 2021), options and hierarchical RL. _Assessment: not
new; large engineering cost to test well._

**P5 — Support-based self-knowledge with boundary experiments.** Each knowledge unit stores where it has been
verified; predictions outside that support are flagged and the agent experiments at the boundary. Closest: KWIK
(Li et al. 2008), R-max, learning-progress curiosity (Oudeyer, Kaplan & Hafner 2007), disagreement-based
exploration (Pathak et al. 2019). _Assessment: not new._

**P6 — Hidden-variable invention from structured residuals.** When prediction errors are structured across entities,
invent a latent variable that explains them. Closest: hidden-variable discovery in Bayesian networks (Elidan et al.
2000; Elidan & Friedman 2005), factor analysis of residuals. _Assessment: not new; P1's concept creation is a special
case._

## 3. Selection

**No candidate qualifies as a new principle on review.** Rather than relabel one, the track selects the candidate
whose *empirical* claim is most open and most decisively testable with the available compute:

**P1 is selected as the primary hypothesis.** Reasons: (i) it addresses four of the mandate's questions in one
mechanism (concept formation, general vs incidental, transfer, uncertainty-driven experiments) plus knowledge
accumulation; (ii) it makes sharp predictions where a strong generic learner — a recurrent agent meta-trained on
the same task distribution, the kind of model that won Phase III — should fail: perceptual correlations reversed,
familiar concepts in unfamiliar combinations, an unseen concept, and a concept that changes mid-episode; (iii) each
component can be removed by an ablation while the rest of the system is kept.

The honest framing for everything that follows: **U1 tests whether a known idea (concepts defined by interventional
consequences, identified by experiment) provides an advantage that a strong meta-learned agent cannot match under
controlled shifts.** A positive answer would be a new empirical finding about a combination, not a new principle.
A negative answer would be equally informative: it would mean that the generic meta-learner already captures what
this structure provides.

## 4. Design commitments made before any experiment

* One standalone learning system (`U1_IMPLEMENTATION/`), no language model, no access to hidden rules.
* Procedurally generated worlds with hidden causal laws; held-out worlds with new surface features, reversed
  perceptual correlations, unfamiliar combinations, an unseen concept, and a mid-episode change.
* Strong alternatives with the same information: a recurrent agent meta-trained by actor-critic on far more episodes
  than the proposed learner sees, a Bayesian learner with the same machinery but no concepts, a bandit learner, an
  oracle and a random agent.
* Ablations removing each component (active identification, open concept creation, cue-validity tracking,
  contradiction-triggered re-identification).
* Thresholds, statistics and untouched confirmatory seeds fixed in `EXPERIMENTS_AND_BASELINES.md` before the
  confirmatory run; development seeds only for building and tuning (tuning on the in-distribution condition only).
