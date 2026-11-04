# U1 — The Candidate Principle (selected hypothesis)

> **Status: REJECTED** (`SCIENTIFIC_VERDICT.md`). It was a candidate, never a discovery: the novelty review found no new
> principle among the six generated, and this was the most testable of them.
>
> Two preregistered confirmatory attempts failed the primary hypothesis, each passing 1 of 4 shift conditions. In
> attempt 2 the components that define the principle did not help:
> * active identification by information gain (M4 −0.016);
> * opening new concepts (M2 +0.005; a new concept in 2 of 500 unseen-role worlds);
> * accumulation without retraining (A +0.006).
>
> The text below is kept unchanged as the hypothesis that was tested.

## Name
**Interventional concept formation with active identification** (implemented as IRA, "interventional role
abstraction").

## 1. Precise scientific claim
A learner whose knowledge is indexed by concepts defined through the consequences of its own interventions — not by
perceptual similarity — and which (i) identifies those concepts in a new situation by running the interventions
that best discriminate between candidate assignments, (ii) treats perceptual cues only as a prior whose validity it
estimates online, (iii) opens a new concept when consequences contradict all known ones and (iv) re-identifies an
entity when its consequences contradict its current concept, retains more of an oracle's return than a recurrent
agent meta-trained on the training distribution when the world's laws are preserved but (a) perceptual correlations
reverse, (b) familiar concepts appear in unfamiliar combinations, (c) an unseen concept carries the reward, or
(d) an entity's concept changes mid-episode — and it accumulates new concepts across worlds without retraining.

## 2. Mathematical formulation
`MATHEMATICAL_FORMULATION.md` §2 (posterior over concept assignments and cue reliability, information-gain action
selection, surprise-triggered re-identification, additive consolidation).

## 3. Information available to the learner
Entity appearances (all four features, with no indication of which, if any, is informative); its own actions;
binary outcomes and rewards. Not available: roles, laws, cue dimension, cue validity, condition, change times.
The learner *does* assume the form of its own model (entities have latent concepts; outcomes of single and pairwise
interventions depend on concepts through Bernoulli laws). That structural assumption is the hypothesis being tested,
and the critique (`INDEPENDENT_CRITIQUE.md`) examines whether it amounts to built-in knowledge of the environment.

## 4. Internal state and its evolution
Persistent: the concept library (counts; grows by addition). Within a world: the exact posterior over concept
assignments and cue reliability, updated after every outcome; an entity's evidence is discarded when its surprise
exceeds a threshold. After a world: consolidation of the MAP assignment into the library.

## 5. How experience changes future capability
Each world adds counts to concept-level laws and appearance statistics and may add concepts. Future worlds start
from sharper priors over laws (fewer experiments to identify an entity), and newly added concepts make previously
unseen entities identifiable.

## 6. Difference from the closest predecessors
* Hierarchical Bayesian causal schemas (Kemp, Goodman & Tenenbaum 2010) learn object types and type-level causal laws
  from observation; this learner *acts*, chooses discriminating experiments, and tracks the validity of perceptual
  cues for type membership. Mathematically the inference is the same family (hierarchical Bayes over types).
* Latent-cause models (Gershman, Blei & Niv 2010) open new causes on surprising evidence; here new concepts are opened
  per entity inside one world and consolidated across worlds.
* Bisimulation / MDP homomorphisms aggregate states with equal dynamics given a known model; here equivalence is
  inferred from few interventions under uncertainty.
* Meta-learned recurrent agents (RL²) can, in principle, learn to infer types in context; they do not represent types
  explicitly and cannot add a type after training without further training.

The difference from each predecessor is a **combination and an acting setting**, not a new mathematical object.

## 7. Behaviour existing approaches are predicted to struggle with
Under the four shifts above, a meta-learned recurrent agent trained on the training distribution is predicted to lose
substantially more of the oracle's return than the proposed learner, and to show no improvement across repeated
worlds containing the same unseen concept.

## 8. What would falsify the hypothesis
The meta-learned agent retaining as much return under the shifts (the preregistered criterion in
`EXPERIMENTS_AND_BASELINES.md` §4 fails), or the advantage surviving when a component is removed (that component then
does not explain it), or the learner failing to accumulate across worlds with an unseen concept.
