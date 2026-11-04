# U1 — Prior Art and Novelty Ledger

_Novelty levels used: **new implementation** (known method, new code); **new combination** (known components joined);
**new algorithm** (a new procedure with a definable mathematical difference); **new empirical finding** (a
reproducible result not previously reported); **new principle** (a new explanatory mechanism of learning). The review
was done from the director's knowledge of the literature with web checks of the closest items (marked ✓; pages
opened 2026-11-04: Kemp et al. 2010, Kosoy et al. 2022, Doshi-Velez & Konidaris HiP-MDP / Killian et al. 2017). Other
references were not re-opened and are cited from memory; any of them could be imprecise in detail, and the absence
of a closer match is not proof that none exists. Five rows were added after the independent critique (finding F10;
the critic checked them by web search; the director cites them from memory) — the original review missed them, and
Alchemy in particular already reports the generic comparison this track set out to make._

## 1. Ledger for the selected mechanism (interventional concept formation with active identification)

| component | closest prior work | exact mathematical similarity | exact difference | experimental evidence for the difference | limitation | verdict |
|---|---|---|---|---|---|---|
| Concepts as latent types with type-level causal laws, learned hierarchically across objects | ✓ Kemp, Goodman & Tenenbaum (2010), *Learning to learn causal models*, Cognitive Science | Same model family: latent object types, type-level Bernoulli/Beta laws, hierarchical Bayesian inference, faster learning about new objects of a known type | Ours is embedded in an acting agent; types are inferred from the agent's own interventions inside short episodes; consolidation across episodes is a hard MAP update | Filled from results: `SCIENTIFIC_VERDICT.md` | No change to the inference itself | **not new** (the model) |
| Opening a new concept when evidence contradicts known ones | Gershman, Blei & Niv (2010) latent-cause inference (Chinese-restaurant prior); Dirichlet-process mixtures | CRP-style prior over membership; marginal-likelihood comparison of existing vs new | Per entity within one world, with laws for the new concept learned in the same world and consolidated after it | `ira_closed` ablation in the unseen-concept condition | Hard consolidation can fragment concepts | **not new** |
| Equivalence by consequences rather than appearance | Bisimulation (Givan, Dean & Greig 2003; Ferns, Panangaden & Precup 2004), MDP homomorphisms (Ravindran & Barto 2002) | Entities equivalent iff their consequence distributions match | Equivalence inferred from few noisy interventions, probabilistically; not a model-minimisation algorithm | `entity_bayes` (no concepts) vs IRA | — | **not new** (concept) |
| Choosing experiments that discriminate between hypotheses | Lindley (1956); MacKay (1992); query by committee (Seung, Opper & Sompolinsky 1992); information-directed sampling (Russo & Van Roy 2014); active causal learning in people (Bramley et al. 2017) | Expected information gain about the latent assignment (BALD form) | Combined additively with expected reward, decaying with the horizon | `ira_noactive` ablation | Heuristic weighting, not Bayes-optimal control | **not new** |
| Perceptual cue as a revisable prior with tracked validity | Attention / cue-weight learning in categorisation (Kruschke 1992, ALCOVE); invariance across environments (Peters et al. 2016; Arjovsky et al. 2019) | A mixture weight on a feature-based likelihood, inferred from evidence | Validity is re-estimated inside each world from the agent's own interventional evidence | `ira_nocuetrack` in the cue-reversed condition | Three-point grid for $w$ | **not new** |
| Surprise-triggered re-identification | Change-point detection (Adams & MacKay 2007), latent-cause switching, volatility models | Discard evidence after a surprise threshold | Per entity | `ira_nochange` in the role-switch condition | Threshold fixed a priori | **not new** |
| Knowledge accumulates without retraining | Non-parametric Bayes with sufficient statistics; Drescher (1991) schemas; modular continual learning | Additive count updates | — | Sequential unseen-concept experiment | Counts never forget wrong consolidations | **not new** |
| The whole agent evaluated against meta-learned recurrent agents under controlled shifts | ✓ Kosoy et al. (2022), causal overhypotheses: RL agents vs children in a blicket environment; RL² (Duan et al. 2016) / Wang et al. (2016) | Same question family: do generic RL / meta-RL agents learn transferable causal structure? | Our shifts (cue reversal, recombination, unseen concept, change) and our matched-information comparison are specific to this study | The confirmatory experiment | One environment family, designed by the same team as the learner | **possible new empirical finding** (to be decided by the result) |
| **(added after the critique, F10)** A structured Bayesian observer with the true task structure vs meta-RL agents | Wang et al. (2021), *Alchemy: a benchmark and analysis toolkit for meta-reinforcement learning agents* (NeurIPS Datasets & Benchmarks) — includes a Bayesian ideal observer far ahead of the meta-RL agents | Same comparison: an agent with the environment's latent causal structure built in vs generic meta-RL | Ours tests held-out *shifts* and includes a meta-learner trained on the same kinds of variation (R1) | `TRANSFER_RESULTS.md` | — | **not new** as a generic finding ("structured Bayes beats meta-RL"); only the specific shift/robustness result could be |
| **(added, F10)** Theory-based exploration and planning with object-level causal models | Tsividis et al. (2021), EMPA (*Human-level reinforcement learning through theory-based modeling, exploration, and planning*) | Object-type causal theories, Bayesian theory learning, exploration aimed at uncertain theory components | EMPA plans over Atari-like games with a hand-designed theory language; ours is a minimal latent-type model with BALD-style probing | — | — | **not new** (principle and much of the mechanism) |
| **(added, F10)** Causal reasoning acquired by meta-RL | Dasgupta et al. (2019), *Causal reasoning from meta-reinforcement learning* | Meta-RL agents learn to perform interventions and causal inference in context | We use such an agent as the comparator rather than proposing it | — | — | context for the comparator |
| **(added, F10)** Bayes-adaptive meta-RL with an explicit task posterior | Zintgraf et al. (2020), VariBAD | Explicit (approximate) posterior over a latent task variable driving the policy | Ours is exact enumeration over discrete per-entity types rather than a learned continuous task embedding | — | — | **not new** (posterior-driven adaptation) |
| **(added, F10)** Clustering entities into types by relational data | Kemp, Tenenbaum, Griffiths, Yamada & Ueda (2006), infinite relational model | CRP over entity types with type-pair Beta-Bernoulli relations, Gibbs inference | `recluster.py` *is* this model applied to the agent's stored worlds | — | — | **not new** |
| Latent per-task parameters for transfer | ✓ Hidden-parameter MDPs (Doshi-Velez & Konidaris 2016; Killian et al. 2017) | Latent variables explain variation across related tasks | Ours: discrete per-entity latent types with compositional pair laws, not a continuous task-level parameter | — | — | **not new** (related framework) |

## 2. Ledger for the rejected candidates (summary)

| candidate | closest prior work | verdict |
|---|---|---|
| P2 invariance certification by self-designed interventions | Invariant causal prediction (Peters, Bühlmann & Meinshausen 2016); IRM (Arjovsky et al. 2019); active ICP (Gamella & Heinze-Deml 2020) | not new |
| P3 contradiction-driven refinement | Drescher (1991) schema mechanism; ripple-down rules (Compton & Jansen 1990); XCS (Wilson 1995) | not new |
| P4 composition-closed growth | DreamCoder (Ellis et al. 2021); options / hierarchical RL | not new |
| P5 support-based self-knowledge | KWIK (Li, Littman & Walsh 2008); R-max; learning progress (Oudeyer, Kaplan & Hafner 2007); disagreement exploration (Pathak, Gandhi & Gupta 2019) | not new |
| P6 hidden-variable invention | Elidan, Lotner, Friedman & Koller (2000); Elidan & Friedman (2005) | not new |

## 3. Overall novelty verdict (before the experiment)

| level | status |
|---|---|
| new principle | **no** |
| new algorithm | **no** — every update rule is a standard Bayesian or heuristic rule |
| new combination | yes, modest: the combination in one acting agent with tracked cue validity and per-entity re-identification |
| new implementation | yes |
| new empirical finding | **open** — decided by the confirmatory experiment and the critique (`SCIENTIFIC_VERDICT.md`) |
