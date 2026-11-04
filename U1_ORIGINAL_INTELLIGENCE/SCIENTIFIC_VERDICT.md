# U1 — Scientific Verdict

_Research Director, 2026-11-04. Based on two preregistered confirmatory attempts (`TRANSFER_RESULTS.md`), the
independent critique and its review of the revision (`INDEPENDENT_CRITIQUE.md`), and the novelty ledger
(`PRIOR_ART_AND_NOVELTY.md`)._

## Verdict

**No new computational principle of intelligence was discovered. The selected hypothesis — that interventional
concept formation with active identification lets a learner retain more of its performance than a strong
meta-trained recurrent agent under held-out shifts — is rejected as preregistered, in the one co-designed
environment family tested.**

It failed in both preregistered confirmatory attempts:
* attempt 1: 1 of 4 shift conditions, frozen agent;
* attempt 2: 1 of 4, after the independent critique's fixes.

The one condition that passed (role switch) is attributable to component (iv) of the principle, re-identification
after contradiction, which is implemented as a fixed-threshold heuristic.

The other defining mechanisms did not contribute measurably:
* experiment selection by information gain;
* opening new concepts on contradiction;
* accumulation without retraining.

What clearly helped is known: change detection, and ordinary hierarchical-Bayesian transfer through a concept
library.

**Decision: REJECT.** No claim of a new principle, algorithm or empirical finding is made.

## 1. What was asked and what was done

The mandate asked for a principle missing from existing learning algorithms, not a renamed or recombined known
architecture, tested against strong baselines on held-out causal structure, unfamiliar combinations and changing
conditions.

1. Six candidate principles were generated and reviewed against prior art. **None qualified as new**
   (`ORIGINALITY_THESIS.md`).
2. The most testable one was selected and stated in its eight required parts (`NEW_INTELLIGENCE_PRINCIPLE.md`):
   concepts defined by interventional consequences, identified by self-chosen discriminating experiments, with cue
   validity tracked online, new concepts opened on contradiction, and re-identification after change.
3. It was implemented as an exact Bayesian agent (IRA, `U1_IMPLEMENTATION/u1/ira.py`) and tested in procedurally
   generated hidden-law worlds. The comparators were meta-trained recurrent agents with matched information and
   far more training experience.

## 2. Evidence

### Attempt 1 (agent frozen at 6776cfc; universes 1–3)

The meta-trained GRU matched or beat IRA in-distribution and under cue reversal, recombination and an unseen
concept. **H1 was not supported (1 of 4 shift conditions).**

IRA won only under a mid-episode role switch (+0.173). The ablation shows the cause: the hand-written surprise
reset (M3 +0.180).

The critique showed that this agent never probed — 0 probes in 56,000 confirmatory steps — because of a defect in
how its information gain was computed. Attempt 1 therefore did not test active identification.

### Attempt 2 (revision R1; universes 4–8)

R1 is the agent that does implement information gain over concepts and laws, with an unbiased prior. It was tested
on five new universes with universe-level statistics. The narrow meta-trained GRU was tested in stage 1.

| condition | IRA − GRU (mean of 5 universe means) | universes > 0 | passes |
|---|---|---|---|
| iid | −0.031 | 1 | non-inferior (≥ −0.05) |
| cue_reversed | +0.084 | 2 | no |
| recombined | −0.006 | 1 | no |
| novel_role | −0.044 | 2 | no |
| role_switch | **+0.214** | 5 | yes |

**H1 is not supported**, so the stage-2 comparators were not needed. On dev, the relational meta-learner trained on
broad variation exceeded IRA in four of five conditions, including the role switch (R1-D).

## 3. Status of each part of the principle

| component (from `NEW_INTELLIGENCE_PRINCIPLE.md` §1) | evidence | status |
|---|---|---|
| (i) identify concepts by running the most discriminating interventions | M4 −0.016 (β = 0 is at least as good); β = 1 worse everywhere; in attempt 1 the agent never probed | **refuted** in this environment family, for this implementation (BALD-style information gain added to expected reward, β selected on `iid`): β = 0 is at least as good (M4 −0.016, not significant) |
| (ii) perceptual cues as a prior whose validity is estimated online | M1 +0.019 (attempt 1); +0.120 in attempt 2, the same sign in all five universes (+0.031, +0.084, +0.031, +0.005, +0.450) but very uneven in size, so not significant | **not established** (no analysis explains the unevenness) |
| (iii) open a new concept when consequences contradict all known ones | M2 0.000 / +0.005; a new concept in 2 of 500 unseen-role worlds (attempt 2) and in none in attempt 1. The CRP prior puts a fresh label 5–7 nats behind every established concept | **not supported; not exercised as implemented** |
| (iv) re-identify an entity when its consequences contradict its concept | M3 +0.180 / +0.221 | **works**, but it is a designer-written rule (fixed threshold); meta-learners trained with mid-episode switches matched or beat it on dev (their dev training included t = 10, so this is not a held-out comparison) |
| accumulate concepts across worlds without retraining | A +0.027 (p = 0.009, positive in 3/3 universes, below the 0.05 margin) / +0.006; K unchanged except universe 4 of attempt 2 (7 → 8); additive consolidation alone collapsed the library in development (D-2) and needed periodic re-clustering of stored worlds | **not supported** |
| knowledge indexed by concepts transfers across worlds | M5 +0.083 / +0.135 vs. no library | **works** — this is hierarchical Bayesian transfer (Kemp et al. 2010), not new |

## 4. Novelty ledger — final levels

| level | status | reason |
|---|---|---|
| new principle | **no** | every component has close precedent (Kemp et al. 2006/2010; Gershman et al. 2010; Lindley/MacKay; Tsividis et al. 2021) |
| new algorithm | **no** | standard Bayesian updates, BALD-style information gain, collapsed Gibbs (infinite relational model) |
| new combination | modest, yes | the combination in one acting agent with cue-validity tracking and per-entity re-identification. It **did not** outperform generic meta-learners |
| new implementation | yes | `U1_IMPLEMENTATION/` |
| new empirical finding | **no** | the preregistered empirical claim failed twice; the parts that worked are known (hierarchical Bayesian transfer, change detection) |

## 5. What the track learned (claims that survive)

Only descriptive statements about this environment family survive (revised after the post-results critique,
`INDEPENDENT_CRITIQUE.md` P-1–P-9).

**Development-only observations** (one universe, not held out): the relational meta-learner trained on broad
variation scored above IRA in four of five conditions, including `role_switch` (0.770 vs 0.682). Its training
switch times included t = 10. The broad GRU was at 0.673 there.

1. A meta-trained recurrent agent (flat GRU, RL²-style, 2.56 M episodes), trained without any shift:
   * matched or beat the structured Bayesian agent under recombination and an unseen concept in both attempts;
   * under cue reversal it matched IRA in attempt 1 but fell sharply in two of five universes in attempt 2
     (0.936 → 0.733 pooled; IRA ahead by +0.084);
   * its clear failure is a mid-episode change (post-switch return ≈ 0.10).
   Over both attempts it matched or beat IRA in 3 of 4 shifts in attempt 1 and in 2 of 4 in attempt 2.
2. A hand-written surprise-reset rule gives a structured agent robustness to change (post-switch 0.50–0.55, against
   0.10 without it).
3. In 20-step episodes with these laws, choosing actions by expected information gain about concepts and laws was no
   better than greedy exploitation of the posterior (β = 0 at least as good), and more information seeking was
   worse.
4. Additive MAP consolidation of concepts collapses into a few mixed concepts. Periodic re-clustering of stored
   experience (an infinite relational model over the agent's own worlds) is needed, so "accumulation without
   retraining" did not hold even in development.

## 6. Integrity record

* Both attempts were frozen by hash before confirmatory data existed. Every confirmatory training and evaluation
  invocation re-checked the hash.
* One post-freeze change: R1's analysis code (`stats.py`, deviation R1-X1) had a one-line bug in its stage-2
  presence check. It was fixed after the run. Per-cell means had been printed in the evaluation logs during the
  run, but no contrast or test had been computed. The fix cannot change any number: the critic reproduced the
  committed summary byte for byte. No evaluation code changed.
* Attempt 1's generator defect (universe 2) was disclosed, and the frozen code was run without the impossible
  condition.
* The critique was written by an agent that did not build the system. Every blocker it raised was verified and
  either fixed (R1) or disclosed.
* Its R1 review found a crash in the R1 runner before the freeze. That crash would have been misreported as a
  scientific outcome by the stage gate.
* Development history is disclosed in full, including that IRA's shift performance was seen while the environment
  was designed (D-3, D-5, R1-D).
* No test result, citation or number in the U1 documents is invented. Each comes from a committed file or a
  committed log.

## 7. What would be needed for a defensible new principle

Nothing in this track supports continuing to develop IRA as a candidate principle. What the evidence suggests
instead:

* **The bar is a meta-learner with relational structure and broad training, not a narrow GRU.** On dev, `rel_broad`
  was the strongest agent overall. Any future candidate principle should be tested against it from the start, on
  universes not used in design.
* **Novelty detection under strong knowledge is an open, concrete question, not a finding of this track.**
  * Whether a learned library makes an unseen concept harder to recognise was not preregistered, and the data are
    mixed. Library − no library on `novel_role` was:
    * +0.027, +0.068, +0.026 in attempt 1;
    * −0.083, −0.206, +0.027, −0.036, +0.132 in attempt 2.
  * A mechanism that recognises when well-established knowledge does not apply, without a designer-set threshold,
    could be the subject of a new, separately preregistered track. It would first need
  its own prior-art review (latent-cause inference, change-point and open-set recognition are the obvious
  precedents).
* **Environments must be designed by someone other than the people who build the learner.** Critique F3 and F7
  showed that co-design was this track's largest validity threat. A third party, or a procedurally generated
  family fixed before any learner exists, would remove it.

This verdict closes U1 as preregistered. The track's code, data, failed attempts and critiques remain in the
repository as the record.
