# U1 — Scientific Verdict

_Research Director, 2026-11-04. Based on two preregistered confirmatory attempts (`TRANSFER_RESULTS.md`), the
independent critique and its review of the revision (`INDEPENDENT_CRITIQUE.md`), and the novelty ledger
(`PRIOR_ART_AND_NOVELTY.md`)._

## Verdict

R1_VERDICT_HEADLINE

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

The critique showed that this agent never probed — 0 probes in 36,000 confirmatory steps — because of a defect in
how its information gain was computed. Attempt 1 therefore did not test active identification.

### Attempt 2 (revision R1; universes 4–8)

R1_EVIDENCE

## 3. Status of each part of the principle

R1_COMPONENTS

## 4. Novelty ledger — final levels

| level | status | reason |
|---|---|---|
| new principle | **no** | every component has close precedent (Kemp et al. 2006/2010; Gershman et al. 2010; Lindley/MacKay; Tsividis et al. 2021) |
| new algorithm | **no** | standard Bayesian updates, BALD-style information gain, collapsed Gibbs (infinite relational model) |
| new combination | modest, yes | the combination in one acting agent with cue-validity tracking and per-entity re-identification. It **did not** outperform generic meta-learners |
| new implementation | yes | `U1_IMPLEMENTATION/` |
| new empirical finding | R1_FINDING_LEVEL |

## 5. What the track learned (claims that survive)

R1_SURVIVING

## 6. Integrity record

* Both attempts were frozen by hash before confirmatory data existed, and each confirmatory invocation re-checked
  the hash.
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

R1_NEXT
