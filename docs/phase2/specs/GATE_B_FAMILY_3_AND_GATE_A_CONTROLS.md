# Spec: Gate B family 3 ("compositional rule-of-rules") and Gate A controls

_Written by the director on 2026-11-04 after the critic's design review and BEFORE any implementation.
To be implemented by a candidate-free lane (human-validation or infrastructure), never by the
cognitive-architecture lane, which must not see this family's code before the freeze._

## 1. Family `composition` (Gate B family 3)

**Why.** The candidate's own lane authored the v2 families and viewed its dev results on them. Gate B
needs one family whose rules the candidate's authors never saw.

**Structure (same contracts as `hpe.tasks`: `TaskItem`, `Observation`, `Feedback`; train/holdout
symbol pools; fixed eval sets; leakage checks).**
- Symbols: 12 binary features as in Phase I; 64 train / 32 holdout symbols.
- Two *primitive* rules drawn from the task seed: `r1` = a conjunction over feature group A (features
  0–5), `r2` = a parity over feature group B (features 6–11). Each yields a binary value per symbol.
- A *composition operator* `op` drawn per task seed from {AND, OR, XOR, IMPLIES}; the correct action for
  a single-slot item is `op(r1(x), r2(x))` (2 actions). A paired variant (2 slots) asks which of two
  symbols satisfies the composition (2 actions).
- *Shared-structure* variant: `r1`, `r2` are shared with the `conjunction` family's latent rules so a
  learner that has abstracted them can compose; *specialist-should-win control*
  (`shared_structure=False`): primitives drawn independently so nothing transfers.
- Label noise 0.05 on feedback only; difficulty for curriculum = margin proxy = number of features
  that flip the composed label (1 = hardest).

**Measurement.** `hpe.tasks.assess` on the fixed holdout set (chance 0.5); few-shot transfer probe on
clones; `practice()` unchanged.

**Pre-registered expectation (not a claim).** A linear raw-feature learner cannot exceed ≈ 0.75 (the
parity component is non-linear); exemplar learners are limited by holdout generalisation; a learner
that composes abstracted primitives can approach 1.0. If no model exceeds 0.75 the family is
uninformative and must be reported as such, not dropped.

**Tests required.** Determinism; train/holdout disjointness; class balance; linear probe ceiling
measured and recorded; specialist-control divergence from the shared variant.

## 2. Gate A controls (per candidate, synthetic)

- **Permuted-feedback copy.** Identical architecture and update calls, but every `Feedback` the
  learner receives has its reward / correct action replaced by a draw from the same marginal
  distribution (shuffled across the practice stream with a seeded permutation). Expected held-out
  gain over the frozen copy ≤ 0.02; a larger gain means "learning" is coming from something other
  than outcome information (e.g. exposure statistics) and must be explained before Gate A passes.
- **Age-script-only null.** The `static_persona` baseline with the same scripted age gates
  (perceptual gate, plasticity schedule applied to a non-learning readout). Candidate gain vs this
  null ≥ 0.10 is required; this is the H2 null (age labels and scripted maturation alone).
- Both run under `hpe.evaluation.learning_lab` with common random numbers, 10 seeds, dev seeds for
  development and seeds 500–509 only at confirmation.

## 3. Implementation ownership and sequencing

| Item | Lane | Must be done before |
|---|---|---|
| `hpe.tasks.v2.composition` + tests | human-validation or infrastructure (candidate-free) | Gate B freeze |
| permuted-feedback option in `learning_lab` | science/evaluation (candidate-free) | Gate A freeze |
| `age_script_only` policy class | science/evaluation | Gate A freeze |
| hash of this spec recorded in the freeze record | director | Gate B freeze |
