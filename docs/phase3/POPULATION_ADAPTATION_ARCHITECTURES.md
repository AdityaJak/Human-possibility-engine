# Population Adaptation Architectures (Phase III design team)

_Lane `research/adaptation`. Everything here is **exploratory** (dev fit, tune select and report; synthetic
toys). No gate is evaluated in this document. Results are copied from committed artifacts under
`reports/phase3/candidates/`; numbers are not rounded in our favour. A name is not novelty: for every
mechanism we name the existing method it most resembles and the baseline that would make it redundant._

## 0. Common ground

**The bar.** A recurrent population model already adapts to each person *in context*: its hidden state is
updated by every choice and outcome (Phase II X-009/X-010; thesis §1). A candidate adds value only if it
beats (i) the same population network with its recurrent state running and the explicit adaptation
switched off, (ii) a separately trained same-size `pop_gru` twin (Gate B1 comparator after critic G-1),
(iii) conventional adapters on the same backbone (Gate B2; evaluation lane), and (iv) the strongest
tune-selected comparator (Gate C).

**Interface (harness-owned modes).** Every candidate subclasses `hpe.population.core.adaptation.AdaptiveBase`
and implements only `init_state` (the prior person state), `update_state(state, history)` (history = a
truncated, re-keyed copy of the first t trials), `predict_with_states(seq, states)` and `twin_config`.
The harness builds the adapt schedule from the person's own prefixes every `update_every` trials, the
frozen schedule (`init_state` throughout) and the length-matched donor schedule (Gate A2 control). The
candidates do not implement `predict_logprobs`; `check_causal` passes in every mode (unit tests).

**Equal treatment.** Every population network here (`pop_gru`, the F1/F3 backbone, the F2 world model,
the F3 GRU challenger) is trained by `core.nets.train` with the core defaults (Adam, batch 8, ≤ 150
epochs, patience 10, plateau lr halving, early stopping on a seeded 20 % participant split of dev). The
F1/F3 backbone *is* a `pop_gru` (same class `nets.PopRNN`, same trainer, same config keys); it is
trained once per (training data, config) inside a process and shared by the F1/F3 variants so that their
ablations are exactly paired. Its CPU cost is attributed to every model that uses it.

**Code.** `packages/hpe/population/candidates/` — `common.py` (shared backbone, trainer settings, twin
config), `f1_latent.py`, `f3_arbiter.py`, `f2_world_policy.py`, `toy.py` (synthetic populations),
`analysis.py` (paired ablation contrasts from the runner's parts). Tests: `tests/population/candidates/`.
Experiment definitions: `experiments/phase3/candidates/`. Results: `reports/phase3/candidates/`.

RESULTS_PLACEHOLDER
