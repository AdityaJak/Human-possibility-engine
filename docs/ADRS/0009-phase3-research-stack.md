# ADR-0009: Phase III research stack — optional PyTorch extra, `hpe.population`, Phase III partitions
Status: accepted (2026-11-04, Chief Architect / Research Director).

**Context.** Phase III tests population-trained models with individual adaptation against strong
sequence-model and meta-learning baselines (GRU/LSTM at several capacities, a causal transformer,
per-person fine-tuning, Bayesian head updates, first-order meta-learning). Phase II's numpy GRU uses
hand-written backpropagation; writing and verifying second-order meta-learning, LSTMs and attention
by hand would cost more than the science it serves and would be a new source of bugs that could
favour one model over another.

**Decision.**
1. Add an optional extra `research = ["torch>=2.5,<2.6"]` installed from the CPU wheel index
   (`--index-url https://download.pytorch.org/whl/cpu`). Core packages (contracts, world, futures,
   the API and the in-browser Pyodide engine) do not import torch; `hpe.population` imports it lazily
   and its tests skip when it is absent. CI installs the extra in a dedicated job.
2. New package `hpe.population`: `core/` (orchestrator: encoding v3, data access with the test lock,
   the `SequencePredictor` interface, evaluation, a causality check every model must pass),
   `baselines/` (evaluation team), `candidates/` (design team), `continual/`, `affect/`.
3. Phase III partitions for studies never used before (`reports/phase3/partitions/psych101_p3.json`,
   seed 20261106, test digest `eb14a4d9…`); the Phase II partition file stays authoritative for the
   families it covers. Test access requires a freeze record in `reports/phase3/freeze/`.
4. Determinism: every torch job sets `torch.manual_seed` from an `hpe.core.rng` named stream,
   `torch.use_deterministic_algorithms(True)` and an explicit thread count; results are reported
   with the thread count.

**Consequences.** Phase II numpy results stay as committed. The torch re-implementation of the
population GRU is checked against the Phase II `ref_gru` on the tune partition before it is used as a
comparator, and the Phase II `ref_gru` itself remains in the baseline set.
