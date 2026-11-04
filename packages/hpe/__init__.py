"""Human Possibility Engine.

Synthetic research prototype; not validated for predicting human futures.

Package map (each sub-package has exactly one owning lane, see AGENTS.md):

- ``hpe.contracts``   frozen, versioned shared schemas and protocols (orchestrator)
- ``hpe.core``        deterministic RNG streams, hashing, small utilities (orchestrator)
- ``hpe.tasks``       novel-symbol task families, held-out splits, assessment (orchestrator)
- ``hpe.world``       society simulation kernel (sim/world lane)
- ``hpe.development`` developmental learner policy (research/development lane)
- ``hpe.evaluation``  baselines, statistics, experiment runner (science/evaluation lane)
- ``hpe.storage``     persistence, event logs, checkpoints (platform/backend lane)
- ``hpe.futures``     Future Self scenario engine and evidence registry (platform/backend lane)
"""

__version__ = "0.1.0"
