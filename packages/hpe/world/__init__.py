"""Society simulation kernel (sim/world lane).

``World`` implements ``hpe.contracts.protocols.WorldRunner``: a deterministic, event-driven,
checkpointable artificial society whose agents' skills change only through learning
episodes handed to injected policies, and are measured only by held-out assessment.
See ``ASSUMPTIONS.md`` in this package for every modelling approximation.
"""

from hpe.world.engine import SNAPSHOT_FORMAT, WORLD_VERSION, World, stage_for_age

__all__ = ["SNAPSHOT_FORMAT", "WORLD_VERSION", "World", "stage_for_age"]
