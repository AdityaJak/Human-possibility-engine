"""Phase II task families (v2). See ``world.py`` for the families and their held-out discipline."""

from hpe.tasks.v2.delayed import practice_delayed
from hpe.tasks.v2.world import V2_FAMILIES, V2_SPECS, V2Config, V2World

__all__ = ["V2_FAMILIES", "V2_SPECS", "V2Config", "V2World", "practice_delayed"]
