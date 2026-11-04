"""Dataset adapters for Phase III families (evaluation team). Importing this package registers them."""

from hpe.population.families import horizon_external as horizon_external
from hpe.population.families import restless4 as restless4
from hpe.population.families import spatial_bandit as spatial_bandit
from hpe.population.families import two_step as two_step

try:  # external GBE family (needs the affect package, which imports torch: optional research extra)
    from hpe.population.families import gbe_risk as gbe_risk
except ModuleNotFoundError as _e:
    if not (_e.name or "").startswith("torch"):
        raise
