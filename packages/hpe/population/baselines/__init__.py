"""Phase III comparator pool (evaluation team; ``docs/phase3/ADAPTATION_BENCHMARKS.md``).

Importing this package registers: ``pop_tf`` (causal transformer), the GRU-backbone adapters ``ft_head``,
``ft_full``, ``bayes_head``, ``exemplar_mix``, ``reptile_gru``, ``cavia_gru``, the hierarchical Bayesian cognitive
model ``hb_cog``, the mixture ``gru_hb_mix`` and the Phase II reference ``ref_gru_p2``. ``pop_gru`` / ``pop_lstm`` are registered by the core.
Torch is required for every module except ``hb_cog`` and ``ref_gru_p2``.
"""

from hpe.population.baselines import hb_cog as hb_cog
from hpe.population.baselines import ref_gru_p2 as ref_gru_p2

try:
    from hpe.population.baselines import adapters as adapters
    from hpe.population.baselines import mixture as mixture
    from hpe.population.baselines import tf as tf
except ModuleNotFoundError as e:  # torch (optional research extra) missing
    if not (e.name or "").startswith("torch"):
        raise
