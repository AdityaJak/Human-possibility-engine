"""Quick functional check of social mechanism v2 (dev seed, small world). Exploratory only.

python experiments/phase2/social/quick_check.py [n_agents] [years] [seed]
"""

from __future__ import annotations

import json
import sys

import hpe.development.social  # noqa: F401  (registers developmental_social)
from hpe.contracts.registry import get_factory
from hpe.contracts.schemas import LogConfig, SimulationConfig
from hpe.world import World


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    years = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    assert 1000 <= seed <= 2049, "dev/tune seeds only"
    params: dict[str, object] = {"social_v2": {}}
    cfg = SimulationConfig(
        seed=seed,
        n_agents=n,
        years=years,
        policy_class="developmental",
        policy_params=params,
        log=LogConfig(
            frames_every_ticks=0,
            decision_sample_rate=0.0,
            log_learning_episodes=False,
            checkpoint_every_years=None,
        ),
    )
    w = World(cfg, get_factory("developmental_social", params))
    w.run()
    st = w._sv2_state.stats
    friends = sum(1 for a in w.agents() for e in w.edges(a.agent_id) if e.kind == "friend") / 2
    out = {
        "stats": {k: round(v, 3) for k, v in st.items()},
        "friend_ties": friends,
        "mean_info_gain": round(st["info_gain_sum"] / max(1.0, st["info_gain_n"]), 4),
        "trust_honest": w._sv2_state.trust_honest,
        "trust_misinformer": w._sv2_state.trust_misinformer,
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
