"""H3 transfer probe: shared representation vs independent specialists.

This test only checks that the measurement runs correctly and leaves agents unmodified. It
prints the measured result and deliberately does NOT assert which arm wins: that is a
scientific outcome for the preregistered experiments in the evaluation lane.
"""

import numpy as np

import hpe.development as dev
from hpe.contracts.conformance import make_init
from hpe.core import stream
from hpe.tasks import (
    TRAINING_FAMILIES,
    SymbolWorld,
    TaskWorldConfig,
    policy_state_hash,
    practice,
    transfer_probe,
)

SHOTS = (0, 4, 16, 64)


def _develop(pol, world, seed):
    rng = stream(seed, "develop", world.seed)
    for _ in range(4):
        for fam in TRAINING_FAMILIES:
            practice(pol, world, fam, 20, rng, agent_id="a", tick=0, age_years=20, mode="bandit", noise=0.05)


def test_transfer_probe_shared_vs_specialist():
    results = {}
    for tmode in ("shared", "orthogonal"):
        for name, params in (("shared_rep", {}), ("specialist", {"use_shared_representation": False})):
            aucs = []
            for seed in range(5):
                world = SymbolWorld(seed=600 + seed, config=TaskWorldConfig(transfer_mode=tmode))
                pol = dev.build_factory(params)(make_init(world, seed=seed))
                _develop(pol, world, seed)
                h = policy_state_hash(pol)
                tp = transfer_probe(pol, world, "barter", stream(seed, "probe"), age_years=20, shots=SHOTS)
                assert policy_state_hash(pol) == h, "probe must run on a clone"
                assert tp.accuracy[0] == 0.5, "untrained barter head must score exactly chance"
                assert all(0.0 <= a <= 1.0 for a in tp.accuracy)
                aucs.append(tp.area_under_curve)
            results[(tmode, name)] = (float(np.mean(aucs)), float(np.std(aucs) / np.sqrt(len(aucs))))
    for (tmode, name), (m, se) in sorted(results.items()):
        print(f"transfer AUC world={tmode:10s} {name:10s} {m:.3f} ± {se:.3f} (5 seeds, shots {SHOTS})")
    assert len(results) == 4
