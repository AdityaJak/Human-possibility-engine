"""Exploratory pilot (dev seeds): how data-hungry is the v2 `conjunction` family for each learner?"""

import time

import hpe.development as dev
import hpe.evaluation.baselines  # noqa: F401
from hpe.contracts.conformance import make_init
from hpe.contracts.registry import get_factory
from hpe.core import stream
from hpe.tasks import assess, practice
from hpe.tasks.v2 import V2World

for seed in (1004, 1011):
    w = V2World(seed=seed)
    init = make_init(w, seed=seed)  # type: ignore[arg-type]
    init.families = w.families_meta()
    runs = [
        ("exemplar_only", dev.build_hybrid_factory({"use_parametric": False})),
        ("exemplar_only_h3", dev.build_hybrid_factory({"use_parametric": False, "kernel_width": 3.0})),
        ("hybrid", get_factory("hybrid_exemplar")),
        ("hybrid_latent", dev.build_hybrid_factory({"exemplar_space": "latent"})),
        ("plain_rl", get_factory("plain_rl")),
        ("memory_only", get_factory("memory_only")),
    ]
    for name, fac in runs:
        pol = fac(init)
        rng = stream(seed, "pilot", name)
        t = time.perf_counter()
        accs = []
        for n in (400, 400, 400):
            practice(pol, w, "conjunction", n, rng, agent_id="a", tick=0, age_years=20.0, mode="full")
            accs.append(round(assess(pol, w, "conjunction", 20.0).accuracy, 3))
        print(f"seed {seed} {name:18s} acc@400/800/1200 = {accs}  ({time.perf_counter() - t:.1f}s)", flush=True)
