"""Smoke check of the lane's new pieces (not a result): composition world, age_script_only, lab options."""

import time

import numpy as np

from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.contracts.registry import get_factory
from hpe.core import stream
from hpe.development.schedules import plasticity
from hpe.evaluation.baselines import scripted_plasticity
from hpe.evaluation.learning_lab import LabContext, resolve_params, run_learning_lab
from hpe.tasks import SymbolWorld
from hpe.tasks.v2.composition import COMPOSITION_FAMILIES, CompositionWorld

for seed in (1000, 1001, 1002):
    w = CompositionWorld(seed)
    print(seed, w.describe(), "shares", {k: round(v, 3) for k, v in w.class_shares().items()}, "clean", w.leakage_report()["clean"])
    for f in COMPOSITION_FAMILIES:
        items = w.eval_set(f)
        rng = stream(0, "s")
        tr = [w.sample_item(f, rng) for _ in range(100)]
        print(
            "   ",
            f,
            "eval",
            np.bincount([i.correct_action for i in items], minlength=2).tolist(),
            "train",
            np.bincount([i.correct_action for i in tr], minlength=2).tolist(),
            "ctx",
            items[0].context.shape,
            "margin",
            items[0].margin,
        )
assert all(abs(scripted_plasticity(a) - plasticity(a)) < 1e-12 for a in np.linspace(0, 100, 401))
print("schedule equality ok")
fac = get_factory("age_script_only")
run_policy_conformance(fac)
print("age_script_only conformant")
sw = SymbolWorld(seed=1000)
pol = fac(make_init(sw))
it = sw.eval_set("sort")[0]
p8 = pol.action_probabilities(sw.observe(it, "a", 0, 8.0, False))
p80 = pol.action_probabilities(sw.observe(it, "a", 0, 80.0, False))
print("age8", np.round(p8, 3), "age80", np.round(p80, 3), "same argmax", p8.argmax() == p80.argmax())
t0 = time.time()
p = resolve_params(
    {
        "n_agents": 2,
        "episodes_per_family": 40,
        "block": 10,
        "checkpoint_every": 20,
        "task_world": "composition",
        "families": ["primitive_a", "primitive_b", "composition"],
        "skip_transfer": True,
        "skip_retention": True,
    }
)
out = run_learning_lab(get_factory("plain_rl"), LabContext("t", "rl", 1000, "r", p))
mets = sorted({o.metric for o in out})
print("composition lab:", len(out), [m for m in mets if "final" in m or "leak" in m][:8], f"{time.time() - t0:.1f}s")
p2 = resolve_params(
    {
        "n_agents": 2,
        "episodes_per_family": 40,
        "block": 10,
        "checkpoint_every": 20,
        "feedback_permutation": True,
        "skip_transfer": True,
        "skip_retention": True,
    }
)
out2 = run_learning_lab(get_factory("plain_rl"), LabContext("t", "rl", 1000, "r", p2))
print("permuted lab finals", np.round([o.value for o in out2 if o.metric == "final_heldout_accuracy"], 3))
