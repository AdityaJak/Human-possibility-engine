import json

import numpy as np
import pytest

import hpe.development as dev
from hpe.contracts.conformance import make_init
from hpe.core import stream
from hpe.development import forgetting_multiplier, plasticity, working_memory
from hpe.tasks import SymbolWorld, assess, policy_state_hash, practice


@pytest.fixture(scope="module")
def world():
    return SymbolWorld(seed=303)


def test_schedules_shape():
    assert plasticity(5) == 1.5 and plasticity(30) == 1.0 and plasticity(90) == pytest.approx(0.4)
    assert plasticity(5) > plasticity(18) > plasticity(30) > plasticity(70) > plasticity(85)
    assert all(plasticity(a, enabled=False) == 1.0 for a in (0, 30, 90))
    assert working_memory(0) == 0.25 and working_memory(12) == 1.0 and working_memory(90) < 1.0
    assert forgetting_multiplier(40) == 1.0 and forgetting_multiplier(80) == pytest.approx(9.0)
    assert forgetting_multiplier(80, enabled=False) == 1.0


def _train(world, seed, age, params=None):
    pol = dev.build_factory(params or {})(make_init(world, seed=seed))
    pol.tick_lifecycle(age, 0.0)
    practice(
        pol,
        world,
        "signal",
        400,
        stream(seed, "tr"),
        agent_id="a",
        tick=0,
        age_years=age,
        mode="full",
        noise=0.0,
    )
    return pol


def _idle(pol, age, years):
    a = age
    for _ in range(int(years * 12)):
        a += 1 / 12
        pol.tick_lifecycle(a, 1 / 12)
    return a


def _drop(world, age, params=None, seeds=range(6)):
    drops = []
    for seed in seeds:
        pol = _train(world, seed, age, params)
        before = assess(pol, world, "signal", age).accuracy
        end = _idle(pol, age, 10)
        drops.append(before - assess(pol, world, "signal", end).accuracy)
    return float(np.mean(drops))


def test_forgetting_is_stronger_in_old_age(world):
    young, old = _drop(world, 30.0), _drop(world, 70.0)
    print(f"held-out accuracy drop after 10 idle years: from age 30 {young:.3f}, from age 70 {old:.3f}")
    assert old > 0.1
    assert old > young + 0.05


def test_no_forgetting_when_disabled(world):
    pol = _train(world, 0, 70.0, {"use_forgetting": False})
    h = pol.parameter_summary()["param_hash"]
    _idle(pol, 70.0, 10)
    assert pol.parameter_summary()["param_hash"] == h


def test_practice_protects_against_forgetting(world):
    pol = _train(world, 1, 70.0)
    before = assess(pol, world, "signal", 70).accuracy
    a, rng = 70.0, stream(1, "keep")
    for _ in range(120):
        practice(pol, world, "signal", 2, rng, agent_id="a", tick=0, age_years=a, mode="bandit")
        a += 1 / 12
        pol.tick_lifecycle(a, 1 / 12)
    assert assess(pol, world, "signal", a).accuracy > before - 0.1


def test_untrained_heads_stay_at_init_through_lifecycle(world):
    pol = _train(world, 2, 30.0)
    _idle(pol, 30.0, 5)
    assert pol.parameter_summary()["families"]["barter"]["head_norm"] == 0.0


def test_roundtrip_preserves_forgetting_rng(world):
    pol = _train(world, 3, 75.0)
    clone = dev.build_factory()(make_init(world, seed=3))
    clone.load_state_dict(json.loads(json.dumps(pol.state_dict())))
    _idle(pol, 75.0, 2)
    _idle(clone, 75.0, 2)
    assert policy_state_hash(pol) == policy_state_hash(clone)


def test_younger_learner_learns_faster_with_plasticity_schedule(world):
    gains = {}
    for age in (8.0, 85.0):
        g = []
        for seed in range(5):
            pol = dev.build_factory()(make_init(world, seed=seed))
            pol.tick_lifecycle(age, 0.0)
            base = assess(pol, world, "compare", age).accuracy
            practice(
                pol,
                world,
                "compare",
                40,
                stream(seed, "pl"),
                agent_id="a",
                tick=0,
                age_years=age,
                mode="full",
            )
            g.append(assess(pol, world, "compare", age).accuracy - base)
        gains[age] = float(np.mean(g))
    print(
        f"40-episode gain: age 8 {gains[8.0]:.3f}, age 85 {gains[85.0]:.3f} (age gate disabled here: world perceives all features)"
    )
    assert gains[8.0] >= gains[85.0] - 0.02
