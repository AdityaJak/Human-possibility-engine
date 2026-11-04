"""developmental_social: Phase I learner + Phase II social options (defaults identical to Phase I)."""

from __future__ import annotations

import json

import numpy as np

from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.contracts.registry import get_factory
from hpe.contracts.runtime import Demonstration
from hpe.core import stream
from hpe.development.social import DevelopmentalSocialLearner, build_social_factory, split_params
from hpe.tasks import SymbolWorld, policy_state_hash


def test_split_params_strips_world_and_social_keys():
    dev, social = split_params({"use_memory": False, "social_v2": {"p_known": 0.5}, "teaching_weight": 2.0})
    assert dev == {"use_memory": False}
    assert social == {"social_v2": {"p_known": 0.5}, "teaching_weight": 2.0}


def test_registered_and_conformant_with_learning():
    factory = get_factory("developmental_social", {"social_v2": {}, "teaching_weight": 1.5})
    run_policy_conformance(factory, check_learning=True)


def test_default_is_identical_to_phase1_learner():
    world = SymbolWorld(seed=1001)
    a = get_factory("developmental")(make_init(world, seed=3))
    b = build_social_factory({"social_v2": {}})(make_init(world, seed=3))
    assert isinstance(b, DevelopmentalSocialLearner) and b.teaching_weight == 1.0
    rng_a, rng_b = stream(5, "x"), stream(5, "x")
    item = world.sample_item("sort", stream(1, "i"))
    obs = world.observe(item, "a0000", 0, 12.0)
    fb = world.feedback(item, item.correct_action, "full", stream(2, "f"))
    for pol, rng in ((a, rng_a), (b, rng_b)):
        pol.learn_from_demonstration(Demonstration(obs, item.correct_action, fb, "a0001", 0.8))
        pol.decide(obs, rng)
    sa, sb = a.state_dict(), b.state_dict()
    sb.pop("teaching_weight")
    sb.pop("demo_counts")
    assert sa.pop("policy_class") == "developmental" and sb.pop("policy_class") == "developmental_social"
    assert json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)


def test_teaching_weight_scales_instructional_demonstrations_only():
    world = SymbolWorld(seed=1002)
    item = world.sample_item("compare", stream(1, "i"))
    obs = world.observe(item, "a0000", 0, 12.0)
    full = world.feedback(item, item.correct_action, "full", stream(2, "f"))
    bandit = world.feedback(item, item.correct_action, "bandit", stream(2, "f"))
    w1 = build_social_factory({"teaching_weight": 1.0})(make_init(world, seed=4))
    w2 = build_social_factory({"teaching_weight": 2.0})(make_init(world, seed=4))
    u1 = w1.learn_from_demonstration(Demonstration(obs, item.correct_action, full, "t", 0.4))
    u2 = w2.learn_from_demonstration(Demonstration(obs, item.correct_action, full, "t", 0.4))
    assert u1 is not None and u2 is not None and u2.param_delta_norm > u1.param_delta_norm
    # outcome-only (bandit) demonstrations are not scaled: compare on fresh, identical policies
    f1 = build_social_factory({"teaching_weight": 1.0})(make_init(world, seed=4))
    f2 = build_social_factory({"teaching_weight": 2.0})(make_init(world, seed=4))
    b1 = f1.learn_from_demonstration(Demonstration(obs, item.correct_action, bandit, "t", 0.4))
    b2 = f2.learn_from_demonstration(Demonstration(obs, item.correct_action, bandit, "t", 0.4))
    assert b1 is not None and b2 is not None and np.isclose(b1.param_delta_norm, b2.param_delta_norm)
    assert w2.demo_counts["full"] == 1 and f2.demo_counts["bandit"] == 1


def test_state_round_trip_includes_social_fields():
    world = SymbolWorld(seed=1003)
    pol = build_social_factory({"teaching_weight": 1.7})(make_init(world, seed=6))
    item = world.sample_item("signal", stream(1, "i"))
    obs = world.observe(item, "a0000", 0, 12.0)
    pol.learn_from_demonstration(
        Demonstration(
            obs,
            item.correct_action,
            world.feedback(item, item.correct_action, "full", stream(2, "f")),
            "t",
            0.9,
        )
    )
    clone = build_social_factory({})(make_init(world, seed=6))
    clone.load_state_dict(json.loads(json.dumps(pol.state_dict())))
    assert clone.teaching_weight == 1.7 and clone.demo_counts == pol.demo_counts
    assert policy_state_hash(clone) == policy_state_hash(pol)
