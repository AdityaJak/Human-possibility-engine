import numpy as np
import pytest

import hpe.development as dev
from hpe.contracts.conformance import make_init
from hpe.contracts.runtime import Demonstration, Feedback
from hpe.core import stream
from hpe.tasks import SymbolWorld, assess, practice


@pytest.fixture(scope="module")
def world():
    return SymbolWorld(seed=404)


def _demos(
    pol, world, n, seed, *, wrong=False, mode="full", trust=0.9, source="teacher", age=20.0, family="signal"
):
    rng = stream(seed, "demos", source)
    for _ in range(n):
        it = world.sample_item(family, rng)
        obs = world.observe(it, "a", 0, age)
        label = 1 - it.correct_action if wrong else it.correct_action
        if mode == "full":
            fb = Feedback(mode="full", reward=1.0, correct_action=label, source="peer")
        else:
            fb = Feedback(mode="none", source="peer")
        pol.learn_from_demonstration(Demonstration(obs, label, fb, source, trust))


def test_demonstrations_improve_heldout_skill(world):
    gains = []
    for seed in range(5):
        pol = dev.build_factory()(make_init(world, seed=seed))
        base = assess(pol, world, "signal", 20).accuracy
        _demos(pol, world, 200, seed)
        gains.append(assess(pol, world, "signal", 20).accuracy - base)
    print(f"demonstration learning gain {np.mean(gains):.3f}")
    assert np.mean(gains) > 0.15


def test_unlabeled_imitation_also_teaches(world):
    gains = []
    for seed in range(5):
        pol = dev.build_factory({"use_feedback": False})(make_init(world, seed=seed))
        base = assess(pol, world, "compare", 20).accuracy
        _demos(pol, world, 300, seed, mode="none", family="compare")
        gains.append(assess(pol, world, "compare", 20).accuracy - base)
    print(f"unlabeled imitation gain (compare) {np.mean(gains):.3f}")
    assert np.mean(gains) > 0.1


def test_social_learning_ablation_ignores_demonstrations(world):
    pol = dev.build_factory({"use_social_learning": False})(make_init(world, seed=0))
    h = pol.parameter_summary()["param_hash"]
    _demos(pol, world, 50, 0)
    assert pol.parameter_summary()["param_hash"] == h


def _misinformation_drop(world, skeptic: bool) -> float:
    drops = []
    for seed in range(5):
        pol = dev.build_factory({"use_skepticism": skeptic})(make_init(world, seed=seed))
        practice(
            pol,
            world,
            "signal",
            400,
            stream(seed, "tr"),
            agent_id="a",
            tick=0,
            age_years=25,
            mode="full",
            noise=0.0,
        )
        before = assess(pol, world, "signal", 25).accuracy
        _demos(pol, world, 150, seed, wrong=True, trust=0.8, source="rumour", age=25.0)
        drops.append(before - assess(pol, world, "signal", 25).accuracy)
    return float(np.mean(drops))


def test_skepticism_limits_misinformation_damage(world):
    with_s, without_s = _misinformation_drop(world, True), _misinformation_drop(world, False)
    print(
        f"accuracy drop after 150 high-trust misinformation demos: skeptical {with_s:.3f}, credulous {without_s:.3f}"
    )
    assert with_s < without_s


def test_source_credibility_table_is_bounded(world):
    pol = dev.build_factory({"max_tracked_sources": 8})(make_init(world, seed=1))
    practice(pol, world, "signal", 200, stream(1, "tr"), agent_id="a", tick=0, age_years=25, mode="full")
    for i in range(20):
        _demos(pol, world, 2, i, source=f"s{i:02d}")
    assert len(pol.source_credibility) == 8
    assert list(pol.source_credibility)[-1] == "s19"
