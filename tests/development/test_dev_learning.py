"""Learning is real: held-out skill improves, parameters move, and only when learning is enabled.

Task seeds used here (202, 303) differ from the seeds used for the light hyperparameter
sweep during development (7, 21).
"""

import numpy as np
import pytest

import hpe.development as dev
from hpe.contracts.conformance import make_init
from hpe.contracts.runtime import Demonstration, Feedback
from hpe.core import stream
from hpe.development.diagnostics import episode_cost_us, learning_curve
from hpe.tasks import TRAINING_FAMILIES, SymbolWorld, assess, practice

SEEDS = range(5)


@pytest.fixture(scope="module")
def world():
    return SymbolWorld(seed=202)


@pytest.mark.parametrize("family", TRAINING_FAMILIES)
def test_full_feedback_improves_heldout_accuracy(world, family):
    gains, finals = [], []
    for seed in SEEDS:
        pol = dev.build_factory()(make_init(world, seed=seed))
        curve = learning_curve(pol, world, family, checkpoints=(0, 300), mode="full", seed=seed)
        gains.append(curve.gain)
        finals.append(curve.accuracy[-1])
    print(
        f"{family}: held-out gain {np.mean(gains):.3f} (final {np.mean(finals):.3f}) over {len(gains)} seeds"
    )
    assert np.mean(gains) > 0.15
    chance = 1 / {"sort": 3, "compare": 2, "signal": 2}[family]
    assert np.mean(finals) > chance + 0.15


def test_bandit_only_feedback_improves(world):
    gains = []
    for seed in SEEDS:
        pol = dev.build_factory()(make_init(world, seed=seed))
        curve = learning_curve(pol, world, "compare", checkpoints=(0, 300), mode="bandit", seed=seed)
        gains.append(curve.gain)
    print(f"bandit-only compare gain {np.mean(gains):.3f}")
    assert np.mean(gains) > 0.15


def test_parameters_change_as_a_consequence_of_experience(world):
    pol = dev.build_factory()(make_init(world, seed=1))
    s0 = pol.parameter_summary()
    assert s0["families"]["sort"]["head_norm"] == 0.0 and s0["total_updates"] == 0
    stats = practice(pol, world, "sort", 50, stream(1, "p"), agent_id="a", tick=0, age_years=15, mode="full")
    s1 = pol.parameter_summary()
    assert s1["param_hash"] != s0["param_hash"]
    assert s1["families"]["sort"]["head_norm"] > 0.0
    assert s1["families"]["sort"]["n_updates"] == 50
    assert s1["encoder_drift_from_init"][0] > 0.0
    assert stats.delta_norm_sum > 0.0 and stats.n_updates == 50
    # untouched family heads have not moved
    assert s1["families"]["signal"]["head_norm"] == 0.0


def test_no_parameter_change_when_learning_disabled(world):
    pol = dev.build_factory({"use_learning": False})(make_init(world, seed=1))
    h0 = _learnable_hash(pol)
    acc0 = assess(pol, world, "sort", 20).accuracy
    rng = stream(2, "p")
    for mode in ("bandit", "full"):
        practice(pol, world, "sort", 50, rng, agent_id="a", tick=0, age_years=20, mode=mode)  # type: ignore[arg-type]
    item = world.sample_item("sort", rng)
    obs = world.observe(item, "a", 0, 20.0)
    fb = Feedback(mode="full", reward=1.0, correct_action=item.correct_action, source="peer")
    assert pol.learn_from_demonstration(Demonstration(obs, item.correct_action, fb, "t", 1.0)) is None
    for i in range(24):
        pol.tick_lifecycle(20 + i / 12, 1 / 12)
    assert _learnable_hash(pol) == h0
    assert assess(pol, world, "sort", 20).accuracy == acc0
    assert pol.parameter_summary()["total_updates"] == 0 and pol.parameter_summary()["memory_items"] == 0


def _learnable_hash(pol):
    """State hash excluding bookkeeping that is not learned (current age, compute counters)."""
    sd = dict(pol.state_dict())
    sd.pop("compute", None)
    sd.pop("age", None)
    from hpe.core import stable_hash

    return stable_hash(sd)


def test_no_task_learning_without_feedback(world):
    pol = dev.build_factory({"use_feedback": False})(make_init(world, seed=1))
    h0 = pol.parameter_summary()["param_hash"]
    practice(pol, world, "compare", 100, stream(3, "p"), agent_id="a", tick=0, age_years=20, mode="full")
    assert pol.parameter_summary()["param_hash"] == h0
    full = dev.build_factory()(make_init(world, seed=1))
    practice(full, world, "compare", 100, stream(3, "p"), agent_id="a", tick=0, age_years=20, mode="none")
    assert full.parameter_summary()["param_hash"] == h0  # feedback mode "none" never updates task params


def test_unpractised_family_scores_exactly_chance(world):
    pol = dev.build_factory()(make_init(world, seed=4))
    for fam in TRAINING_FAMILIES:
        practice(pol, world, fam, 100, stream(4, fam), agent_id="a", tick=0, age_years=20, mode="full")
    r = assess(pol, world, "barter", 20)
    assert r.accuracy == pytest.approx(0.5) and r.expected_accuracy == pytest.approx(0.5)


def test_self_estimate_is_separate_from_measured_skill(world):
    pol = dev.build_factory()(make_init(world, seed=5))
    est0 = pol.self_estimates()["signal"]
    assert est0["mean"] == pytest.approx(0.5) and est0["n_feedback"] == 0
    practice(pol, world, "signal", 200, stream(5, "p"), agent_id="a", tick=0, age_years=20, mode="full")
    est = pol.self_estimates()["signal"]
    measured = assess(pol, world, "signal", 20).accuracy
    assert est["n_feedback"] == 200 and 0.0 < est["sd"] < est0["sd"]
    assert est["mean"] > 0.6 and measured > 0.6
    print(f"signal self-estimate {est['mean']:.3f} vs measured held-out {measured:.3f}")


def test_memory_and_working_memory_gate(world):
    young = dev.build_factory()(make_init(world, seed=6))
    adult = dev.build_factory()(make_init(world, seed=6))
    practice(young, world, "sort", 300, stream(6, "p"), agent_id="a", tick=0, age_years=1.0, mode="full")
    practice(adult, world, "sort", 300, stream(6, "p"), agent_id="a", tick=0, age_years=20.0, mode="full")
    assert young.parameter_summary()["memory_items"] < adult.parameter_summary()["memory_items"] <= 256
    off = dev.build_factory({"use_memory": False})(make_init(world, seed=6))
    practice(off, world, "sort", 100, stream(6, "p"), agent_id="a", tick=0, age_years=20.0, mode="full")
    assert off.parameter_summary()["memory_items"] == 0


@pytest.mark.parametrize("family", ["sort", "signal", "barter"])
def test_batch_matches_single_probabilities(world, family):
    pol = dev.build_factory()(make_init(world, seed=7))
    practice(pol, world, family, 60, stream(7, "p"), agent_id="a", tick=0, age_years=20, mode="full")
    items = world.eval_set(family)[:12]
    ctx = np.stack([world.perceive(it.context, 9.0) for it in items])
    pb = pol.action_probabilities_batch(family, "task", ctx, items[0].n_actions, 9.0)
    for i, it in enumerate(items):
        assert np.allclose(pb[i], pol.action_probabilities(world.observe(it, "a", 0, 9.0)), atol=1e-12)


def test_explanations(world):
    pol = dev.build_factory()(make_init(world, seed=8))
    practice(pol, world, "sort", 50, stream(8, "p"), agent_id="a", tick=0, age_years=20, mode="full")
    obs = world.observe(world.sample_item("sort", stream(8, "e")), "a", 0, 20.0)
    obs.meta["explain"] = True
    ex = pol.decide(obs, stream(8, "d"), explore=False).explanation
    assert set(ex) >= {"scores", "probs", "top_latent", "mastery_mean", "temperature"}
    assert len(ex["scores"]) == 3


def test_episode_cost_is_reported(world):
    pol = dev.build_factory()(make_init(world, seed=9))
    us = episode_cost_us(pol, world, "sort", n=500)
    print(f"decide+learn cost: {us:.1f} us/episode")
    assert us < 400  # loose regression guard only; real numbers are reported, not asserted
