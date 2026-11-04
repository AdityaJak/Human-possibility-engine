import numpy as np
import pytest

from hpe.contracts import registry
from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.core import stream
from hpe.evaluation.baselines import MemoryOnlyPolicy, PlainRLPolicy, StaticPersonaPolicy
from hpe.tasks import TRAINING_FAMILIES, SymbolWorld, assess, policy_state_hash, practice


@pytest.mark.parametrize("name", ["static_persona", "memory_only", "plain_rl"])
def test_registered_and_conformant(name):
    factory = registry.get_factory(name)
    run_policy_conformance(factory, check_learning=(name in ("plain_rl", "memory_only")))


def test_registry_lists_baselines():
    assert {"static_persona", "memory_only", "plain_rl"} <= set(registry.registered())


def _train(pol, world, n=200, mode="full", seed=0):
    for fam in TRAINING_FAMILIES:
        practice(
            pol, world, fam, n, stream(seed, "train", fam), agent_id="a", tick=0, age_years=20.0, mode=mode
        )


def test_static_persona_never_changes_and_stays_near_chance():
    w = SymbolWorld(seed=21)
    accs = []
    for i in range(12):
        pol = StaticPersonaPolicy(make_init(w, agent_id=f"a{i}", seed=i))
        h = policy_state_hash(pol)
        _train(pol, w, n=60)
        assert policy_state_hash(pol) == h, "static persona must not change with experience"
        accs.append(np.mean([assess(pol, w, f, 20.0).accuracy for f in TRAINING_FAMILIES]))
    chance = np.mean([1 / 3, 1 / 2, 1 / 2])
    assert abs(np.mean(accs) - chance) < 0.1


def test_static_personas_differ_by_trait_seed():
    w = SymbolWorld(seed=21)
    a = StaticPersonaPolicy(make_init(w, agent_id="a1", seed=1))
    b = StaticPersonaPolicy(make_init(w, agent_id="a2", seed=2))
    it = w.eval_set("sort")[0]
    oa, ob = w.observe(it, "a1", 0, 20.0), w.observe(it, "a2", 0, 20.0)
    assert not np.allclose(a.action_probabilities(oa), b.action_probabilities(ob))


def test_memory_only_has_no_parameters_but_improves_through_storage():
    w = SymbolWorld(seed=5)
    pol = MemoryOnlyPolicy(make_init(w))
    before = np.mean([assess(pol, w, f, 20.0).accuracy for f in TRAINING_FAMILIES])
    _train(pol, w, n=200, mode="bandit")
    after = np.mean([assess(pol, w, f, 20.0).accuracy for f in TRAINING_FAMILIES])
    assert after > before + 0.1
    summ = pol.parameter_summary()
    assert summ["learns_parameters"] is False and sum(summ["memory_items"].values()) > 0


def test_memory_capacity_is_bounded():
    w = SymbolWorld(seed=5)
    init = make_init(w)
    init.base_params["memory_capacity"] = 16
    pol = MemoryOnlyPolicy(init)
    _train(pol, w, n=50)
    assert all(v <= 16 for v in pol.parameter_summary()["memory_items"].values())


def test_memory_stores_demonstrations_weighted_by_trust():
    from hpe.contracts.runtime import Demonstration

    w = SymbolWorld(seed=5)
    pol = MemoryOnlyPolicy(make_init(w))
    rng = stream(0, "demo")
    item = w.sample_item("signal", rng)
    obs = w.observe(item, "a", 0, 20.0)
    fb = w.feedback(item, item.correct_action, "full", rng)
    pol.learn_from_demonstration(Demonstration(obs, item.correct_action, fb, "peer", 0.25))
    assert pol.stores["task:signal"].weights[-1] == pytest.approx(0.25)


def test_plain_rl_parameters_change_only_with_feedback():
    w = SymbolWorld(seed=8)
    pol = PlainRLPolicy(make_init(w))
    h0 = policy_state_hash(pol)
    _train(pol, w, n=20, mode="none")
    assert policy_state_hash(pol) == h0, "no feedback -> no parameter change"
    _train(pol, w, n=20, mode="bandit")
    assert policy_state_hash(pol) != h0
    assert pol.parameter_summary()["weight_norms"]["task:sort"] > 0


def test_plain_rl_learns_held_out_and_does_not_share_across_families():
    w = SymbolWorld(seed=8)
    pol = PlainRLPolicy(make_init(w))
    _train(pol, w, n=300, mode="bandit")
    acc = np.mean([assess(pol, w, f, 20.0).accuracy for f in TRAINING_FAMILIES])
    assert acc > 0.65
    # independent specialists: an unpractised family is still exactly at chance
    assert assess(pol, w, "barter", 20.0).accuracy == pytest.approx(0.5)


def test_batch_matches_single_after_training():
    w = SymbolWorld(seed=13)
    for cls in (StaticPersonaPolicy, MemoryOnlyPolicy, PlainRLPolicy):
        pol = cls(make_init(w))
        _train(pol, w, n=40, mode="bandit")
        for fam in ("sort", "compare", "signal", "barter"):
            items = w.eval_set(fam)[:10]
            ctx = np.stack([w.perceive(it.context, 20.0) for it in items])
            pb = pol.action_probabilities_batch(fam, "task", ctx, items[0].n_actions, 20.0)
            for i, it in enumerate(items):
                assert np.allclose(pb[i], pol.action_probabilities(w.observe(it, "a", 0, 20.0)), atol=1e-9)
