"""Candidate A — hybrid parametric + exemplar learner: conformance, controls, learning, persistence."""

import json
import time

import numpy as np
import pytest

import hpe.development as dev
from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.contracts.registry import get_factory
from hpe.contracts.runtime import Feedback, Observation
from hpe.core import stream
from hpe.tasks import SymbolWorld, assess, policy_state_hash, practice

CONTROLS = {
    "fixed_gate": {"use_learned_gate": False, "fixed_gate": 0.5},
    "exemplar_only": {"use_parametric": False},
    "parametric_only": {"use_exemplar": False},
    "no_ordinal": {"use_ordinal": False},
    "no_consolidation": {"use_consolidation": False, "use_forgetting": False},
    "latent_space": {"exemplar_space": "latent"},
    "isotropic_kernel": {"use_learned_kernel": False},
}


def test_registered_and_conformant_default():
    run_policy_conformance(get_factory("hybrid_exemplar"), check_learning=True)


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_controls_conformant(name):
    params = CONTROLS[name]
    run_policy_conformance(dev.build_hybrid_factory(params), check_learning=True)


def test_frozen_learner_conformant_and_static():
    f = dev.build_hybrid_factory({"use_learning": False})
    run_policy_conformance(f, check_learning=False)
    w = SymbolWorld(seed=1001)
    pol = f(make_init(w, seed=1001))
    before = policy_state_hash(pol)
    practice(pol, w, "sort", 50, stream(1001, "t"), agent_id="a0000", tick=0, age_years=20.0, mode="full")
    assert policy_state_hash(pol) == before


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        dev.build_hybrid_factory({"use_exemplars": True})


def test_heldout_learning_improves_over_seeds():
    gains = []
    for seed in range(1000, 1005):
        w = SymbolWorld(seed=seed)
        pol = get_factory("hybrid_exemplar")(make_init(w, seed=seed))
        base = assess(pol, w, "sort", 20.0).accuracy
        practice(
            pol, w, "sort", 300, stream(seed, "learn"), agent_id="a0000", tick=0, age_years=20.0, mode="full"
        )
        gains.append(assess(pol, w, "sort", 20.0).accuracy - base)
    assert np.mean(gains) > 0.2, gains


def test_gate_and_kernel_change_only_with_learning():
    w = SymbolWorld(seed=1002)
    pol = get_factory("hybrid_exemplar")(make_init(w, seed=1002))
    g0 = [g.copy() for g in pol.gate_w]
    practice(pol, w, "compare", 120, stream(1002, "l"), agent_id="a0000", tick=0, age_years=20.0, mode="full")
    fi = pol.fam_index["compare"]
    assert not np.allclose(pol.gate_w[fi], g0[fi])
    assert pol.stores[fi].size > 0
    assert not np.allclose(pol.logw[fi], 0.0)


def test_ordinal_family_spreads_mass_to_neighbours():
    """On a declared ordinal family a single labelled exemplar should favour neighbouring levels
    over distant ones (metric output structure)."""
    w = SymbolWorld(seed=1003)
    init = make_init(w, seed=1003)
    init.families = {
        **init.families,
        "scale": {"n_slots": 1, "n_actions": 9, "n_features": 12, "ordinal": True},
    }
    pol = dev.build_hybrid_factory({"use_parametric": False})(init)
    x = np.ones((1, 12), dtype=np.float32)
    obs = Observation("a0000", 0, 20.0, "task", "scale", x, 9)
    ra = np.zeros(9)
    ra[4] = 1.0
    pol.learn(obs, 4, Feedback(mode="full", reward=1.0, correct_action=4, rewards_all=ra))
    p = pol.action_probabilities(obs)
    assert p[4] > p[3] > p[1] and p[4] > p[5] > p[7]
    flat = dev.build_hybrid_factory({"use_parametric": False, "use_ordinal": False})(init)
    flat.learn(obs, 4, Feedback(mode="full", reward=1.0, correct_action=4, rewards_all=ra))
    q = flat.action_probabilities(obs)
    assert np.isclose(q[3], q[7])


def test_state_roundtrip_exact_after_learning():
    w = SymbolWorld(seed=1004)
    f = get_factory("hybrid_exemplar")
    pol = f(make_init(w, seed=1004))
    practice(pol, w, "signal", 80, stream(1004, "l"), agent_id="a0000", tick=0, age_years=20.0, mode="bandit")
    sd = json.loads(json.dumps(pol.state_dict()))
    other = f(make_init(w, seed=1004))
    other.load_state_dict(sd)
    assert policy_state_hash(other) == policy_state_hash(pol)
    for fam in ("sort", "compare", "signal", "barter"):
        a = assess(pol, w, fam, 20.0).expected_accuracy
        b = assess(other, w, fam, 20.0).expected_accuracy
        assert a == b


def test_batch_matches_single_and_is_fast():
    w = SymbolWorld(seed=1005)
    pol = get_factory("hybrid_exemplar")(make_init(w, seed=1005))
    practice(pol, w, "sort", 200, stream(1005, "l"), agent_id="a0000", tick=0, age_years=20.0, mode="full")
    items = w.eval_set("sort")
    ctx = np.stack([w.perceive(it.context, 20.0) for it in items])
    t = time.perf_counter()
    pb = pol.action_probabilities_batch("sort", "task", ctx, 3, 20.0)
    dt = time.perf_counter() - t
    for i, it in enumerate(items[:10]):
        assert np.allclose(pb[i], pol.action_probabilities(w.observe(it, "a0000", 0, 20.0)), atol=1e-9)
    assert dt < 0.5, dt
