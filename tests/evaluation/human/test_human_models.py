import math
from pathlib import Path

import numpy as np

from hpe.contracts.runtime import random_policy_factory
from hpe.evaluation.human import harness, models, psych101
from hpe.evaluation.human.definition import load_definition
from hpe.evaluation.human.psych101 import IGTTrial, MCPLTrial, Participant

ROOT = Path(__file__).resolve().parents[3]


def _mcpl_participant(n: int = 60, seed: int = 0) -> Participant:
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        c1, c2 = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        truth = min(8, c1 + c2)
        trials.append(MCPLTrial(c1, c2, truth, truth, True))
    return Participant("collsiöö2023MCPL/exp1.csv", str(seed), "mcpl", 9, trials)


def test_encoding_shapes_and_feedback_modes():
    p = _mcpl_participant(5)
    p.trials[-1].correct = None
    steps = models.encode(p)
    assert len(steps) == 5
    obs, _a, fb = steps[0]
    assert obs.context.shape == (1, models.N_FEATURES) and obs.n_actions == 9 and fb.mode == "full"
    assert steps[-1][2].mode == "none"
    g = Participant("steingroever2015data/exp1.csv", "0", "igt", 4, [IGTTrial(1, 100.0, 1250.0)])
    (o2, _, f2) = models.encode(g, scale=500.0)[0]
    assert o2.n_actions == 4 and f2.mode == "bandit" and f2.reward == 0.0
    assert models.igt_reward(0.0, 500.0) == 0.5


def test_mcpl_context_is_informative_and_bounded():
    x = models.mcpl_context(4, 0)
    assert x.min() >= -1 and x.max() <= 1 and x[0, 4] == 1 and x[0, 5] == 1 and x[0, 10] == 1.0


def test_delta_rule_learns_toward_truth():
    p = _mcpl_participant(80)
    f = models.reference_factory("ref_delta_rule", {"alpha": 0.1, "sigma": 1.0}, [], "mcpl")
    pc, top = harness.run_participant(f, models.encode(p), harness.make_init(p.key, 0))
    assert pc[-20:].mean() > pc[:10].mean() + 0.1
    assert top[-20:].mean() > 0.5


def test_instance_model_and_static_refs_produce_valid_probs():
    train = [_mcpl_participant(40, s) for s in range(3)]
    p = _mcpl_participant(30, 9)
    for name, params in (
        ("ref_instance_based", {"h": 1.0, "s": 1.0, "kappa": 0.05}),
        ("ref_bias", {"smoothing": 1.0}),
        ("ref_population_static", {"smoothing": 1.0}),
    ):
        f = models.reference_factory(name, params, train, "mcpl")
        pol = f(harness.make_init(p.key, 0))
        for obs, a, fb in models.encode(p):
            pr = pol.action_probabilities(obs)
            assert pr.shape == (9,) and abs(pr.sum() - 1) < 1e-9 and (pr > 0).all()
            pol.learn(obs, a, fb)


def test_igt_rw_prefers_rewarded_deck():
    trials = [IGTTrial(2, 50.0, 0.0) for _ in range(20)] + [IGTTrial(0, 100.0, 1250.0) for _ in range(5)]
    g = Participant("steingroever2015data/exp1.csv", "0", "igt", 4, trials)
    f = models.reference_factory("ref_delta_rule", {"alpha": 0.3, "beta": 6.0}, [], "igt")
    pol = f(harness.make_init(g.key, 0))
    for obs, a, fb in models.encode(g, scale=500.0):
        pol.learn(obs, a, fb)
    pr = pol.action_probabilities(models.encode(g)[0][0])
    assert pr[2] > pr[0] and pr[2] > pr[1]


def test_random_policy_nll_is_exactly_chance_and_lapse_math():
    p = _mcpl_participant(20)
    pc, _ = harness.run_participant(random_policy_factory, models.encode(p), harness.make_init(p.key, 0))
    assert abs(harness.lapse_nll(pc, 0.0, 9) - math.log(9)) < 1e-12
    assert abs(harness.lapse_nll(np.full(5, 0.9), 1.0, 4) - math.log(4)) < 1e-12
    assert harness.lapse_nll(np.zeros(3), 0.0, 4) < 30  # clipped, finite


def test_split_is_stratified_deterministic_and_disjoint(psych_jsonl):
    parts = psych101.load_participants(psych_jsonl, "mcpl")
    tr, te = harness.split_participants(parts, 0.7, 5)
    tr2, _te2 = harness.split_participants(parts, 0.7, 5)
    assert [p.key for p in tr] == [p.key for p in tr2] and {p.key for p in tr}.isdisjoint({p.key for p in te})
    assert len(tr) + len(te) == len(parts)
    assert sum(p.experiment.endswith("exp1.csv") for p in tr) == 7  # 70% of 10, stratified
    _, te3 = harness.split_participants(parts, 0.7, 6)
    assert {p.key for p in te3} != {p.key for p in te}


def test_unregistered_policy_class_is_reported_not_crashed(psych_jsonl):
    defn = load_definition(ROOT / "experiments/exp07_human_heldout.yaml")
    parts = psych101.load_participants(psych_jsonl, "mcpl")
    tr, te = harness.split_participants(parts, 0.7, 1)
    r = harness.evaluate_model("definitely_not_a_class", "mcpl", tr, te, defn, seed=1)
    assert r.status == "unavailable" and "not registered" in r.note
    r2 = harness.evaluate_model("ref_population_static", "igt", tr, te, defn, seed=1)
    assert r2.status == "unavailable"


def test_fitting_uses_train_only_and_evaluates_each_test_participant(psych_jsonl):
    defn = load_definition(ROOT / "experiments/exp07_human_heldout.yaml")
    defn.grids["ref_delta_rule"]["mcpl"] = {"alpha": [0.03, 0.1], "sigma": [1.0]}
    parts = psych101.load_participants(psych_jsonl, "mcpl")
    tr, te = harness.split_participants(parts, 0.7, 2)
    r = harness.evaluate_model("ref_delta_rule", "mcpl", tr, te, defn, seed=2)
    assert r.status == "ok" and set(r.test_nll) == {p.key for p in te}
    assert r.n_grid == 2 * len(defn.lapse_grid)
    assert r.lapse in defn.lapse_grid and r.params["alpha"] in (0.03, 0.1)
