"""Gate B family 3 (`composition`): determinism, held-out discipline, exact eval balance, specialist
divergence, difficulty proxy, feedback noise, and a measured linear-probe ceiling (recorded by the
lane's script; here only sanity-bounded)."""

import numpy as np
import pytest

from hpe.contracts.conformance import make_init
from hpe.contracts.runtime import RandomPolicy
from hpe.core import stream
from hpe.tasks import assess
from hpe.tasks.v2.composition import (
    COMPOSITION_FAMILIES,
    OPS,
    CompositionConfig,
    CompositionWorld,
    compose,
)

DEV_SEEDS = (1000, 1001, 1002, 1003)


def test_deterministic_and_clean():
    a, b = CompositionWorld(1000), CompositionWorld(1000)
    assert a.fingerprint() == b.fingerprint() != CompositionWorld(1001).fingerprint()
    assert a.op in OPS and a.leakage_report()["clean"]
    assert set(a.train_ids).isdisjoint(set(a.holdout_ids))
    assert a.describe()["shared_structure"] is True and a.taught == a.composed


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_pools_hold_both_labels_and_eval_sets_exactly_balanced(seed):
    w = CompositionWorld(seed)
    for f in ("primitive_a", "primitive_b", "composition"):
        for split in ("train", "holdout"):
            ids = w.pool(split)
            counts = np.bincount(w.labels[f][ids], minlength=2)
            assert counts.min() >= w.config.min_per_label, (f, split, counts)
    for f in COMPOSITION_FAMILIES:
        items = w.eval_set(f)
        counts = np.bincount([i.correct_action for i in items], minlength=2)
        assert counts.tolist() == [30, 30], (f, counts)
        again = CompositionWorld(seed).eval_set(f)
        assert [i.symbol_ids for i in items] == [i.symbol_ids for i in again]


@pytest.mark.parametrize("fam", COMPOSITION_FAMILIES)
def test_holdout_only_eval_and_train_only_practice(fam):
    w = CompositionWorld(1002)
    hold, train = set(int(i) for i in w.holdout_ids), set(int(i) for i in w.train_ids)
    assert all(set(i.symbol_ids) <= hold for i in w.eval_set(fam))
    rng = stream(0, "t")
    for _ in range(60):
        assert set(w.sample_item(fam, rng).symbol_ids) <= train
    leak = w.leakage_counts()
    assert leak["practice_items"] == 60 and leak["holdout_symbols_in_practice"] == 0


def test_labels_follow_the_composition_rule_and_pairs_have_one_positive():
    w = CompositionWorld(1003)
    x = w.symbols.astype(np.float64)
    expect = compose(w.op, w.composed.r1(x), w.composed.r2(x))
    assert np.array_equal(w.labels["composition"], expect)
    assert np.array_equal(w.labels["primitive_a"], w.taught.r1(x))
    assert np.array_equal(w.labels["primitive_b"], w.taught.r2(x))
    for it in w.eval_set("composition_pair"):
        y = w.labels["composition"][list(it.symbol_ids)]
        assert int(y.sum()) == 1 and it.correct_action == int(np.argmax(y)) and it.context.shape == (2, 12)


def test_specialist_control_differs_only_in_the_composed_primitives():
    shared = CompositionWorld(1000)
    ctrl = CompositionWorld(1000, CompositionConfig(shared_structure=False))
    assert shared.taught == ctrl.taught and shared.composed != ctrl.composed
    assert np.array_equal(shared.labels["primitive_a"], ctrl.labels["primitive_a"])
    assert np.array_equal(shared.symbols, ctrl.symbols)
    assert not np.array_equal(shared.labels["composition"], ctrl.labels["composition"])


def test_difficulty_proxy_and_curriculum_ordering():
    w = CompositionWorld(1001)
    rng = stream(3, "d")
    easy = np.mean([w.sample_item("composition", rng, difficulty=0.0).margin for _ in range(40)])
    hard = np.mean([w.sample_item("composition", rng, difficulty=1.0).margin for _ in range(40)])
    assert easy >= hard
    assert 0 <= w.margins.min() <= w.margins.max() <= w.config.n_features


def test_feedback_noise_only_corrupts_feedback():
    w = CompositionWorld(1002)
    rng = stream(0, "fb")
    item = w.sample_item("composition", rng)
    fbs = [w.feedback(item, item.correct_action, "bandit", rng, noise=0.5) for _ in range(300)]
    assert 0.35 < np.mean([f.reward for f in fbs]) < 0.65
    assert w.score("composition", item.symbol_ids)[0] == item.correct_action
    full = w.feedback(item, 0, "full", rng, noise=0.0)
    assert full.correct_action == item.correct_action and full.rewards_all is not None


def test_random_policy_at_chance_and_majority_baseline_is_half():
    w = CompositionWorld(1000)
    pol = RandomPolicy(make_init(w))  # type: ignore[arg-type]
    for f in COMPOSITION_FAMILIES:
        r = assess(pol, w, f, 20.0)
        assert abs(r.accuracy - 0.5) < 1e-9


def test_parity_component_defeats_a_linear_probe_on_composition():
    """Least-squares linear probe on raw features: fit on train symbols, score on holdout.
    The pre-registered expectation is <= ~0.75; this test only guards against the family being
    linearly solvable (the measured ceiling is recorded by the lane's script)."""
    accs = []
    for seed in DEV_SEEDS:
        w = CompositionWorld(seed)
        X = np.concatenate([w.symbols, np.ones((len(w.symbols), 1), dtype=np.float32)], axis=1).astype(
            np.float64
        )
        y = w.labels["composition"].astype(np.float64) * 2 - 1
        tr, ho = w.train_ids, w.holdout_ids
        wts, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred = np.sign(X[ho] @ wts)
        # balanced accuracy on the holdout symbols (classes are unbalanced at the symbol level)
        acc = np.mean([(pred[y[ho] == c] == c).mean() for c in (-1.0, 1.0)])
        accs.append(acc)
    assert max(accs) < 0.9, accs
