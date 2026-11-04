import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hpe.contracts.conformance import make_init
from hpe.contracts.runtime import RandomPolicy
from hpe.core import stream
from hpe.tasks import (
    ALL_FAMILIES,
    FAMILY_SPECS,
    SymbolWorld,
    TaskWorldConfig,
    assess,
    policy_state_hash,
    transfer_probe,
)


def test_world_is_deterministic():
    a, b = SymbolWorld(seed=11), SymbolWorld(seed=11)
    assert a.fingerprint() == b.fingerprint()
    assert SymbolWorld(seed=12).fingerprint() != a.fingerprint()


def test_no_symbol_leakage_between_train_and_holdout():
    w = SymbolWorld(seed=4)
    rep = w.leakage_report()
    assert rep["clean"] and rep["overlap"] == 0
    assert set(w.train_ids).isdisjoint(set(w.holdout_ids))


def test_eval_sets_use_only_holdout_symbols_and_are_fixed():
    w = SymbolWorld(seed=4)
    hold = set(int(i) for i in w.holdout_ids)
    for fam in ALL_FAMILIES:
        items = w.eval_set(fam)
        assert all(set(it.symbol_ids) <= hold for it in items)
        again = SymbolWorld(seed=4).eval_set(fam)
        assert [it.symbol_ids for it in items] == [it.symbol_ids for it in again]


def test_training_samples_use_only_train_symbols():
    w = SymbolWorld(seed=4)
    train = set(int(i) for i in w.train_ids)
    rng = stream(0, "t")
    for fam in ALL_FAMILIES:
        for _ in range(50):
            assert set(w.sample_item(fam, rng).symbol_ids) <= train


def test_classes_are_not_degenerate():
    w = SymbolWorld(seed=9)
    for fam in ALL_FAMILIES:
        items = w.eval_set(fam, n=200)
        counts = np.bincount([it.correct_action for it in items], minlength=FAMILY_SPECS[fam]["n_actions"])
        assert counts.min() / counts.sum() > 0.15, f"{fam} class balance {counts}"


def test_random_policy_scores_at_chance_on_expected_accuracy():
    w = SymbolWorld(seed=2)
    pol = RandomPolicy(make_init(w))
    for fam in ALL_FAMILIES:
        r = assess(pol, w, fam, 20.0)
        assert abs(r.expected_accuracy - r.chance) < 1e-9
        assert abs(r.accuracy - r.chance) < 1e-9  # ties credited fractionally


def test_age_gate_masks_features_monotonically():
    w = SymbolWorld(seed=1)
    vis = [w.visible_features(a) for a in (0, 2, 5, 8, 10, 30)]
    assert vis == sorted(vis) and vis[0] == w.config.gate_min_features and vis[-1] == w.config.n_features
    assert w.visible_features(0, enable_gate=False) == w.config.n_features
    ctx = w.sample_item("sort", stream(0, "g")).context
    assert np.all(w.perceive(ctx, 0.0)[:, w.config.gate_min_features :] == 0)


def test_feedback_noise_never_changes_truth():
    w = SymbolWorld(seed=1)
    rng = stream(0, "fb")
    item = w.sample_item("sort", rng)
    fbs = [w.feedback(item, item.correct_action, "bandit", rng, noise=0.5) for _ in range(200)]
    assert 0.3 < np.mean([f.reward for f in fbs]) < 0.7
    assert w.score("sort", item.symbol_ids)[0] == item.correct_action


def test_orthogonal_transfer_world_changes_only_transfer_family():
    a = SymbolWorld(seed=3, config=TaskWorldConfig(transfer_mode="shared"))
    b = SymbolWorld(seed=3, config=TaskWorldConfig(transfer_mode="orthogonal"))
    for fam in ("sort", "compare", "signal"):
        assert [i.correct_action for i in a.eval_set(fam)] == [i.correct_action for i in b.eval_set(fam)]
    assert [i.correct_action for i in a.eval_set("barter")] != [
        i.correct_action for i in b.eval_set("barter")
    ]


def test_transfer_probe_does_not_modify_agent():
    w = SymbolWorld(seed=3)
    pol = RandomPolicy(make_init(w))
    h = policy_state_hash(pol)
    tp = transfer_probe(pol, w, "barter", stream(0, "tp"), age_years=20.0, shots=(0, 4))
    assert len(tp.accuracy) == 2 and policy_state_hash(pol) == h


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), age=st.floats(0, 100))
def test_property_assessment_bounds(seed, age):
    w = SymbolWorld(seed=seed)
    r = assess(RandomPolicy(make_init(w)), w, "compare", age)
    assert 0 <= r.ci_low <= r.accuracy <= r.ci_high <= 1
