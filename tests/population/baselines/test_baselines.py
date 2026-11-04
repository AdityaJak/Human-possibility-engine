"""Comparator pool: every member fits on a tiny synthetic population, is causal (adapt and frozen), deterministic,
normalised, and supports the frozen / donor variants. The GRU adapters' frozen variant equals ``pop_gru``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, GenericTrial
from hpe.population.core.adaptation import AdaptiveBase, overrides_harness
from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import EncodedSeq, encode, input_dim
from hpe.population.core.interface import get_model

K_MAX = 4


def _toy(n: int, seed: int, k: int = 3) -> EncodedSeq:
    """A learner-like synthetic participant: sticky choices biased toward the best of k options."""
    rng = np.random.default_rng(seed)
    p = Participant("toy/exp1.csv", str(seed), "toy", k)
    best = int(rng.integers(0, k))
    c = 0
    for t in range(n):
        if rng.uniform() < 0.3:
            c = best if rng.uniform() < 0.6 else int(rng.integers(0, k))
        mode = "full" if t % 4 == 0 else ("none" if t % 7 == 0 else "bandit")
        ra = tuple(float(v) for v in rng.uniform(0, 1, k)) if mode == "full" else None
        feats = (1.0 if t % 2 else -1.0,) + (0.0,) * (N_FEATURES - 1)
        r = float(rng.uniform() < (0.8 if c == best else 0.3))
        p.trials.append(GenericTrial(c, k, feats, mode, r, None, ra, forced=t < 2, block=t // 20))
    s = encode(p, "toy", K_MAX)
    s.key = f"toy#{seed}"
    return s


@pytest.fixture(scope="module")
def pop() -> list[EncodedSeq]:
    return [_toy(40, i) for i in range(12)]


def _fit(name: str, cfg: dict[str, Any], seqs: list[EncodedSeq]):  # type: ignore[no-untyped-def]
    m = get_model(name).factory(cfg, K_MAX, input_dim(K_MAX))
    m.fit(seqs, seed=7, threads=1, budget_s=60)
    return m


def _check(name: str, cfg: dict[str, Any], seqs: list[EncodedSeq]) -> None:
    m1, m2 = _fit(name, cfg, seqs), _fit(name, cfg, seqs)
    s = seqs[0]
    a, b = m1.predict_logprobs(s), m2.predict_logprobs(s)
    assert a.shape == (s.T, K_MAX)
    assert np.array_equal(a, b), f"{name}: not deterministic"
    assert np.allclose(np.exp(a[:, :3]).sum(axis=1), 1.0, atol=1e-5) and np.isneginf(a[:, 3]).all()
    assert check_causal(m1, s) == [], f"{name}: causality violation"
    fr = m1.predict_logprobs(s, mode="frozen")
    dn = m1.predict_logprobs(s, donor=seqs[1])
    assert fr.shape == dn.shape == a.shape and np.isfinite(fr[:, :3]).all() and np.isfinite(dn[:, :3]).all()
    d = m1.describe()
    assert d["population_params"] == m1.param_count() >= 1
    assert np.array_equal(m1.predict_logprobs(s), a), f"{name}: predicting mutated the model"
    if getattr(m1, "adaptive", False):
        assert not np.array_equal(a, fr), f"{name}: adaptation had no effect"
        assert isinstance(m1, AdaptiveBase) and not overrides_harness(m1), (
            f"{name}: must use the core harness"
        )
        # rows before the first update use the prior state in every mode
        m = m1.update_every
        assert np.array_equal(a[:m], fr[:m]) and np.array_equal(dn[:m], fr[:m])


torch = pytest.importorskip("torch")
BB = {"hidden": 6, "lr": 0.01, "dropout": 0.0, "max_epochs": 3, "patience": 10, "batch_size": 8}


@pytest.mark.parametrize(
    "name,cfg",
    [
        ("pop_tf", {"window": 8, "d_model": 8, "heads": 2, "max_epochs": 2}),
        ("ft_head", {"backbone": BB, "backbone_seed": 3, "m": 5, "n": 5, "lr": 0.05}),
        ("ft_full", {"backbone": BB, "backbone_seed": 3, "m": 10, "n": 2, "lr": 0.01, "l2": 1.0}),
        (
            "bayes_head",
            {"backbone": BB, "backbone_seed": 3, "m": 5, "prior_var": 1.0, "predictive": "laplace"},
        ),
        ("bayes_head", {"backbone": BB, "backbone_seed": 3, "m": 5, "prior_var": "eb", "predictive": "map"}),
        ("exemplar_mix", {"backbone": BB, "backbone_seed": 3, "m": 3, "w": 0.3, "bw": 1.0, "decay": 0.98}),
        (
            "cavia_gru",
            {
                "backbone": BB,
                "backbone_seed": 3,
                "m": 10,
                "c_dim": 2,
                "inner_lr": 1.0,
                "n_inner": 2,
                "max_epochs": 2,
            },
        ),
        (
            "reptile_gru",
            {
                "backbone": BB,
                "backbone_seed": 3,
                "m": 10,
                "n": 2,
                "lr": 0.01,
                "meta_iters": 3,
                "meta_batch": 2,
            },
        ),
        ("hb_cog", {"c": 0.1, "tau": 1.0, "state_features": [0]}),
        ("ref_gru_p2", {"hidden": 4, "lr": 0.01, "max_epochs": 2, "patience": 2}),
    ],
)
def test_pool_member(name: str, cfg: dict[str, Any], pop: list[EncodedSeq]) -> None:
    _check(name, cfg, pop)


def test_adapter_frozen_variant_equals_pop_gru(pop: list[EncodedSeq]) -> None:
    cfg = dict(BB)
    g = get_model("pop_gru").factory(cfg, K_MAX, input_dim(K_MAX))
    g.fit(pop, seed=11, threads=1, budget_s=60)
    a = _fit("ft_head", {"backbone": cfg, "backbone_seed": 11, "m": 5, "n": 3, "lr": 0.01}, pop)
    for s in pop[:3]:
        assert np.array_equal(g.predict_logprobs(s), a.predict_logprobs(s, mode="frozen"))


def test_hb_cog_option_features_and_reset(pop: list[EncodedSeq]) -> None:
    m = _fit("hb_cog", {"c": 1.0, "option_features": [[0], [1], [2]], "reset_on_block": True}, pop)
    assert check_causal(m, pop[2]) == []


def test_ref_gru_p2_inputs_rebuild_phase2_layout() -> None:
    from hpe.population.baselines.ref_gru_p2 import phase2_inputs

    s = _toy(12, 3)
    xs = phase2_inputs(s, 3)
    assert xs.shape == (12, N_FEATURES + 2 * 3 + 2)
    assert not xs[0, N_FEATURES:].any()  # nothing before the first choice
    # step 1 follows a full-feedback step (t=0): previous-correct one-hot and feedback flag are set
    assert xs[1, N_FEATURES + 3 + 1 : N_FEATURES + 2 * 3 + 1].sum() == 1.0 and xs[1, -1] == 1.0


def test_hb_cog_model_based_hybrid_on_a_two_stage_toy() -> None:
    from hpe.population.families.two_step import parse_two_step

    from ..families.test_families import TWO_STAGE

    p = parse_two_step(TWO_STAGE * 1, "kool2017cost/exp2.csv", "1")
    seqs = []
    for i in range(4):
        s = encode(p, "two_step", K_MAX)
        s.key = f"two#{i}"
        seqs.append(s)
    m = _fit("hb_cog", {"c": 1.0, "state_features": [0, 2, 3], "mb": True}, seqs)
    assert m.theta.shape == (1800, 6)
    assert check_causal(m, seqs[0]) == []
