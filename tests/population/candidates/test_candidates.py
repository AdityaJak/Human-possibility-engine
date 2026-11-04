"""Design-team candidates: registration, causality in both modes, determinism, donor control, and the
toy verification (adaptation helps when individuals differ and does not when they do not)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hpe.population.candidates import common  # noqa: E402
from hpe.population.candidates.toy import make_population  # noqa: E402
from hpe.population.core import evaluate  # noqa: E402
from hpe.population.core.conformance import check_causal  # noqa: E402
from hpe.population.core.encoding import input_dim  # noqa: E402
from hpe.population.core.interface import MODELS, get_model  # noqa: E402

CANDIDATES = [
    "f1_latent",
    "f1_trait",
    "f1_context",
    "f1_fixedstep",
    "f1_bolt",
    "f3_arbiter",
    "f3_fixedmix",
    "f3_fixedshare",
    "f3_gruepi",
    "f2_world_policy",
    "f2_delta",
]
SMALL = {"hidden": 8, "lr": 0.01, "max_epochs": 2, "meta_epochs": 1, "m": 8, "batch_size": 8}


def test_all_candidates_registered_as_adaptive_candidates():
    for name in CANDIDATES:
        spec = get_model(name)
        assert spec.owner == "candidates" and spec.kind == "candidate" and spec.adaptive
    assert {n for n, s in MODELS.items() if s.owner == "candidates"} == set(CANDIDATES)


@pytest.mark.parametrize("name", CANDIDATES)
def test_candidate_is_causal_deterministic_and_normalised(name):
    common.clear_backbone_cache()
    train = make_population(12, 30, hetero=True, seed=7, kind="bandit")
    probe = make_population(2, 30, hetero=True, seed=7, offset=500, kind="bandit")
    spec = get_model(name)
    m1 = spec.factory(dict(SMALL), 4, input_dim(4))
    log = m1.fit(train, seed=3, threads=1, budget_s=60)
    assert isinstance(log, dict)
    common.clear_backbone_cache()
    m2 = spec.factory(dict(SMALL), 4, input_dim(4))
    m2.fit(train, seed=3, threads=1, budget_s=60)
    s = probe[0]
    for kw in ({"mode": "adapt"}, {"mode": "frozen"}, {"donor": probe[1]}):
        a = m1.predict_logprobs(s, **kw)
        b = m2.predict_logprobs(s, **kw)
        assert a.shape == (s.T, 4) and np.array_equal(a, b), kw
        assert np.isneginf(a[:, 2:]).all()
        assert np.allclose(np.exp(a[:, :2]).sum(axis=1), 1.0, atol=1e-5)
    assert check_causal(m1, s) == []
    assert m1.param_count() > 0 and "per_person_params" in m1.describe()


def _gain(name: str, kind: str, hetero: bool, cfg: dict, n_train: int, T: int) -> tuple[float, float]:
    """Mean (adapt − frozen) NLL on 60 held-out synthetic people and its standard error."""
    common.clear_backbone_cache()
    train = make_population(n_train, T, hetero=hetero, seed=11, kind=kind)
    test = make_population(60, T, hetero=hetero, seed=11, offset=10_000, kind=kind)
    m = get_model(name).factory(dict(cfg), 4, input_dim(4))
    m.fit(train, seed=5, threads=2, budget_s=300)
    d = np.array(
        [
            evaluate.score(s, m.predict_logprobs(s), 0.0).nll
            - evaluate.score(s, m.predict_logprobs(s, mode="frozen"), 0.0).nll
            for s in test
        ]
    )
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


TOY = [
    # (candidate, toy kind, config, n_train, T)
    ("f2_world_policy", "bandit", {"hidden": 16, "lr": 0.01}, 60, 60),
    ("f3_arbiter", "cue", {"hidden": 16, "lr": 0.01, "bandwidth": 0.5}, 60, 60),
    ("f1_latent", "cue", {"hidden": 16, "lr": 0.01, "m": 10, "meta_epochs": 8}, 60, 60),
]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "kind", "cfg", "n", "T"), TOY)
def test_toy_adaptation_helps_only_when_individuals_differ(name, kind, cfg, n, T):
    het, het_se = _gain(name, kind, True, cfg, n, T)
    hom, hom_se = _gain(name, kind, False, cfg, n, T)
    # individuals differ: adaptation must beat the frozen variant clearly
    assert het < -0.01 and het < -3 * het_se, (het, het_se)
    # nobody differs: no spurious adaptation gain
    assert hom > -0.005, (hom, hom_se)
