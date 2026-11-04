from pathlib import Path

import numpy as np
from hv_fixtures import write_heartsteps_csv

from hpe.evaluation.human import causal, heartsteps


def _simulate(beta: float, n_users: int = 40, n_t: int = 120, p: float = 0.6, seed: int = 0):
    rng = np.random.default_rng(seed)
    ys, gs, as_, us = [], [], [], []
    for u in range(n_users):
        ui = rng.normal(0, 1)
        for _t in range(n_t):
            x = rng.normal()
            a = float(rng.random() < p)
            y = 1.0 + ui + 0.5 * x + beta * a + rng.normal(0, 1)
            ys.append(y)
            gs.append([1.0, x])
            as_.append([a])
            us.append(u)
    return np.array(ys), np.array(gs), np.array(as_), np.array(us)


def test_wcls_recovers_known_effect_with_valid_ci():
    y, g, a, u = _simulate(0.3)
    fit = causal.wcls(y, g, a, np.full_like(a, 0.6), u, n_bootstrap=300, seed=1)
    row = fit.summary()[0]
    assert abs(row["estimate"] - 0.3) < 0.08
    assert row["ci_low_bootstrap"] < 0.3 < row["ci_high_bootstrap"]
    assert row["ci_low_sandwich"] < 0.3 < row["ci_high_sandwich"]
    assert fit.n_clusters == 40 and fit.df == 39


def test_wcls_null_effect_not_significant_and_deterministic():
    y, g, a, u = _simulate(0.0, seed=3)
    f1 = causal.wcls(y, g, a, np.full_like(a, 0.6), u, n_bootstrap=200, seed=5)
    f2 = causal.wcls(y, g, a, np.full_like(a, 0.6), u, n_bootstrap=200, seed=5)
    assert np.array_equal(f1.bootstrap, f2.bootstrap)
    assert f1.summary()[0]["p_value"] > 0.01


def test_wcls_misspecified_controls_still_unbiased_under_randomisation():
    # outcome depends nonlinearly on x; controls only include an intercept
    rng = np.random.default_rng(11)
    n = 6000
    x = rng.normal(size=n)
    a = (rng.random(n) < 0.6).astype(float)[:, None]
    y = np.sin(3 * x) + x**2 + 0.25 * a[:, 0] + rng.normal(0, 0.5, n)
    fit = causal.wcls(y, np.ones((n, 1)), a, np.full_like(a, 0.6), np.arange(n) // 50)
    assert abs(fit.beta[0] - 0.25) < 0.06


def test_two_level_treatment_and_moderators(tmp_path: Path):
    path = tmp_path / "hs.csv"
    write_heartsteps_csv(path, beta_walk=0.4, beta_sed=0.0, n_users=40, n_dec=150, seed=2)
    pts = heartsteps.analysis_sample(heartsteps.load_decision_points(path))
    y = np.log(np.array([d.steps30 for d in pts]) + 0.5)
    pre = np.log(np.array([d.steps30_pre for d in pts]) + 0.5)
    day = np.array([d.study_day for d in pts], float)
    walk = np.array([bool(d.send and d.send_walking) for d in pts], float)
    sed = np.array([bool(d.send and d.send_sedentary) for d in pts], float)
    a2 = np.column_stack([walk, sed])
    probs = np.column_stack([np.full(len(y), walk.mean()), np.full(len(y), sed.mean())])
    fit = causal.wcls(
        y, np.column_stack([np.ones(len(y)), pre, day]), a2, probs, np.array([d.user for d in pts])
    )
    assert abs(fit.beta[0] - 0.4) < 0.12 and abs(fit.beta[1]) < 0.12
    tm = causal.wcls(
        y,
        np.column_stack([np.ones(len(y)), pre, day]),
        walk[:, None],
        np.full((len(y), 1), walk.mean()),
        np.array([d.user for d in pts]),
        moderators=np.column_stack([np.ones(len(y)), day - day.mean()]),
    )
    assert len(tm.beta) == 2 and tm.names == ["A0:f0", "A0:f1"]
    assert abs(causal.percent_effect(np.log(1.14)) - 0.14) < 1e-12


def test_heartsteps_loader_and_validation(tmp_path: Path):
    path = tmp_path / "hs.csv"
    write_heartsteps_csv(path, beta_walk=0.2, beta_sed=0.0, n_users=5, n_dec=20, seed=0)
    pts = heartsteps.load_decision_points(path)
    assert len(pts) == 100 and pts[0].study_day == 0 and pts[-1].study_day == 3
    v = heartsteps.validate(pts)
    assert v["users"] == 5 and 0 < v["p_send_given_available"] < 1
    assert v["analysis_sample"] == sum(d.available for d in pts)
    assert v["complete_case_n"] <= v["analysis_sample"]
