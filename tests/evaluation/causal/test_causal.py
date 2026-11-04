import json
from pathlib import Path

import numpy as np
import pytest

from hpe.evaluation.causal import features, hte, mrt_sim, ope, partition


def test_partition_is_deterministic_and_tamper_evident(tmp_path: Path):
    users = list(range(1, 38))
    a = partition.partition_record(users)
    b = partition.partition_record(users)
    assert a == b and len(a["dev"]) == 22 and len(a["test"]) == 15
    assert not set(a["dev"]) & set(a["test"])
    p = tmp_path / "partition.json"
    partition.write_partition(users, p)
    assert partition.load_partition(p)["test"] == a["test"]
    rec = json.loads(p.read_text())
    rec["test"] = rec["test"][:-1]
    p.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError):
        partition.load_partition(p)


def test_simulated_mrt_has_design_probabilities():
    d = mrt_sim.MRTDesign(n_users=20, n_days=10)
    data = mrt_sim.simulate("constant", d, seed=1)
    assert 0.55 < data.a.mean() < 0.65
    assert len(np.unique(data.users)) == 20
    assert 0.7 * 20 * 50 < len(data.y) < 0.9 * 20 * 50


def test_wcls_recovers_known_effect_and_moderator():
    d = mrt_sim.MRTDesign(n_users=60, n_days=42, sd_noise=0.8)
    ests, ests1 = [], []
    for r in range(12):
        data = mrt_sim.simulate("context", d, seed=100 + r)
        ests.append(mrt_sim.fit_marginal(data)["estimate"])
        mm = mrt_sim.fit_moderated(data, data.context - d.p_context, "ctx")
        ests1.append(mm["beta1"]["estimate"])
    assert abs(np.mean(ests) - d.beta0) < 0.03
    assert abs(np.mean(ests1) - d.beta1) < 0.06


def test_null_scenario_has_zero_truth_and_carryover_keeps_marginal():
    d = mrt_sim.MRTDesign(n_users=40, n_days=20, sd_noise=0.8)
    assert mrt_sim.simulate("null", d, 3).truth["beta0"] == 0.0
    ests = [mrt_sim.fit_marginal(mrt_sim.simulate("carryover", d, 200 + r))["estimate"] for r in range(10)]
    assert abs(np.mean(ests) - d.beta0) < 0.04


def test_holm_matches_reference():
    # step-down Holm: sorted p (0.01, 0.03, 0.04) x (3, 2, 1) = (0.03, 0.06, 0.04) -> monotone (0.03, 0.06, 0.06)
    assert hte.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert hte.holm([0.5]) == [0.5]


def _frame_from_sim(scenario: str, seed: int, n_users: int = 40) -> features.Frame:
    d = mrt_sim.MRTDesign(n_users=n_users, n_days=20, sd_noise=0.8)
    data = mrt_sim.simulate(scenario, d, seed)
    n = len(data.y)
    mods = {
        "evening": (np.arange(n) % 2).astype(float),
        "weekend": np.zeros(n),
        "log_pre_c": data.log_pre - data.log_pre.mean(),
        "home": data.context,
        "day_c": data.day / 19 - 0.5,
    }
    return features.Frame(
        users=data.users,
        y=data.y,
        steps=np.expm1(data.y),
        a=data.a,
        a_walk=data.a,
        a_sed=np.zeros(n),
        log_pre=data.log_pre,
        study_day=data.day,
        moderators=mods,
        is_randomized=np.ones(n, dtype=bool),
    )


def test_moderated_effects_find_the_planted_moderator():
    fr = _frame_from_sim("context", 7, n_users=80)
    rows = hte.moderated_effects(fr, list(features.MODERATOR_NAMES), p=0.6, n_bootstrap=50, seed=1)
    by = {r.moderator: r for r in rows}
    assert by["home"].beta1 > 0.1 and by["home"].p_raw < 0.05
    assert all(0 <= r.p_holm <= 1 for r in rows)
    sel = hte.select_moderators(rows, max_select=2, p_threshold=0.1)
    assert "home" in sel and len(sel) <= 2


def test_ope_ipw_and_dr_match_known_policy_values():
    fr = _frame_from_sim("context", 11, n_users=80)
    q = ope.OutcomeModel.fit(fr, ["home"])
    res = ope.evaluate_policies(
        fr,
        {"always": ope.always, "never": ope.never, "behaviour": ope.behaviour(0.6)},
        p=0.6,
        q=q,
        n_bootstrap=100,
        seed=2,
    )
    # behaviour-policy IPW value equals the sample mean outcome exactly
    assert res["behaviour"]["ipw"] == pytest.approx(float(fr.y.mean()))
    # always - never recovers the marginal effect (beta0 = 0.115) within sampling error
    diff = res["always"]["ipw"] - res["never"]["ipw"]
    assert abs(diff - 0.115) < 0.08
    assert abs(res["always"]["dr"] - res["never"]["dr"] - 0.115) < 0.08
    lo, hi = res["always"]["ipw_minus_never_ci"]
    assert lo < diff < hi


def test_contextual_policy_sends_where_cate_positive():
    fr = _frame_from_sim("context", 5, n_users=80)
    q = ope.OutcomeModel.fit(fr, ["home"])
    pi = ope.contextual_greedy(q)(fr)
    assert set(np.unique(pi)) <= {0.0, 1.0}
    cate = q.cate(fr)
    assert np.all((pi == 1) == (cate > 0))


def test_features_from_synthetic_csv(tmp_path: Path):
    cols = [
        "user.index",
        "decision.index",
        "sugg.decision.utime",
        "sugg.gmtoff",
        "avail",
        "is.randomized",
        "send",
        "send.active",
        "send.sedentary",
        "jbsteps30.zero",
        "jbsteps30pre.zero",
        "jbsteps30",
        "dec.location.category",
    ]
    rows = [
        [
            "1",
            "0",
            "2015-07-22 16:31:53",
            "-14400.0",
            "True",
            "True",
            "True",
            "True",
            "False",
            "100",
            "50",
            "100",
            "home",
        ],
        [
            "1",
            "1",
            "2015-07-25 23:10:00",
            "-14400.0",
            "True",
            "False",
            "False",
            "False",
            "False",
            "20",
            "10",
            "",
            "work",
        ],
        ["2", "0", "2015-07-22 16:31:53", "-14400.0", "False", "True", "", "", "", "", "", "", "home"],
    ]
    p = tmp_path / "s.csv"
    p.write_text(",".join(cols) + "\n" + "\n".join(",".join(r) for r in rows) + "\n")
    loaded = features.load_rows(p)
    assert len(loaded) == 3 and loaded[0].local_hour == pytest.approx(12 + 31 / 60) and loaded[1].weekday == 5
    fr = features.build_frame(loaded)
    assert (
        len(fr) == 2
        and fr.moderators["weekend"].tolist() == [0.0, 1.0]
        and fr.moderators["home"].tolist() == [1.0, 0.0]
    )
    assert fr.moderators["evening"].tolist() == [0.0, 1.0]
    assert len(features.build_frame(loaded, users=[2])) == 0
    assert features.describe(fr)["is_randomized_equals_send"] is True
