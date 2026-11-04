import numpy as np
import pytest
from scipy import stats as sps

from hpe.evaluation import stats


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1.0, 12), rng.normal(0.2, 2.0, 9)
    t, df, p = stats.welch_t(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert t == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    assert df > 0


def test_cohen_d_known_value():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([1.0, 3.0, 5.0])
    # pooled sd = 2, mean diff = 1 -> d = 0.5
    assert stats.cohen_d(a, b) == pytest.approx(0.5)
    assert stats.hedges_g(a, b) == pytest.approx(0.5 * (1 - 3 / (4 * 4 - 1)))


def test_cohen_dz_known_value():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    c = np.array([0.0, 0.0, 2.0, 2.0])  # diffs 1,2,1,2 -> mean 1.5, sd 0.57735
    assert stats.cohen_dz(t, c) == pytest.approx(1.5 / np.std([1, 2, 1, 2], ddof=1))


def test_paired_bootstrap_contains_true_diff_and_is_seeded():
    rng = np.random.default_rng(1)
    c = rng.normal(0, 1, 30)
    t = c + 0.5 + rng.normal(0, 0.1, 30)
    r1 = stats.paired_bootstrap(t, c, seed=3)
    r2 = stats.paired_bootstrap(t, c, seed=3)
    assert r1 == r2
    assert r1.ci_low < 0.5 < r1.ci_high and r1.n == 30


def test_unpaired_bootstrap_estimate():
    r = stats.unpaired_bootstrap([3.0, 3.0, 3.0], [1.0, 1.0], seed=0)
    assert r.estimate == pytest.approx(2.0) and r.ci_low == pytest.approx(2.0)


def test_exact_sign_flip_permutation():
    # all 5 differences positive -> exact two-sided p = 2/32
    p = stats.permutation_test_paired([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert p == pytest.approx(2 / 32)
    assert stats.permutation_test_paired([1, 2, 3, 4, 5], [0] * 5, alternative="greater") == pytest.approx(
        1 / 32
    )


def test_unpaired_permutation_detects_large_difference():
    p = stats.permutation_test_unpaired([10.0] * 8, [0.0] * 8, n_perm=500, seed=1)
    assert p < 0.01


def test_holm_matches_reference():
    # reference: statsmodels multipletests([0.01, 0.04, 0.03], method='holm') -> [0.03, 0.06, 0.06]
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = stats.holm([0.2, float("nan"), 0.01])
    assert np.isnan(out[1]) and out[2] == pytest.approx(0.02) and out[0] == pytest.approx(0.2)


def test_ece_perfect_and_bad():
    p = np.array([0.2] * 50 + [0.8] * 50)
    o = np.array([1] * 10 + [0] * 40 + [1] * 40 + [0] * 10)
    assert stats.expected_calibration_error(p, o) == pytest.approx(0.0)
    assert stats.expected_calibration_error(np.ones(10), np.zeros(10)) == pytest.approx(1.0)


def test_summarize_and_wilson():
    s = stats.summarize([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["n"] == 3 and s["ci_low"] <= 2 <= s["ci_high"]
    lo, hi = stats.wilson_interval(5, 10)
    assert lo < 0.5 < hi
