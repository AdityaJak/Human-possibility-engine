"""Statistics for preregistered experiment analysis.

All resampling procedures take an explicit integer ``seed`` and use ``hpe.core.rng`` streams, so
analyses regenerate bit-identically from raw logs. Unit of analysis is chosen by the caller
(typically: one value per seed / independent society).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from hpe.core.rng import stream
from hpe.tasks.assessment import wilson_interval

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "calibration_bins",
    "cohen_d",
    "cohen_dz",
    "expected_calibration_error",
    "hedges_g",
    "holm",
    "paired_bootstrap",
    "permutation_test_paired",
    "permutation_test_unpaired",
    "summarize",
    "unpaired_bootstrap",
    "welch_t",
    "wilson_interval",
]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n: int
    n_boot: int
    method: str


def _arr(x: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    stat: Callable[[np.ndarray], float] = lambda v: float(np.mean(v)),
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for a one-sample statistic."""
    a = _arr(x)
    if a.size == 0:
        return BootstrapResult(
            float("nan"), float("nan"), float("nan"), level, 0, n_boot, "percentile_bootstrap"
        )
    est = float(stat(a))
    if a.size == 1:
        return BootstrapResult(est, est, est, level, 1, n_boot, "percentile_bootstrap")
    rng = stream(seed, "bootstrap", "one_sample", a.size)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.array([stat(a[i]) for i in idx]) if stat is not None else a[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return BootstrapResult(est, float(lo), float(hi), level, int(a.size), n_boot, "percentile_bootstrap")


def paired_bootstrap(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """CI for mean(treatment - control) resampling paired units (e.g. seeds)."""
    t = np.asarray(treatment, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.shape != c.shape:
        raise ValueError("paired samples must have equal length")
    ok = np.isfinite(t) & np.isfinite(c)
    d = t[ok] - c[ok]
    if d.size == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), level, 0, n_boot, "paired_bootstrap")
    est = float(d.mean())
    if d.size == 1:
        return BootstrapResult(est, est, est, level, 1, n_boot, "paired_bootstrap")
    rng = stream(seed, "bootstrap", "paired", d.size)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return BootstrapResult(est, float(lo), float(hi), level, int(d.size), n_boot, "paired_bootstrap")


def unpaired_bootstrap(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """CI for mean(treatment) - mean(control), resampling each group independently."""
    t, c = _arr(treatment), _arr(control)
    if t.size == 0 or c.size == 0:
        return BootstrapResult(
            float("nan"), float("nan"), float("nan"), level, 0, n_boot, "unpaired_bootstrap"
        )
    est = float(t.mean() - c.mean())
    rng = stream(seed, "bootstrap", "unpaired", t.size, c.size)
    bt = t[rng.integers(0, t.size, size=(n_boot, t.size))].mean(axis=1)
    bc = c[rng.integers(0, c.size, size=(n_boot, c.size))].mean(axis=1)
    lo, hi = np.quantile(bt - bc, [(1 - level) / 2, 1 - (1 - level) / 2])
    return BootstrapResult(
        est, float(lo), float(hi), level, int(min(t.size, c.size)), n_boot, "unpaired_bootstrap"
    )


def permutation_test_paired(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> float:
    """Sign-flip permutation test on paired differences (exact when n <= 16)."""
    t = np.asarray(treatment, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(c)
    d = t[ok] - c[ok]
    n = d.size
    if n == 0:
        return float("nan")
    obs = d.mean()
    if n <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        rng = stream(seed, "permutation", "paired", n)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * d[None, :]).mean(axis=1)
    return _p_from_null(null, obs, alternative, exact=n <= 16)


def permutation_test_unpaired(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> float:
    t, c = _arr(treatment), _arr(control)
    if t.size == 0 or c.size == 0:
        return float("nan")
    pooled = np.concatenate([t, c])
    obs = t.mean() - c.mean()
    rng = stream(seed, "permutation", "unpaired", t.size, c.size)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pooled)
        null[i] = p[: t.size].mean() - p[t.size :].mean()
    return _p_from_null(null, obs, alternative, exact=False)


def _p_from_null(null: np.ndarray, obs: float, alternative: str, *, exact: bool) -> float:
    eps = 1e-12
    if alternative == "greater":
        k = int(np.sum(null >= obs - eps))
    elif alternative == "less":
        k = int(np.sum(null <= obs + eps))
    else:
        k = int(np.sum(np.abs(null) >= abs(obs) - eps))
    if exact:
        return k / null.size
    return (k + 1) / (null.size + 1)


def welch_t(
    treatment: Sequence[float] | np.ndarray, control: Sequence[float] | np.ndarray
) -> tuple[float, float, float]:
    """Welch's unequal-variance t test. Returns (t, df, two-sided p)."""
    t, c = _arr(treatment), _arr(control)
    vt, vc = t.var(ddof=1) / t.size, c.var(ddof=1) / c.size
    se = math.sqrt(vt + vc)
    if se == 0:
        return (float("nan"), float("nan"), float("nan"))
    tstat = (t.mean() - c.mean()) / se
    df = (vt + vc) ** 2 / (vt**2 / (t.size - 1) + vc**2 / (c.size - 1))
    p = 2 * sps.t.sf(abs(tstat), df)
    return (float(tstat), float(df), float(p))


def cohen_d(treatment: Sequence[float] | np.ndarray, control: Sequence[float] | np.ndarray) -> float:
    t, c = _arr(treatment), _arr(control)
    nt, nc = t.size, c.size
    if nt < 2 or nc < 2:
        return float("nan")
    sp = math.sqrt(((nt - 1) * t.var(ddof=1) + (nc - 1) * c.var(ddof=1)) / (nt + nc - 2))
    return float((t.mean() - c.mean()) / sp) if sp > 0 else float("nan")


def hedges_g(treatment: Sequence[float] | np.ndarray, control: Sequence[float] | np.ndarray) -> float:
    t, c = _arr(treatment), _arr(control)
    d = cohen_d(t, c)
    df = t.size + c.size - 2
    j = 1 - 3 / (4 * df - 1) if df > 0 else float("nan")
    return float(d * j)


def cohen_dz(treatment: Sequence[float] | np.ndarray, control: Sequence[float] | np.ndarray) -> float:
    """Standardised mean of paired differences."""
    t = np.asarray(treatment, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(c)
    d = t[ok] - c[ok]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") * float(np.sign(d.mean())) if d.mean() != 0 else 0.0
    return float(d.mean() / sd)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone), NaNs passed through."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = ok.size
    if m == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


def calibration_bins(
    predicted: Sequence[float] | np.ndarray, observed: Sequence[float] | np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Reliability-diagram bins: mean prediction vs mean observed outcome per bin."""
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = np.clip(p[ok], 0, 1), o[ok]
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = which == b
        if m.any():
            bins.append(
                {
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "mean_predicted": float(p[m].mean()),
                    "mean_observed": float(o[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return bins


def expected_calibration_error(
    predicted: Sequence[float] | np.ndarray, observed: Sequence[float] | np.ndarray, n_bins: int = 10
) -> float:
    bins = calibration_bins(predicted, observed, n_bins)
    total = sum(b["n"] for b in bins)
    if total == 0:
        return float("nan")
    return float(sum(b["n"] * abs(b["mean_predicted"] - b["mean_observed"]) for b in bins) / total)


def summarize(values: Sequence[float] | np.ndarray, *, seed: int = 0, n_boot: int = 2000) -> dict[str, float]:
    a = _arr(values)
    if a.size == 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0.0,
        }
    b = bootstrap_ci(a, n_boot=n_boot, seed=seed)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "ci_low": b.ci_low,
        "ci_high": b.ci_high,
        "n": float(a.size),
    }
