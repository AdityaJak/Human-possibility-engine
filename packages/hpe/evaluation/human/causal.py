"""Centred and weighted least squares (WCLS) for micro-randomised trials.

Estimand: the *causal excursion effect* of delivering a suggestion (vs not) at an available
decision point on the proximal outcome, marginalised over study days (Boruvka et al., 2018,
JASA; the method named in Klasnja et al., 2019). With a constant randomisation probability
``p`` and numerator probability ``p_tilde = p`` the WCLS weights are all 1, and the estimator
is least squares with the treatment indicator centred at ``p``::

    Y_t = g_t' alpha + (A_t - p) f_t' beta + e_t

Identification rests on randomisation of ``A_t`` given availability (sequential ignorability by
design), consistency, and positivity (0 < p < 1). Control covariates ``g_t`` only improve
precision; they do not need to be correctly specified for ``beta`` to be consistent when
``f_t = 1`` and centring uses the true randomisation probability.

Uncertainty: cluster-robust (by participant) sandwich standard errors with a CR1 small-sample
factor and t(G-1) reference distribution, and a participant-level cluster bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from hpe.core.rng import stream


@dataclass(slots=True)
class WCLSFit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    df: int
    n_obs: int
    n_clusters: int
    coef_all: np.ndarray
    treatment_slice: slice
    bootstrap: np.ndarray | None = None  # (n_boot, n_beta)
    meta: dict = field(default_factory=dict)

    def summary(self, level: float = 0.95) -> list[dict]:
        tcrit = float(stats.t.ppf(0.5 + level / 2, self.df))
        out = []
        for i, name in enumerate(self.names):
            b, s = float(self.beta[i]), float(self.se[i])
            row = {
                "term": name,
                "estimate": b,
                "se_cluster_robust": s,
                "t": b / s if s > 0 else float("nan"),
                "df": self.df,
                "p_value": float(2 * stats.t.sf(abs(b / s), self.df)) if s > 0 else float("nan"),
                "ci_low_sandwich": b - tcrit * s,
                "ci_high_sandwich": b + tcrit * s,
            }
            if self.bootstrap is not None:
                lo, hi = np.quantile(self.bootstrap[:, i], [(1 - level) / 2, 0.5 + level / 2])
                row["ci_low_bootstrap"] = float(lo)
                row["ci_high_bootstrap"] = float(hi)
                row["n_bootstrap"] = int(self.bootstrap.shape[0])
            out.append(row)
        return out


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def wcls(
    y: np.ndarray,
    controls: np.ndarray,
    treatments: np.ndarray,
    probs: np.ndarray,
    clusters: np.ndarray,
    *,
    moderators: np.ndarray | None = None,
    treatment_names: list[str] | None = None,
    moderator_names: list[str] | None = None,
    numerator_probs: np.ndarray | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> WCLSFit:
    """Fit WCLS.

    Args:
        y: (N,) proximal outcome.
        controls: (N, q) control covariates g_t (include an intercept column).
        treatments: (N, m) 0/1 indicators for each non-reference treatment level.
        probs: (N, m) randomisation probabilities of each level.
        clusters: (N,) participant ids.
        moderators: (N, r) effect moderators f_t (default: intercept only -> marginal effect).
        numerator_probs: p_tilde (default = probs, giving unit weights).
    """
    n = len(y)
    f = np.ones((n, 1)) if moderators is None else moderators
    ptil = probs if numerator_probs is None else numerator_probs
    # Weights W = prod over levels of (p_tilde/p)^A, times reference-level ratio for A = 0.
    a_any = treatments.sum(axis=1)
    w = np.ones(n)
    for j in range(treatments.shape[1]):
        w *= np.where(treatments[:, j] == 1, ptil[:, j] / probs[:, j], 1.0)
    w *= np.where(a_any == 0, (1 - ptil.sum(axis=1)) / (1 - probs.sum(axis=1)), 1.0)

    centred = treatments - ptil
    blocks = [centred[:, [j]] * f for j in range(treatments.shape[1])]
    X = np.hstack([controls, *blocks])
    coef = _ols(X, y, w)
    resid = y - X @ coef

    # Cluster-robust sandwich with CR1 small-sample correction.
    uniq, inv = np.unique(clusters, return_inverse=True)
    g = len(uniq)
    xw = X * w[:, None]
    bread = np.linalg.pinv(X.T @ xw)
    scores = np.zeros((g, X.shape[1]))
    np.add.at(scores, inv, xw * resid[:, None])
    meat = scores.T @ scores
    k = X.shape[1]
    cr1 = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = cr1 * bread @ meat @ bread
    q = controls.shape[1]
    tslice = slice(q, k)
    beta = coef[tslice]
    se = np.sqrt(np.clip(np.diag(cov)[tslice], 0, None))

    tn = treatment_names or [f"A{j}" for j in range(treatments.shape[1])]
    mn = moderator_names or (["intercept"] if moderators is None else [f"f{r}" for r in range(f.shape[1])])
    names = [f"{t}:{m}" for t in tn for m in mn]

    boot = None
    if n_bootstrap > 0:
        rng = stream(seed, "wcls", "cluster_bootstrap")
        idx_by_cluster = [np.flatnonzero(inv == c) for c in range(g)]
        boot = np.empty((n_bootstrap, len(beta)))
        for b in range(n_bootstrap):
            pick = rng.integers(0, g, size=g)
            rows = np.concatenate([idx_by_cluster[c] for c in pick])
            cb = _ols(X[rows], y[rows], w[rows])
            boot[b] = cb[tslice]
    return WCLSFit(names, beta, se, g - 1, n, g, coef, tslice, boot)


def percent_effect(beta_log: float) -> float:
    """Multiplicative effect on (steps + 0.5) implied by a log-scale coefficient."""
    return float(np.expm1(beta_log))
