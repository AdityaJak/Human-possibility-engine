"""Off-policy evaluation of suggestion policies on micro-randomised-trial data.

Target: the mean proximal outcome V(pi) = E[Y_t under pi] over available decision points, where
``pi(x_t)`` gives the probability of sending a suggestion given the decision-point context. Because
the logging policy randomised with known probability ``p`` at every available point, importance
weights are exact:

    IPW:  V = mean_t [ pi(A_t|x_t) / p(A_t) * Y_t ]
    DR:   V = mean_t [ sum_a pi(a|x_t) Q(x_t,a) + pi(A_t|x_t)/p(A_t) * (Y_t - Q(x_t,A_t)) ]

with an outcome model Q fitted on the *dev* partition only (never on the rows being evaluated).
Uncertainty: participant cluster bootstrap. This is a retrospective estimate under the identification
assumptions of the trial; it is not a prospective result and says nothing about long-run effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hpe.core.rng import stream
from hpe.evaluation.causal.features import Frame

Policy = Callable[[Frame], np.ndarray]  # returns P(send = 1 | x_t) per row


def always(frame: Frame) -> np.ndarray:
    return np.ones(len(frame))


def never(frame: Frame) -> np.ndarray:
    return np.zeros(len(frame))


def behaviour(p: float) -> Policy:
    return lambda frame: np.full(len(frame), p)


@dataclass(slots=True)
class OutcomeModel:
    """Linear Q(x, a) = g(x)' alpha + a * (beta0 + beta' m(x)), least squares on dev."""

    moderators: list[str]
    coef: np.ndarray

    @staticmethod
    def design(frame: Frame, moderators: list[str], a: np.ndarray) -> np.ndarray:
        g = frame.controls()
        m = np.column_stack([np.ones(len(frame))] + [frame.moderators[k] for k in moderators])
        return np.hstack([g, a[:, None] * m])

    @classmethod
    def fit(cls, frame: Frame, moderators: list[str]) -> OutcomeModel:
        X = cls.design(frame, moderators, frame.a)
        coef, *_ = np.linalg.lstsq(X, frame.y, rcond=None)
        return cls(list(moderators), coef)

    def predict(self, frame: Frame, a: np.ndarray) -> np.ndarray:
        return self.design(frame, self.moderators, a) @ self.coef

    def cate(self, frame: Frame) -> np.ndarray:
        return self.predict(frame, np.ones(len(frame))) - self.predict(frame, np.zeros(len(frame)))


def contextual_greedy(model: OutcomeModel) -> Policy:
    """Send iff the dev-fitted conditional effect is positive."""
    return lambda frame: (model.cate(frame) > 0).astype(float)


def _value(
    frame: Frame, pi: np.ndarray, p: float, q: OutcomeModel | None, idx: np.ndarray
) -> tuple[float, float]:
    a = frame.a[idx]
    y = frame.y[idx]
    pi_a = np.where(a == 1, pi[idx], 1 - pi[idx])
    p_a = np.where(a == 1, p, 1 - p)
    w = pi_a / p_a
    ipw = float(np.mean(w * y))
    if q is None:
        return ipw, float("nan")
    sub = frame.subset(idx)
    q1, q0 = q.predict(sub, np.ones(len(sub))), q.predict(sub, np.zeros(len(sub)))
    q_a = np.where(a == 1, q1, q0)
    dm = pi[idx] * q1 + (1 - pi[idx]) * q0
    dr = float(np.mean(dm + w * (y - q_a)))
    return ipw, dr


def evaluate_policies(
    frame: Frame,
    policies: dict[str, Policy],
    *,
    p: float,
    q: OutcomeModel | None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """IPW and DR values with participant-cluster bootstrap CIs, plus contrasts vs ``never``."""
    n = len(frame)
    uniq, inv = np.unique(frame.users, return_inverse=True)
    by_user = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = stream(seed, "ope", "cluster_bootstrap")
    pis = {name: pol(frame) for name, pol in policies.items()}
    full = np.arange(n)
    point = {name: _value(frame, pi, p, q, full) for name, pi in pis.items()}
    boots = {name: np.empty((n_bootstrap, 2)) for name in pis}
    for b in range(n_bootstrap):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([by_user[c] for c in pick])
        for name, pi in pis.items():
            boots[name][b] = _value(frame, pi, p, q, idx)
    out: dict[str, dict] = {}
    for name in pis:
        ipw, dr = point[name]
        bi, bd = boots[name][:, 0], boots[name][:, 1]
        row = {
            "ipw": ipw,
            "ipw_ci": [float(np.quantile(bi, 0.025)), float(np.quantile(bi, 0.975))],
            "dr": dr,
            "dr_ci": [float(np.nanquantile(bd, 0.025)), float(np.nanquantile(bd, 0.975))] if q else None,
            "share_send": float(pis[name].mean()),
            "effective_sample_size_ipw": float(
                (
                    np.sum(
                        np.where(frame.a == 1, pis[name], 1 - pis[name]) / np.where(frame.a == 1, p, 1 - p)
                    )
                    ** 2
                )
                / np.sum(
                    (np.where(frame.a == 1, pis[name], 1 - pis[name]) / np.where(frame.a == 1, p, 1 - p)) ** 2
                )
            ),
        }
        if "never" in pis and name != "never":
            d_ipw = bi - boots["never"][:, 0]
            row["ipw_minus_never"] = ipw - point["never"][0]
            row["ipw_minus_never_ci"] = [float(np.quantile(d_ipw, 0.025)), float(np.quantile(d_ipw, 0.975))]
            if q:
                d_dr = bd - boots["never"][:, 1]
                row["dr_minus_never"] = dr - point["never"][1]
                row["dr_minus_never_ci"] = [
                    float(np.nanquantile(d_dr, 0.025)),
                    float(np.nanquantile(d_dr, 0.975)),
                ]
        out[name] = row
    return out
