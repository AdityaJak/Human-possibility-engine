"""Heterogeneous treatment effects: WCLS with one pre-listed moderator at a time.

Model (Boruvka et al., 2018): Y_t = g_t' alpha + (A_t - p)(beta0 + beta1 * m_t) + e_t, with the
moderator also in g_t. ``beta1`` is the change in the causal excursion effect per unit of the
moderator. Inference: participant-cluster-robust sandwich (CR1, t with G-1 df) and a participant
cluster bootstrap; Holm correction across the moderators tested together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hpe.evaluation.causal.features import Frame
from hpe.evaluation.human.causal import percent_effect, wcls


@dataclass(slots=True)
class ModeratorResult:
    moderator: str
    beta0: float
    beta1: float
    se1: float
    p_raw: float
    p_holm: float
    ci_low_boot: float
    ci_high_boot: float
    ci_low_sandwich: float
    ci_high_sandwich: float
    n_obs: int
    n_users: int

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def marginal_effect(frame: Frame, *, p: float | None = None, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    n = len(frame)
    pp = float(frame.a.mean()) if p is None else p
    fit = wcls(
        frame.y,
        frame.controls(),
        frame.a[:, None],
        np.full((n, 1), pp),
        frame.users,
        treatment_names=["any_suggestion"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    row = fit.summary()[0]
    row["percent_effect"] = 100.0 * percent_effect(row["estimate"])
    row["centring_p"] = pp
    row["n_obs"] = n
    row["n_users"] = int(len(np.unique(frame.users)))
    return row


def moderated_effects(
    frame: Frame, moderators: list[str], *, p: float | None = None, n_bootstrap: int = 2000, seed: int = 0
) -> list[ModeratorResult]:
    n = len(frame)
    pp = float(frame.a.mean()) if p is None else p
    rows: list[ModeratorResult] = []
    for i, name in enumerate(moderators):
        m = frame.moderators[name]
        controls = np.column_stack([frame.controls(), m])
        f = np.column_stack([np.ones(n), m])
        fit = wcls(
            frame.y,
            controls,
            frame.a[:, None],
            np.full((n, 1), pp),
            frame.users,
            moderators=f,
            treatment_names=["send"],
            moderator_names=["intercept", name],
            n_bootstrap=n_bootstrap,
            seed=seed + i,
        )
        s = fit.summary()
        b0, b1 = s[0], s[1]
        rows.append(
            ModeratorResult(
                moderator=name,
                beta0=b0["estimate"],
                beta1=b1["estimate"],
                se1=b1["se_cluster_robust"],
                p_raw=b1["p_value"],
                p_holm=float("nan"),
                ci_low_boot=b1.get("ci_low_bootstrap", float("nan")),
                ci_high_boot=b1.get("ci_high_bootstrap", float("nan")),
                ci_low_sandwich=b1["ci_low_sandwich"],
                ci_high_sandwich=b1["ci_high_sandwich"],
                n_obs=n,
                n_users=int(len(np.unique(frame.users))),
            )
        )
    adj = holm([r.p_raw for r in rows])
    for r, a in zip(rows, adj, strict=True):
        r.p_holm = a
    return rows


def select_moderators(
    rows: list[ModeratorResult], *, max_select: int = 2, p_threshold: float = 0.10
) -> list[str]:
    """Discovery rule (dev only): up to ``max_select`` moderators with raw p < threshold, ranked by p."""
    ranked = sorted(rows, key=lambda r: r.p_raw)
    return [r.moderator for r in ranked if r.p_raw < p_threshold][:max_select]
