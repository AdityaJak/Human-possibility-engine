"""Synthetic micro-randomised trials with known causal excursion effects.

Used to validate the WCLS estimator (``hpe.evaluation.human.causal.wcls``) that Phase I applied to
HeartSteps: coverage of its 95% intervals, bias, power at the HeartSteps sample size, and type I
error under the null. The generator mimics HeartSteps' structure — participants, 5 decision points a
day over 6 weeks, availability, randomised suggestion with P(send) = 0.6, log-scale outcome with a
participant random effect and a prior-activity covariate — without using any HeartSteps data.

Scenarios (each with the *true marginal excursion effect* computed analytically):

- ``constant``       beta_t = beta0.
- ``context``        beta_t = beta0 + beta1 * (C_t - E[C]) with a binary context C_t (centred, so the
                     marginal effect is beta0 and the moderator coefficient is beta1).
- ``carryover``      as ``constant`` plus a carry-over of the previous suggestion into Y_t
                     (gamma * A_{t-1}): the excursion effect of A_t stays beta0 by randomisation.
- ``decay``          beta_t = beta0 + beta1 * (day_c) with day_c = day/(D-1) - 0.5 (centred): the
                     marginal effect is beta0; a day-moderated WCLS should recover beta1.
- ``nuisance``       as ``constant`` but the control model is misspecified (true nuisance has a
                     quadratic day term and an interaction): WCLS with f_t = 1 and centring at the
                     true p should remain unbiased for beta0.
- ``null``           beta0 = 0 for the type I error rate.

Identification assumptions embodied here and required for HeartSteps: A_t randomised with a known
probability at available points; consistency; no interference between participants; positivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from hpe.core.rng import stream
from hpe.evaluation.human.causal import wcls

Scenario = Literal["constant", "context", "carryover", "decay", "nuisance", "null"]


@dataclass(slots=True)
class MRTDesign:
    n_users: int = 37
    n_days: int = 42
    slots_per_day: int = 5
    p_avail: float = 0.8
    p_send: float = 0.6
    beta0: float = 0.115  # log-scale marginal effect (≈ the HeartSteps point estimate)
    beta1: float = 0.20  # moderator coefficient where applicable
    gamma: float = 0.10  # carry-over of A_{t-1}
    sd_user: float = 0.6
    sd_noise: float = 1.6  # residual SD on the log(steps+0.5) scale (HeartSteps-like)
    p_context: float = 0.4
    a_pre: float = 0.35
    a_day: float = -0.003


@dataclass(slots=True)
class MRTData:
    y: np.ndarray
    a: np.ndarray
    users: np.ndarray
    day: np.ndarray
    log_pre: np.ndarray
    context: np.ndarray
    p: float
    truth: dict[str, float] = field(default_factory=dict)


def simulate(scenario: Scenario, design: MRTDesign, seed: int) -> MRTData:
    rng = stream(seed, "mrt_sim", scenario)
    d = design
    n_points = d.n_days * d.slots_per_day
    ys, As, us, days, pres, ctxs = [], [], [], [], [], []
    for u in range(d.n_users):
        u_eff = rng.normal(0, d.sd_user)
        prev_a = 0.0
        for t in range(n_points):
            if rng.random() >= d.p_avail:
                prev_a = 0.0
                continue
            day = t // d.slots_per_day
            day_c = day / (d.n_days - 1) - 0.5
            log_pre = rng.normal(3.0, 1.5)
            ctx = float(rng.random() < d.p_context)
            a = float(rng.random() < d.p_send)
            beta0 = 0.0 if scenario == "null" else d.beta0
            if scenario == "context":
                beta_t = beta0 + d.beta1 * (ctx - d.p_context)
            elif scenario == "decay":
                beta_t = beta0 + d.beta1 * day_c
            else:
                beta_t = beta0
            mu = 3.5 + d.a_pre * (log_pre - 3.0) + d.a_day * day + u_eff
            if scenario == "nuisance":
                mu += 0.4 * day_c**2 * 4 + 0.15 * (log_pre - 3.0) * day_c
            if scenario == "carryover":
                mu += d.gamma * prev_a
            y = mu + a * beta_t + rng.normal(0, d.sd_noise)
            ys.append(y)
            As.append(a)
            us.append(u)
            days.append(day)
            pres.append(log_pre)
            ctxs.append(ctx)
            prev_a = a
    truth = {"beta0": 0.0 if scenario == "null" else d.beta0}
    if scenario in ("context", "decay"):
        truth["beta1"] = d.beta1
    return MRTData(
        np.array(ys),
        np.array(As),
        np.array(us),
        np.array(days),
        np.array(pres),
        np.array(ctxs),
        d.p_send,
        truth,
    )


def fit_marginal(data: MRTData, *, centre_at_truth: bool = True) -> dict[str, float]:
    n = len(data.y)
    controls = np.column_stack([np.ones(n), data.log_pre, data.day])
    p = data.p if centre_at_truth else float(data.a.mean())
    fit = wcls(data.y, controls, data.a[:, None], np.full((n, 1), p), data.users, treatment_names=["send"])
    row = fit.summary()[0]
    return {
        "estimate": row["estimate"],
        "ci_low": row["ci_low_sandwich"],
        "ci_high": row["ci_high_sandwich"],
        "p": row["p_value"],
    }


def fit_moderated(data: MRTData, moderator: np.ndarray, name: str) -> dict[str, dict[str, float]]:
    n = len(data.y)
    controls = np.column_stack([np.ones(n), data.log_pre, data.day, moderator])
    f = np.column_stack([np.ones(n), moderator])
    fit = wcls(
        data.y,
        controls,
        data.a[:, None],
        np.full((n, 1), data.p),
        data.users,
        moderators=f,
        treatment_names=["send"],
        moderator_names=["intercept", name],
    )
    out = {}
    for row in fit.summary():
        key = "beta0" if row["term"].endswith(":intercept") else "beta1"
        out[key] = {
            "estimate": row["estimate"],
            "ci_low": row["ci_low_sandwich"],
            "ci_high": row["ci_high_sandwich"],
            "p": row["p_value"],
        }
    return out


def validate_estimator(
    n_replicates: int = 500, design: MRTDesign | None = None, seed: int = 20261105
) -> dict:
    """Coverage, bias, power/type-I error of WCLS across scenarios. Deterministic in ``seed``."""
    d = design or MRTDesign()
    results: dict[str, dict] = {}
    for scenario in ("constant", "context", "carryover", "decay", "nuisance", "null"):
        est, cover, reject, est1, cover1, reject1, n_obs = [], [], [], [], [], [], []
        for r in range(n_replicates):
            data = simulate(scenario, d, seed * 1000 + r)
            n_obs.append(len(data.y))
            m = fit_marginal(data)
            est.append(m["estimate"])
            cover.append(m["ci_low"] <= data.truth["beta0"] <= m["ci_high"])
            reject.append(m["p"] < 0.05)
            if scenario in ("context", "decay"):
                mod = data.context - d.p_context if scenario == "context" else data.day / (d.n_days - 1) - 0.5
                mm = fit_moderated(data, mod, "moderator")
                est1.append(mm["beta1"]["estimate"])
                cover1.append(mm["beta1"]["ci_low"] <= data.truth["beta1"] <= mm["beta1"]["ci_high"])
                reject1.append(mm["beta1"]["p"] < 0.05)
        e = np.array(est)
        row: dict[str, Any] = {
            "n_replicates": n_replicates,
            "mean_n_obs": float(np.mean(n_obs)),
            "truth_beta0": data.truth["beta0"],
            "mean_estimate": float(e.mean()),
            "bias": float(e.mean() - data.truth["beta0"]),
            "sd_estimate": float(e.std(ddof=1)),
            "coverage_95": float(np.mean(cover)),
            "rejection_rate_alpha_05": float(np.mean(reject)),
        }
        if est1:
            e1 = np.array(est1)
            row["moderator"] = {
                "truth_beta1": data.truth["beta1"],
                "mean_estimate": float(e1.mean()),
                "bias": float(e1.mean() - data.truth["beta1"]),
                "coverage_95": float(np.mean(cover1)),
                "rejection_rate_alpha_05": float(np.mean(reject1)),
            }
        results[scenario] = row
    return {
        "design": d.__dict__ if hasattr(d, "__dict__") else {k: getattr(d, k) for k in d.__slots__},
        "seed": seed,
        "scenarios": results,
    }
