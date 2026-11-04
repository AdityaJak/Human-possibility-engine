"""``hb_cog`` — hierarchical Bayesian cognitive model, generic over encoding v3 (thesis §4 item 8).

Learner (one *particle* = one parameter vector θ = (α, ρ, β, κ, δ)):

* option values ``Q[key]`` start at ``q0 = 0.5`` (outcomes are in [0, 1]); an option's key is the state key
  (features ``state_features``) plus either the action index or, when ``option_features`` is given, the rounded
  features that describe that option (so an option's value follows it into new pairings, e.g. Garcia's transfer
  phase or a repeated gamble);
* after every step, each option whose outcome was observed (the v3 observed-outcome mask: chosen and, where
  shown, counterfactual) is updated by a delta rule, rate ``α`` for the chosen option and ``ρ·α`` for the others;
  a choice made without feedback is credited (rate ``α``) with the chosen outcome of the immediately following
  feedback step (one-step eligibility — e.g. the spaceship choice of a two-stage day);
* every unobserved value decays towards ``q0`` at rate ``δ``; values and perseveration traces are reset at a block
  start when ``reset_on_block``;
* choice: ``softmax_a(β Q[key_a] + κ · 1[a = last action in this state])`` over available actions.

Hierarchy: a fixed grid of ``P`` particles (5 α × 3 ρ × 5 β × 4 κ × 2 δ = 600). Population prior = mixing weights
``π`` over the particles, fitted on the dev participants by EM (non-parametric maximum likelihood of the mixing
distribution, with ``c`` pseudo-counts per particle). Per person, the posterior over particles is updated online
from the person's own scored choices (likelihood tempered by ``τ``); the prediction for step t is the
posterior-predictive mixture with the posterior from steps < t — causal by construction. ``mode="frozen"``: the
posterior stays at the population prior (each particle's learning still runs). ``donor=``: the posterior is
computed from the donor's whole sequence and held fixed.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from typing import Any, ClassVar

import numpy as np

from hpe.population.core.encoding import EncodedSeq, Layout
from hpe.population.core.interface import ModelSpec, register_model

ALPHA = (0.05, 0.15, 0.3, 0.5, 0.8)
RHO = (0.0, 0.5, 1.0)
BETA = (1.0, 3.0, 6.0, 10.0, 20.0)
KAPPA = (-0.5, 0.0, 0.5, 1.5)
DELTA = (0.0, 0.1)
Q0 = 0.5


def particle_grid() -> np.ndarray:
    return np.array(list(itertools.product(ALPHA, RHO, BETA, KAPPA, DELTA)), dtype=np.float64)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


class _Spec:
    def __init__(self, config: dict[str, Any]) -> None:
        self.state = tuple(int(i) for i in config.get("state_features", []))
        of = config.get("option_features")
        self.options = None if of is None else tuple(tuple(int(i) for i in o) for o in of)
        self.reset = bool(config.get("reset_on_block", False))


def run_particles(seq: EncodedSeq, theta: np.ndarray, spec: _Spec) -> np.ndarray:
    """Per-particle log-probabilities ``[T, P, K]`` (``-inf`` on unavailable actions); row t uses steps < t."""
    L = Layout(seq.k_max)
    T, K = seq.T, seq.k_max
    P = theta.shape[0]
    alpha, rho, beta, kappa, delta = (theta[:, i] for i in range(5))
    Q: dict[tuple, np.ndarray] = {}
    last: dict[tuple, int] = {}
    out = np.full((T, P, K), -np.inf)
    prev_keys: list[tuple] | None = None
    pending: tuple | None = None
    for t in range(T):
        x = seq.x[t].astype(np.float64)
        if t > 0 and spec.reset and x[L.block_start] > 0.5:
            Q.clear()
            last.clear()
            prev_keys, pending = None, None
        if t > 0 and prev_keys is not None:
            obs = x[L.prev_observed] > 0.5
            outs = x[L.prev_outcomes]
            pa = int(seq.actions[t - 1])
            fb = x[L.prev_feedback] > 0.5
            touched = set()
            for a in range(K):
                if obs[a]:
                    k = prev_keys[a]
                    q = Q.get(k, np.full(P, Q0))
                    rate = alpha if a == pa else alpha * rho
                    Q[k] = q + rate * (outs[a] - q)
                    touched.add(k)
            if fb and pending is not None and pending not in touched:
                q = Q.get(pending, np.full(P, Q0))
                Q[pending] = q + alpha * (x[L.prev_reward] - q)
                touched.add(pending)
            pending = None if fb else prev_keys[pa]
            if np.any(delta > 0):
                for k in list(Q):  # insertion order (deterministic); updates are independent per key
                    if k not in touched:
                        Q[k] = Q[k] + delta * (Q0 - Q[k])
        sk = tuple(np.round(x[list(spec.state)], 3)) if spec.state else ()
        keys: list[tuple] = []
        for a in range(K):
            if spec.options is not None and a < len(spec.options):
                keys.append((sk, tuple(np.round(x[list(spec.options[a])], 3))))
            else:
                keys.append((sk, a))
        av = seq.avail[t]
        util = np.full((P, K), -np.inf)
        la = last.get(sk)
        for a in range(K):
            if av[a]:
                util[:, a] = beta * Q.get(keys[a], np.full(P, Q0)) + (kappa if la == a else 0.0)
        m = util.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(util - m).sum(axis=1))
        out[t] = util - lse[:, None]
        last[sk] = int(seq.actions[t])
        prev_keys = keys
    return out


def chosen_loglik(seq: EncodedSeq, lp: np.ndarray) -> np.ndarray:
    """``[T, P]`` log-probability of the observed action, zero on unscored steps."""
    ll = lp[np.arange(seq.T), :, seq.actions]
    return np.where(seq.scored[:, None], ll, 0.0)


class HBCog:
    name = "hb_cog"
    adaptive = True

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.k_max = k_max
        self.spec = _Spec(config)
        self.theta = particle_grid()
        self.log_pi = np.full(self.theta.shape[0], -np.log(self.theta.shape[0]))

    _CACHE: ClassVar[dict[str, np.ndarray]] = {}

    def _dev_loglik(self, train: list[EncodedSeq]) -> np.ndarray:
        h = hashlib.sha256()
        for s in train:
            h.update(f"{s.key}:{s.T}".encode())
            h.update(np.ascontiguousarray(s.x).tobytes())
        h.update(repr((self.spec.state, self.spec.options, self.spec.reset)).encode())
        key = h.hexdigest()
        if key not in HBCog._CACHE:
            HBCog._CACHE[key] = np.stack(
                [chosen_loglik(s, run_particles(s, self.theta, self.spec)).sum(axis=0) for s in train]
            )
        return HBCog._CACHE[key]

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        t0 = time.process_time()
        Lmat = self._dev_loglik(train)  # [N, P]
        N, P = Lmat.shape
        c = float(self.config.get("c", 0.1))
        log_pi = np.full(P, -np.log(P))
        ll_hist = []
        for _it in range(int(self.config.get("em_iters", 300))):
            a = Lmat + log_pi[None, :]
            lse = _logsumexp(a, axis=1)
            r = np.exp(a - lse[:, None])
            pi = (r.sum(axis=0) + c) / (N + c * P)
            new = np.log(pi)
            ll_hist.append(float(lse.sum()))
            if np.max(np.abs(new - log_pi)) < 1e-8:
                log_pi = new
                break
            log_pi = new
        self.log_pi = log_pi
        pi = np.exp(log_pi)
        mean_theta = (pi[:, None] * self.theta).sum(axis=0)
        return {
            "em_iterations": len(ll_hist),
            "dev_marginal_loglik": ll_hist[-1] if ll_hist else None,
            "n_particles": P,
            "prior_mean_theta": dict(
                zip(("alpha", "rho", "beta", "kappa", "delta"), mean_theta.round(4).tolist(), strict=True)
            ),
            "prior_effective_particles": float(1.0 / (pi**2).sum()),
            "cpu_s": round(time.process_time() - t0, 2),
        }

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        tau = float(self.config.get("tau", 1.0))
        lp = run_particles(seq, self.theta, self.spec)  # [T, P, K]
        T = seq.T
        if mode == "frozen":
            logw = np.repeat(self.log_pi[None, :], T, axis=0)
        elif donor is not None:
            ld = chosen_loglik(donor, run_particles(donor, self.theta, self.spec)).sum(axis=0)
            logw = np.repeat((self.log_pi + tau * ld)[None, :], T, axis=0)
        else:
            ll = chosen_loglik(seq, lp)
            cum = np.concatenate(
                [np.zeros((1, ll.shape[1])), np.cumsum(ll, axis=0)[:-1]], axis=0
            )  # steps < t
            logw = self.log_pi[None, :] + tau * cum
        logw = logw - _logsumexp(logw, axis=1)[:, None]
        out = _logsumexp(logw[:, :, None] + lp, axis=1)  # [T, K]
        return np.where(seq.avail, out, -np.inf)

    def param_count(self) -> int:
        return int(self.theta.shape[0] - 1)

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": int(self.theta.shape[0] - 1),
            "note": "population params = particle mixing weights; per person = posterior over the same particles",
            "config": self.config,
        }


def _build(config: dict[str, Any], k: int, d: int) -> HBCog:
    return HBCog(config, k, d)


register_model(
    ModelSpec(
        "hb_cog",
        _build,
        "baselines",
        "comparator",
        {"c": [0.01, 0.1, 1.0], "tau": [0.5, 1.0]},
        adaptive=True,
        note="hierarchical Bayesian delta-rule learner: EM population prior over 600 particles, online per-person posterior",
    )
)
