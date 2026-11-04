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

Optional model-based hybrid (``mb: true``; meant for two_step): transitions from a (state, action) of a step
without feedback to the next step's state are counted; the action's value becomes
``w Σ_s' p(s'|s,a) max_a' Q[s', a'] + (1 − w) Q[s, a]`` with ``w`` a particle dimension {0, .5, 1}.

Hierarchy: a fixed grid of ``P`` particles (5 α × 3 ρ × 5 β × 4 κ × 2 δ = 600; × 3 w with ``mb``). Population prior = mixing weights
``π`` over the particles, fitted on the dev participants by EM (non-parametric maximum likelihood of the mixing
distribution, with ``c`` pseudo-counts per particle). Person-level state (``core.adaptation.AdaptiveBase``; the
harness owns the adapt / frozen / donor modes): the posterior log-weights over particles,
``log π + τ Σ_{s in history, scored} log p_θ(a_s)``, recomputed every ``m`` trials (config ``m``, default 5) from
the history the harness passes; the prediction is the posterior-predictive mixture. Each particle's own value
learning runs in context over the sequence (causal), so ``frozen`` (= prior weights throughout) still learns values.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from typing import Any, ClassVar

import numpy as np

from hpe.population.core.adaptation import AdaptiveBase
from hpe.population.core.encoding import EncodedSeq, Layout
from hpe.population.core.interface import ModelSpec, register_model

ALPHA = (0.05, 0.15, 0.3, 0.5, 0.8)
RHO = (0.0, 0.5, 1.0)
BETA = (1.0, 3.0, 6.0, 10.0, 20.0)
KAPPA = (-0.5, 0.0, 0.5, 1.5)
DELTA = (0.0, 0.1)
W_MB = (0.0, 0.5, 1.0)
Q0 = 0.5


def particle_grid(mb: bool = False) -> np.ndarray:
    """Particles (α, ρ, β, κ, δ, w_mb); w_mb ∈ {0, .5, 1} only for the model-based hybrid, else 0."""
    dims = (ALPHA, RHO, BETA, KAPPA, DELTA, W_MB if mb else (0.0,))
    return np.array(list(itertools.product(*dims)), dtype=np.float64)


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
        self.mb = bool(config.get("mb", False))


def run_particles(seq: EncodedSeq, theta: np.ndarray, spec: _Spec) -> np.ndarray:
    """Per-particle log-probabilities ``[T, P, K]`` (``-inf`` on unavailable actions); row t uses steps < t."""
    L = Layout(seq.k_max)
    T, K = seq.T, seq.k_max
    P = theta.shape[0]
    alpha, rho, beta, kappa, delta = (theta[:, i] for i in range(5))
    w_mb = theta[:, 5] if theta.shape[1] > 5 else np.zeros(P)
    Q: dict[tuple, np.ndarray] = {}
    last: dict[tuple, int] = {}
    trans: dict[
        tuple, dict[tuple, int]
    ] = {}  # (state, action) -> counts of the next state (model-based part)
    avail_of: dict[tuple, tuple[int, ...]] = {}
    out = np.full((T, P, K), -np.inf)
    prev_keys: list[tuple] | None = None
    prev_sk: tuple | None = None
    pending: tuple | None = None
    for t in range(T):
        x = seq.x[t].astype(np.float64)
        sk = tuple(np.round(x[list(spec.state)], 3)) if spec.state else ()
        if t > 0 and spec.reset and x[L.block_start] > 0.5:
            Q.clear()
            last.clear()
            trans.clear()
            prev_keys, prev_sk, pending = None, None, None
        if spec.mb and t > 0 and prev_sk is not None and x[L.prev_feedback] < 0.5 and sk != prev_sk:
            nxt = trans.setdefault((prev_sk, int(seq.actions[t - 1])), {})
            nxt[sk] = nxt.get(sk, 0) + 1
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
        keys: list[tuple] = []
        for a in range(K):
            if spec.options is not None and a < len(spec.options):
                keys.append((sk, tuple(np.round(x[list(spec.options[a])], 3))))
            else:
                keys.append((sk, a))
        av = seq.avail[t]
        avail_of[sk] = tuple(int(a) for a in np.flatnonzero(av))
        util = np.full((P, K), -np.inf)
        la = last.get(sk)
        for a in range(K):
            if av[a]:
                v = Q.get(keys[a], np.full(P, Q0))
                succ = trans.get((sk, a)) if spec.mb else None
                if succ:
                    tot = sum(succ.values())
                    vmb = np.zeros(P)
                    for s2, n2 in sorted(succ.items()):
                        best = np.max(
                            [Q.get((s2, a2), np.full(P, Q0)) for a2 in avail_of.get(s2, (0,))], axis=0
                        )
                        vmb = vmb + (n2 / tot) * best
                    v = w_mb * vmb + (1.0 - w_mb) * v
                util[:, a] = beta * v + (kappa if la == a else 0.0)
        m = util.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(util - m).sum(axis=1))
        out[t] = util - lse[:, None]
        last[sk] = int(seq.actions[t])
        prev_keys, prev_sk = keys, sk
    return out


def chosen_loglik(seq: EncodedSeq, lp: np.ndarray) -> np.ndarray:
    """``[T, P]`` log-probability of the observed action, zero on unscored steps."""
    ll = lp[np.arange(seq.T), :, seq.actions]
    return np.where(seq.scored[:, None], ll, 0.0)


class HBCog(AdaptiveBase):
    name = "hb_cog"
    adaptive = True

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.update_every = max(1, int(config.get("m", 5)))
        self.k_max = k_max
        self.spec = _Spec(config)
        self.theta = particle_grid(self.spec.mb)
        self.log_pi = np.full(self.theta.shape[0], -np.log(self.theta.shape[0]))

    _CACHE: ClassVar[dict[str, np.ndarray]] = {}

    def _dev_loglik(self, train: list[EncodedSeq]) -> np.ndarray:
        h = hashlib.sha256()
        for s in train:
            h.update(f"{s.key}:{s.T}".encode())
            h.update(np.ascontiguousarray(s.x).tobytes())
        h.update(repr((self.spec.state, self.spec.options, self.spec.reset, self.spec.mb)).encode())
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
                zip(
                    ("alpha", "rho", "beta", "kappa", "delta", "w_mb"),
                    mean_theta.round(4).tolist(),
                    strict=True,
                )
            ),
            "prior_effective_particles": float(1.0 / (pi**2).sum()),
            "cpu_s": round(time.process_time() - t0, 2),
        }

    # ---- person-level state (harness-owned modes, core.adaptation.AdaptiveBase) ----------------------
    def init_state(self) -> Any:
        return self.log_pi.copy()

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        """Posterior log-weights from the history's scored choices: log π + τ Σ log p_θ(a_s)."""
        tau = float(self.config.get("tau", 1.0))
        ll = chosen_loglik(history, run_particles(history, self.theta, self.spec)).sum(axis=0)
        return self.log_pi + tau * ll

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        lp = run_particles(seq, self.theta, self.spec)  # [T, P, K]; each particle learns in context (causal)
        out = np.full((seq.T, seq.k_max), -np.inf)
        starts = [st for st, _ in states] + [seq.T]
        for (s, logw), e in zip(states, starts[1:], strict=False):
            if e <= s:
                continue
            lw = logw - _logsumexp(logw[None, :], axis=1)[0]
            out[s:e] = _logsumexp(lw[None, :, None] + lp[s:e], axis=1)
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
