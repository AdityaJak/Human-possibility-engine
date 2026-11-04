"""F2 ``f2_world_policy`` — population world model + individual policy with an online Bayesian posterior.

Definition (docs/phase3/POPULATION_ADAPTATION_ARCHITECTURES.md §F2).

World model (population, trained on *observed outcomes only*, never on choices): a GRU over encoding v3
with the choice slices (previous action, instructed action) zeroed predicts, before trial t, a Gaussian
over the outcome of every option, (μ_k(t), σ_k(t)); loss = Gaussian NLL on the outcomes revealed after
trial t (all options under full feedback, the chosen one under bandit feedback). It is a predictive model
of the task's outcome dynamics, not a causal model of the environment. Ablation ``delta``: μ from a delta
rule per option (rate α, reset at block start), σ = 1/sqrt(1 + decayed observation count).

Individual policy: logits_k(t) = θᵀ f_k(t), f_k = [μ_k − μ̄, σ_k − σ̄, 1[prev action = k], choice trace_k,
one-hot option bias (K−1)], i.e. inverse temperature, uncertainty bonus, perseveration, choice kernel and
side bias. Population prior θ ~ N(m₀, S₀): m₀ = pooled maximum-likelihood fit on dev; S₀ = scale ·
PSD(Cov(per-person MAP) − mean per-person posterior covariance) + 1e-4 I (method-of-moments empirical
Bayes: the spread of individual estimates minus their estimation noise). Online per person: Laplace /
extended-Kalman update after each free choice (one Newton step; precision accumulates the Fisher
information). ``mode="frozen"``: θ = m₀ for everyone (population-only policy). ``donor``: θ after the
donor's whole sequence, held fixed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.core.rng import derive_seed
from hpe.population.candidates import common
from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq, Layout

TRACE_DECAY = 0.7


class WorldGRU(nn.Module):
    def __init__(self, d_in: int, hidden: int, k: int, dropout: float = 0.0) -> None:
        super().__init__()
        L = Layout(k)
        keep = torch.ones(d_in)
        keep[L.prev_action] = 0.0
        keep[L.instructed_action] = 0.0
        keep[L.instructed] = 0.0
        self.register_buffer("keep", keep)
        self.k = k
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.rnn(self.drop(x * self.keep))
        out = self.head(h)
        return torch.sigmoid(out[..., : self.k]), out[..., self.k :].clamp(-9.0, 2.0)


def world_loss(m: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
    assert isinstance(m, WorldGRU)
    L = Layout(m.k)
    mu, logvar = m(b.x)
    y = b.x[:, 1:, L.prev_outcomes]
    obs = b.x[:, 1:, L.prev_observed] > 0.5
    mu, logvar = mu[:, :-1], logvar[:, :-1]
    nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp() + np.log(2 * np.pi))
    return torch.where(obs, nll, torch.zeros_like(nll)).sum(), max(int(obs.sum()), 1)


def delta_world(seq: EncodedSeq, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    L = Layout(seq.k_max)
    K = seq.k_max
    mu, sd = np.zeros((seq.T, K)), np.zeros((seq.T, K))
    q, n = np.full(K, 0.5), np.zeros(K)
    for t in range(seq.T):
        if seq.x[t, L.block_start] > 0.5:
            q, n = np.full(K, 0.5), np.zeros(K)
        obs = seq.x[t, L.prev_observed] > 0.5
        out = seq.x[t, L.prev_outcomes].astype(np.float64)
        q = np.where(obs, q + alpha * (out - q), q)
        n = 0.9 * n + obs
        mu[t], sd[t] = q, 1.0 / np.sqrt(1.0 + n)
    return mu, sd


def policy_features(seq: EncodedSeq, mu: np.ndarray, sd: np.ndarray, k_eff: int) -> np.ndarray:
    """F [T, K, d] with d = 4 + (k_eff − 1)."""
    L = Layout(seq.k_max)
    av = seq.avail.astype(np.float64)
    kt = av.sum(1, keepdims=True)
    mu_c = (mu - (mu * av).sum(1, keepdims=True) / kt) * av
    sd_c = (sd - (sd * av).sum(1, keepdims=True) / kt) * av
    prev = seq.x[:, L.prev_action].astype(np.float64)
    trace = np.zeros_like(prev)
    tr = np.zeros(seq.k_max)
    for t in range(seq.T):
        if seq.x[t, L.block_start] > 0.5:
            tr = np.zeros(seq.k_max)
        tr = TRACE_DECAY * tr + (1 - TRACE_DECAY) * prev[t]
        trace[t] = tr
    bias = np.zeros((seq.T, seq.k_max, max(k_eff - 1, 0)))
    for j in range(1, k_eff):
        bias[:, j, j - 1] = 1.0
    return np.concatenate([mu_c[..., None], sd_c[..., None], prev[..., None], trace[..., None], bias], axis=2)


def _probs(F: np.ndarray, theta: np.ndarray, avail: np.ndarray) -> np.ndarray:
    logits = F @ theta
    return np.exp(common.log_softmax_np(logits, avail))


def pooled_fit(
    Fs: np.ndarray,
    av: np.ndarray,
    a: np.ndarray,
    prior_prec: np.ndarray,
    prior_mean: np.ndarray,
    iters: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP of a multinomial logit with fixed features (damped Newton). Returns (θ, Hessian)."""
    d = Fs.shape[2]
    theta = prior_mean.copy()
    Y = np.zeros(av.shape)
    Y[np.arange(a.shape[0]), a] = 1.0
    H = prior_prec.copy()

    def obj(th: np.ndarray) -> float:
        lp = common.log_softmax_np(Fs @ th, av)
        dv = th - prior_mean
        return float(-lp[np.arange(a.shape[0]), a].sum() + 0.5 * dv @ prior_prec @ dv)

    f0 = obj(theta)
    for _ in range(iters):
        p = np.exp(common.log_softmax_np(Fs @ theta, av))
        g = np.einsum("nkd,nk->d", Fs, p - Y) + prior_prec @ (theta - prior_mean)
        ef = np.einsum("nkd,nk->nd", Fs, p)
        H = np.einsum("nkd,nk,nke->de", Fs, p, Fs) - ef.T @ ef + prior_prec
        step = np.linalg.solve(H + 1e-9 * np.eye(d), g)
        s = 1.0
        while s > 1e-4:
            cand = theta - s * step
            f1 = obj(cand)
            if f1 <= f0:
                break
            s *= 0.5
        if s <= 1e-4:
            break
        theta, done = cand, f0 - f1 < 1e-7
        f0 = f1
        if done:
            break
    return theta, H


def online_laplace(
    F: np.ndarray, a: np.ndarray, scored: np.ndarray, avail: np.ndarray, m0: np.ndarray, P0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predictive probabilities [T, K] before each trial and the final posterior mean."""
    T, K, _ = F.shape
    theta, P = m0.copy(), P0.copy()
    out = np.zeros((T, K))
    for t in range(T):
        p = _probs(F[t], theta, avail[t])
        out[t] = p
        if scored[t]:
            y = np.zeros(K)
            y[a[t]] = 1.0
            g = F[t].T @ (p - y)
            ef = F[t].T @ p
            Hs = (F[t].T * p) @ F[t] - np.outer(ef, ef)
            P = P + Hs
            theta = theta - np.linalg.solve(P, g)
    return out, theta


class F2WorldPolicy:
    adaptive = True

    def __init__(
        self,
        config: dict[str, Any],
        k_max: int,
        d_in: int,
        *,
        world: str = "gru",
        name: str = "f2_world_policy",
    ) -> None:
        self.name, self.world = name, world
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.hidden = int(self.config.get("hidden", 32))
        self.wm = WorldGRU(d_in, self.hidden, k_max, float(self.config.get("dropout", 0.0)))
        self.k_eff = k_max
        d = 4 + k_max - 1
        self.m0 = np.zeros(d)
        self.P0 = np.eye(d)

    def _world(self, seq: EncodedSeq) -> tuple[np.ndarray, np.ndarray]:
        if self.world == "delta":
            return delta_world(seq, float(self.config.get("delta_alpha", 0.3)))
        b = nets.Batch([seq])
        with torch.no_grad():
            mu, logvar = self.wm(b.x)
        return mu[0].double().numpy()[: seq.T], np.exp(0.5 * logvar[0].double().numpy()[: seq.T])

    def _features(self, seq: EncodedSeq) -> np.ndarray:
        mu, sd = self._world(seq)
        return policy_features(seq, mu, sd, self.k_eff)

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        t0 = time.process_time()
        log: dict[str, Any] = {"world": self.world}
        self.k_eff = int(max(s.avail.sum(1).max() for s in train))
        if self.world == "gru":
            wseed = derive_seed(seed, "f2_world") % (2**31 - 1)
            nets.setup_torch(wseed, threads, self.name, repr(sorted(self.config.items())))
            self.wm = WorldGRU(self.d_in, self.hidden, self.k_max, float(self.config.get("dropout", 0.0)))
            log["world_model"] = nets.train(
                self.wm,
                world_loss,
                train,
                seed=wseed,
                key="f2_world",
                budget_s=budget_s,
                **common.trainer_kwargs(self.config),
            )
        feats = [self._features(s) for s in train]
        d = feats[0].shape[2]
        Fs = np.concatenate([f[s.scored] for f, s in zip(feats, train, strict=True)])
        av = np.concatenate([s.avail[s.scored] for s in train])
        a = np.concatenate([s.actions[s.scored] for s in train])
        weak = np.eye(d) * 1e-2
        m0, _ = pooled_fit(Fs, av, a, weak, np.zeros(d))
        maps, covs = [], []
        for f, s in zip(feats, train, strict=True):
            th, H = pooled_fit(f[s.scored], s.avail[s.scored], s.actions[s.scored], np.eye(d) * 0.1, m0)
            maps.append(th)
            covs.append(np.linalg.inv(H))
        M = np.array(maps)
        spread = np.cov(M.T) - np.mean(covs, axis=0)
        ev, evec = np.linalg.eigh((spread + spread.T) / 2)
        S0 = float(self.config.get("prior_scale", 1.0)) * (
            evec * np.clip(ev, 0.0, None)
        ) @ evec.T + 1e-4 * np.eye(d)
        self.m0, self.P0 = m0, np.linalg.inv(S0)
        log.update(
            m0=[round(v, 4) for v in m0.tolist()],
            prior_sd=[round(float(v), 4) for v in np.sqrt(np.diag(S0))],
            map_sd=[round(float(v), 4) for v in M.std(0)],
            n_policy_trials=int(a.shape[0]),
            cpu_s=round(time.process_time() - t0, 2),
        )
        return log

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        self.wm.eval()
        F = self._features(seq)
        if mode == "frozen":
            p = np.stack([_probs(F[t], self.m0, seq.avail[t]) for t in range(seq.T)])
        elif donor is not None:
            Fd = self._features(donor)
            _, th = online_laplace(Fd, donor.actions, donor.scored, donor.avail, self.m0, self.P0)
            p = np.stack([_probs(F[t], th, seq.avail[t]) for t in range(seq.T)])
        else:
            p, _ = online_laplace(F, seq.actions, seq.scored, seq.avail, self.m0, self.P0)
        return common.finalize_logp(p, seq.avail)

    def param_count(self) -> int:
        d = self.m0.shape[0]
        wm = nets.n_params(self.wm) if self.world == "gru" else 1
        return int(wm + d + d * (d + 1) // 2)

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": int(self.m0.shape[0]),
            "per_person_state": "Gaussian posterior over policy weights (online Laplace)",
            "world": self.world,
            "prior_mean": self.m0.tolist(),
            "prior_sd": np.sqrt(np.diag(np.linalg.inv(self.P0))).tolist(),
            "config": self.config,
        }
