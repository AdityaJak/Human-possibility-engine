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
from hpe.population.core.adaptation import AdaptiveBase
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
        mu, logvar, _ = self.run(x, None)
        return mu, logvar

    def run(
        self, x: torch.Tensor, h0: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, h_n = self.rnn(self.drop(x * self.keep), h0)
        out = self.head(h)
        return torch.sigmoid(out[..., : self.k]), out[..., self.k :].clamp(-9.0, 2.0), h_n


def world_loss(m: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
    assert isinstance(m, WorldGRU)
    L = Layout(m.k)
    mu, logvar = m(b.x)
    y = b.x[:, 1:, L.prev_outcomes]
    obs = b.x[:, 1:, L.prev_observed] > 0.5
    mu, logvar = mu[:, :-1], logvar[:, :-1]
    nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp() + np.log(2 * np.pi))
    return torch.where(obs, nll, torch.zeros_like(nll)).sum(), max(int(obs.sum()), 1)


def delta_rows(
    x: np.ndarray, k: int, alpha: float, q: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delta-rule outcome means and a count-based uncertainty for rows ``x`` (continuing from q, n)."""
    L = Layout(k)
    mu, sd = np.zeros((x.shape[0], k)), np.zeros((x.shape[0], k))
    for t in range(x.shape[0]):
        if x[t, L.block_start] > 0.5:
            q, n = np.full(k, 0.5), np.zeros(k)
        obs = x[t, L.prev_observed] > 0.5
        out = x[t, L.prev_outcomes].astype(np.float64)
        q = np.where(obs, q + alpha * (out - q), q)
        n = 0.9 * n + obs
        mu[t], sd[t] = q, 1.0 / np.sqrt(1.0 + n)
    return mu, sd, q, n


def policy_rows(
    x: np.ndarray, avail: np.ndarray, mu: np.ndarray, sd: np.ndarray, k_eff: int, trace0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """F [n, K, d] with d = 4 + (k_eff − 1) for rows ``x``; the choice trace continues from ``trace0``."""
    k = avail.shape[1]
    L = Layout(k)
    av = avail.astype(np.float64)
    kt = av.sum(1, keepdims=True)
    mu_c = (mu - (mu * av).sum(1, keepdims=True) / kt) * av
    sd_c = (sd - (sd * av).sum(1, keepdims=True) / kt) * av
    prev = x[:, L.prev_action].astype(np.float64)
    trace = np.zeros_like(prev)
    tr = trace0.copy()
    for t in range(x.shape[0]):
        if x[t, L.block_start] > 0.5:
            tr = np.zeros(k)
        tr = TRACE_DECAY * tr + (1 - TRACE_DECAY) * prev[t]
        trace[t] = tr
    bias = np.zeros((x.shape[0], k, max(k_eff - 1, 0)))
    for j in range(1, k_eff):
        bias[:, j, j - 1] = 1.0
    F = np.concatenate([mu_c[..., None], sd_c[..., None], prev[..., None], trace[..., None], bias], axis=2)
    return F, tr


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


def laplace_step(
    F: np.ndarray, a: int, avail: np.ndarray, theta: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One online Laplace / extended-Kalman update of a multinomial-logit posterior N(θ, P⁻¹)."""
    p = _probs(F, theta, avail)
    y = np.zeros(F.shape[0])
    y[a] = 1.0
    g = F.T @ (p - y)
    ef = F.T @ p
    P = P + (F.T * p) @ F - np.outer(ef, ef)
    return theta - np.linalg.solve(P, g), P


class F2WorldPolicy(AdaptiveBase):
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
        self.update_every = int(self.config.get("update_every", 1))
        self.k_eff = k_max
        d = 4 + k_max - 1
        self.m0 = np.zeros(d)
        self.P0 = np.eye(d)

    def _carry0(self) -> dict[str, np.ndarray]:
        k = self.k_max
        return {
            "h": np.zeros(self.hidden, dtype=np.float32),
            "q": np.full(k, 0.5),
            "n": np.zeros(k),
            "trace": np.zeros(k),
        }

    def _feature_rows(
        self, x: np.ndarray, avail: np.ndarray, carry: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Policy features for consecutive rows, continuing the world model and trace from ``carry``."""
        new = dict(carry)
        if self.world == "delta":
            mu, sd, new["q"], new["n"] = delta_rows(
                x, self.k_max, float(self.config.get("delta_alpha", 0.3)), carry["q"], carry["n"]
            )
        else:
            with torch.no_grad():
                m, lv, h_n = self.wm.run(torch.from_numpy(x)[None], torch.from_numpy(carry["h"])[None, None])
            mu, sd = m[0].double().numpy(), np.exp(0.5 * lv[0].double().numpy())
            new["h"] = h_n[0, 0].numpy().copy()
        F, new["trace"] = policy_rows(x, avail, mu, sd, self.k_eff, carry["trace"])
        return F, new

    def _features(self, seq: EncodedSeq) -> np.ndarray:
        return self._feature_rows(seq.x, seq.avail, self._carry0())[0]

    # ------------------------------------------------------------------ person-level state
    def init_state(self) -> dict[str, Any]:
        return {"n_proc": 0, "carry": self._carry0(), "theta": self.m0.copy(), "P": self.P0.copy()}

    def update_state(self, state: Any, history: EncodedSeq) -> dict[str, Any]:
        st = state if int(state["n_proc"]) <= history.T else self.init_state()
        n0 = int(st["n_proc"])
        if history.T == n0:
            return st
        F, carry = self._feature_rows(history.x[n0:], history.avail[n0:], st["carry"])
        theta, P = st["theta"].copy(), st["P"].copy()
        for j in range(history.T - n0):
            s = n0 + j
            if history.scored[s]:
                theta, P = laplace_step(F[j], int(history.actions[s]), history.avail[s], theta, P)
        return {"n_proc": history.T, "carry": carry, "theta": theta, "P": P}

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        F = self._features(seq)
        p = np.zeros((seq.T, seq.k_max))
        bounds = [s for s, _ in states] + [seq.T]
        for (s, st), e in zip(states, bounds[1:], strict=True):
            for t in range(s, e):
                p[t] = _probs(F[t], st["theta"], seq.avail[t])
        return common.finalize_logp(p, seq.avail)

    def twin_config(self) -> dict[str, Any]:
        return common.twin_config(self.config)

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        t0 = time.process_time()
        log: dict[str, Any] = {"world": self.world}
        self.k_eff = int(max(s.avail.sum(1).max() for s in train))
        if self.world == "gru":
            bc = common.backbone_config(self.config)
            digest = common.seq_digest(train)
            key = ("f2_world", digest, self.k_max, self.d_in, tuple(sorted(bc.items())))
            wseed = derive_seed(bc["backbone_seed"], "f2_world", digest, repr(sorted(bc.items()))) % (
                2**31 - 1
            )
            nets.setup_torch(wseed, threads, "f2_world")
            self.wm = WorldGRU(self.d_in, self.hidden, self.k_max, bc["dropout"])
            cached = common.cache_get(key)
            if cached is not None:
                self.wm.load_state_dict(cached[0])
                log["world_model"] = {**cached[1], "cached": True}
            else:
                t1 = time.process_time()
                wlog = nets.train(
                    self.wm,
                    world_loss,
                    train,
                    seed=wseed,
                    key="f2_world",
                    budget_s=budget_s,
                    **common.trainer_kwargs(self.config),
                )
                wlog["world_cpu_s"] = round(time.process_time() - t1, 2)
                common.cache_put(key, self.wm, wlog)
                log["world_model"] = {**wlog, "cached": False}
            self.wm.eval()
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
            cpu_s_incl_backbone=round(
                time.process_time()
                - t0
                + (log["world_model"]["world_cpu_s"] if log.get("world_model", {}).get("cached") else 0.0),
                2,
            ),
        )
        return log

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
            "update_every": self.update_every,
            "prior_mean": self.m0.tolist(),
            "prior_sd": np.sqrt(np.diag(np.linalg.inv(self.P0))).tolist(),
            "config": self.config,
        }
