"""Conventional individual-adaptation comparators on a population GRU backbone (thesis §4, pool items 4–7, and
the Gate B2 adapter set of ``docs/phase3/ADAPTATION_BENCHMARKS.md`` §3).

Every adapter is a ``core.adaptation.AdaptiveBase``: the **harness** owns the adaptation modes (critic finding
G-1). The model only defines

* ``update_every`` (= config ``m``): the harness refits the person-level state at rows ``m, 2m, …``;
* ``init_state()``: the population prior (``None`` = population parameters unchanged);
* ``update_state(state, history)``: the new person-level state from ``history`` — a truncated, re-keyed copy of
  the first ``j·m`` trials (own trials in ``adapt``; the donor's, length-matched, in the A2 control);
* ``predict_with_states(seq, states)``: rows ``[start_i, start_{i+1})`` predicted with ``states[i]`` while the
  recurrent state runs over ``x[:t+1]`` (in-context dynamics, causal).

Each refit starts from the population (or meta-learned) parameters, so no state drifts between updates
(``bayes_head`` warm-starts Newton's method from the previous MAP, which does not change the optimum).

* ``ft_head``      — output layer fine-tuned by ``n`` Adam steps (``lr``) on the mean NLL of the history's scored
                     trials (+ optional L2 pull ``l2`` to the population head).
* ``ft_full``      — all weights, BPTT over the history, gradient clip 1, L2 pull ``l2/2·‖θ − θ_pop‖²``
                     (dynamic evaluation).
* ``bayes_head``   — Gaussian prior ``N(W_pop, σ² I)`` on ``W = [weights | bias]``; MAP by damped Newton;
                     ``predictive="laplace"`` uses the Laplace posterior and the mean-field softmax correction
                     ``μ_k / sqrt(1 + π v_k / 8)``. ``prior_var="eb"``: σ² chosen by type-II maximum likelihood
                     (Laplace evidence) on the backbone's held-out validation participants (dev).
* ``exemplar_mix`` — ``p = (1 − w) p_pop + w p_kern``; the state is the person's memory of (hidden state, choice,
                     position) pairs; ``p_kern(a) ∝ Σ_s k(h_s, h_t) γ^{t−1−s} 1[a_s = a] + 1/K`` with a Gaussian
                     kernel (bandwidth ``bw`` × the median squared within-person distance on dev).
* ``reptile_gru``  — first-order meta-learning (Reptile; Nichol, Achiam & Schulman, 2018) of the GRU weights from
                     the population backbone: task = one dev participant; inner loop = ``n`` Adam steps on growing
                     prefixes of the sequence; ``θ ← θ + ε (φ̄ − θ)``. Test: ``ft_full`` from the meta-learned
                     weights (no L2).
* ``cavia_gru``    — CAVIA (Zintgraf et al., 2019): a per-person context vector ``c ∈ R^d`` (``c_dim``) fed to the
                     GRU input at every step and to the head; shared weights initialised from the backbone with the
                     context weights at 0 (so ``c = 0`` reproduces ``pop_gru``) and meta-trained through the inner
                     loop (second order; ``c ← c − α ∇_c NLL(prefix)``, ``n_inner`` steps, fixed ``α``) to predict
                     the suffix. Test: the same inner loop on the history, from ``c = 0``.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.core.rng import stream
from hpe.population.baselines.backbone import (
    backbone_config,
    hidden_states,
    masked_logsoftmax_np,
    train_backbone,
)
from hpe.population.core import nets
from hpe.population.core.adaptation import AdaptiveBase
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import ModelSpec, register_model

States = list[tuple[int, Any]]


def _segments(states: States, T: int) -> list[tuple[int, int, Any]]:
    starts = [s for s, _ in states] + [T]
    return [(s, e, st) for (s, st), e in zip(states, starts[1:], strict=False) if e > s]


class _BackboneAdapter(AdaptiveBase):
    adaptive = True
    name = "adapter"

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.update_every = max(1, int(config.get("m", 25)))
        self.net: nets.PopRNN | None = None
        self.fit_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        bb = backbone_config(self.config)
        bseed = int(self.config.get("backbone_seed", seed))
        t0 = time.process_time()
        self.net, log = train_backbone(
            train, bb, bseed, d_in=self.d_in, k_max=self.k_max, threads=threads, budget_s=budget_s
        )
        extra = self._fit_extra(train, seed=seed, budget_s=max(1.0, budget_s - (time.process_time() - t0)))
        self.fit_info = {**log, **extra}
        return self.fit_info

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        return {}

    def _require_net(self) -> nets.PopRNN:
        if self.net is None:
            raise RuntimeError(f"{self.name}: predict before fit")
        return self.net

    def init_state(self) -> Any:
        return None

    def twin_config(self) -> dict[str, Any] | None:
        return backbone_config(self.config)

    def param_count(self) -> int:
        return nets.n_params(self._require_net())

    def per_person_params(self) -> int:
        return 0

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": self.per_person_params(),
            "update_every": self.update_every,
            "config": self.config,
            "backbone_cpu_s": self.fit_info.get("backbone_cpu_s"),
            "backbone_cached": self.fit_info.get("backbone_cached"),
        }


def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


# ============================================================================ head fine-tuning
class FTHead(_BackboneAdapter):
    name = "ft_head"

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        idx = np.flatnonzero(history.scored)
        if idx.size == 0:
            return state
        H, _ = hidden_states(self._require_net(), history)
        net = self._require_net()
        W0 = net.head.weight.detach().double()
        b0 = net.head.bias.detach().double()
        W = W0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        n, lr, l2 = (
            int(self.config.get("n", 10)),
            float(self.config.get("lr", 0.01)),
            float(self.config.get("l2", 0.0)),
        )
        opt = torch.optim.Adam([W, b], lr=lr)
        Ht, at, avt = _tensor(H[idx]), _tensor(history.actions[idx]), _tensor(history.avail[idx])
        for _ in range(n):
            logp = nets.masked_log_softmax(Ht @ W.T + b, avt)
            loss = -logp.gather(1, at.unsqueeze(1)).mean()
            if l2:
                loss = loss + 0.5 * l2 * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        return (W.detach().numpy().copy(), b.detach().numpy().copy())

    def predict_with_states(self, seq: EncodedSeq, states: States) -> np.ndarray:
        H, out = hidden_states(self._require_net(), seq)
        for s, e, st in _segments(states, seq.T):
            if st is not None:
                W, b = st
                out[s:e] = masked_logsoftmax_np(H[s:e] @ W.T + b, seq.avail[s:e])
        return out

    def per_person_params(self) -> int:
        net = self._require_net()
        return int(net.head.weight.numel() + net.head.bias.numel())


# ============================================================================ full fine-tuning
def nn_clip(net: torch.nn.Module, clip: float = 1.0) -> None:
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)


def _finetune_full(
    net0: nn.Module, seq: EncodedSeq, *, n: int, lr: float, l2: float
) -> dict[str, torch.Tensor] | None:
    """State dict of ``net0`` after ``n`` Adam steps on the mean NLL of all scored steps of ``seq``."""
    b = nets.Batch([seq])
    k = int(b.mask.sum())
    if k == 0 or n <= 0:
        return None
    net = copy.deepcopy(net0)
    net.eval()
    p0 = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n):
        logp, _ = net(b.x, b.avail)
        lp = logp.gather(-1, b.a.unsqueeze(-1)).squeeze(-1)
        loss = -torch.where(b.mask, lp, torch.zeros_like(lp)).sum() / k
        if l2:
            loss = loss + 0.5 * l2 * sum(
                ((p - q) ** 2).sum() for p, q in zip(net.parameters(), p0, strict=True)
            )
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        nn_clip(net)
        opt.step()
    return {kk: v.detach().clone() for kk, v in net.state_dict().items()}


@torch.no_grad()
def _logp_rows(net: nn.Module, seq: EncodedSeq, start: int, end: int, **kw: Any) -> np.ndarray:
    b = nets.Batch([seq])
    logp, _ = net(b.x[:, :end], b.avail[:, :end], **kw)
    out = logp[0, start:end].double().numpy()
    return np.where(seq.avail[start:end], out, -np.inf)


class FTFull(_BackboneAdapter):
    name = "ft_full"

    def _hp(self) -> tuple[int, float, float]:
        return (
            int(self.config.get("n", 5)),
            float(self.config.get("lr", 0.001)),
            float(self.config.get("l2", 0.0)),
        )

    def _base(self) -> nets.PopRNN:
        return self._require_net()

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        n, lr, l2 = self._hp()
        new = _finetune_full(self._base(), history, n=n, lr=lr, l2=l2)
        return state if new is None else new

    def predict_with_states(self, seq: EncodedSeq, states: States) -> np.ndarray:
        base = self._base()
        out = hidden_states(base, seq)[1]
        work = copy.deepcopy(base)
        work.eval()
        for s, e, st in _segments(states, seq.T):
            if st is not None:
                work.load_state_dict(st)
                out[s:e] = _logp_rows(work, seq, s, e)
        return out

    def per_person_params(self) -> int:
        return self.param_count()


# ============================================================================ Bayesian head
EB_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def _phi(H: np.ndarray) -> np.ndarray:
    return np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)


def _probs(Phi: np.ndarray, W: np.ndarray, avail: np.ndarray) -> np.ndarray:
    return np.exp(masked_logsoftmax_np(Phi @ W.T, avail))


def _hessian(A: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """``Σ_t A_t ⊗ φ_t φ_tᵀ`` as a ``(K·D) × (K·D)`` matrix, block (i, j) = ``Φᵀ diag(A[:, i, j]) Φ`` (BLAS)."""
    K, D = A.shape[1], Phi.shape[1]
    Hs = np.zeros((K * D, K * D))
    for i in range(K):
        for j in range(i, K):
            blk = (Phi * A[:, i, j][:, None]).T @ Phi
            Hs[i * D : (i + 1) * D, j * D : (j + 1) * D] = blk
            if j != i:
                Hs[j * D : (j + 1) * D, i * D : (i + 1) * D] = blk.T
    return Hs


def map_head(
    Phi: np.ndarray,
    a: np.ndarray,
    avail: np.ndarray,
    W0: np.ndarray,
    var: float,
    W_init: np.ndarray | None = None,
    iters: int = 25,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """MAP of the multinomial-logistic head under ``N(W0, var I)`` by damped Newton. Returns (W, Hessian, f)."""
    K, D = W0.shape
    W = (W0 if W_init is None else W_init).copy()
    Y = np.zeros((Phi.shape[0], K))
    Y[np.arange(Phi.shape[0]), a] = 1.0
    prec = 1.0 / var

    def objective(Wc: np.ndarray) -> float:
        lp = masked_logsoftmax_np(Phi @ Wc.T, avail)
        return float(-lp[np.arange(Phi.shape[0]), a].sum() + 0.5 * prec * ((Wc - W0) ** 2).sum())

    f = objective(W)
    Hs = np.eye(K * D) * prec
    for _ in range(iters):
        P = _probs(Phi, W, avail)
        G = (P - Y).T @ Phi + prec * (W - W0)
        A = P[:, :, None] * np.eye(K)[None] - P[:, :, None] * P[:, None, :]
        Hs = _hessian(A, Phi) + np.eye(K * D) * prec
        step = np.linalg.solve(Hs, G.reshape(-1)).reshape(K, D)
        t = 1.0
        while True:
            Wn = W - t * step
            fn = objective(Wn)
            if fn <= f + 1e-12 or t < 1e-4:
                break
            t *= 0.5
        done = abs(f - fn) < tol * max(1.0, abs(f))
        W, f = Wn, fn
        if done:
            break
    P = _probs(Phi, W, avail)
    A = P[:, :, None] * np.eye(K)[None] - P[:, :, None] * P[:, None, :]
    Hs = _hessian(A, Phi) + np.eye(K * D) * prec
    return W, Hs, f


class BayesHead(_BackboneAdapter):
    name = "bayes_head"

    def _W0(self) -> np.ndarray:
        net = self._require_net()
        return np.concatenate(
            [net.head.weight.detach().double().numpy(), net.head.bias.detach().double().numpy()[:, None]],
            axis=1,
        )

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        pv = self.config.get("prior_var", 0.1)
        if pv != "eb":
            self.var = float(pv)
            return {"prior_var": self.var}
        bseed = int(self.config.get("backbone_seed", seed))
        _, val = nets.split_train_val(train, bseed, "pop_gru")
        net = self._require_net()
        W0 = self._W0()
        K, D = W0.shape
        ev = dict.fromkeys(EB_GRID, 0.0)
        for s in val:
            idx = np.flatnonzero(s.scored)
            if idx.size == 0:
                continue
            H, _ = hidden_states(net, s)
            Phi = _phi(H[idx])
            for var in EB_GRID:
                _W, Hs, f = map_head(Phi, s.actions[idx], s.avail[idx], W0, var)
                ev[var] += -f - 0.5 * K * D * np.log(var) - 0.5 * np.linalg.slogdet(Hs)[1]
        self.var = max(EB_GRID, key=lambda v: (ev[v], -v))
        return {"prior_var": self.var, "eb_log_evidence": {str(k): round(v, 3) for k, v in ev.items()}}

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        idx = np.flatnonzero(history.scored)
        if idx.size == 0:
            return state
        H, _ = hidden_states(self._require_net(), history)
        W_init = None if state is None else state[0]
        W, Hs, _ = map_head(
            _phi(H[idx]), history.actions[idx], history.avail[idx], self._W0(), self.var, W_init
        )
        return (W, Hs)

    def predict_with_states(self, seq: EncodedSeq, states: States) -> np.ndarray:
        H, out = hidden_states(self._require_net(), seq)
        Phi = _phi(H)
        for s, e, st in _segments(states, seq.T):
            if st is None:
                continue
            W, Hs = st
            mu = Phi[s:e] @ W.T
            if self.config.get("predictive", "map") == "laplace":
                K, D = W.shape
                S = np.linalg.inv(Hs).reshape(K, D, K, D)
                v = np.stack([((Phi[s:e] @ S[k, :, k, :]) * Phi[s:e]).sum(axis=1) for k in range(K)], axis=1)
                mu = mu / np.sqrt(1.0 + np.pi * np.maximum(v, 0.0) / 8.0)
            out[s:e] = masked_logsoftmax_np(mu, seq.avail[s:e])
        return out

    def per_person_params(self) -> int:
        net = self._require_net()
        return int(net.head.weight.numel() + net.head.bias.numel())


# ============================================================================ exemplar mixture
def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances without materialising ``[n, m, d]``."""
    d2 = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


class ExemplarMix(_BackboneAdapter):
    name = "exemplar_mix"

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        net = self._require_net()
        rng = stream(seed, "exemplar_mix_bandwidth")
        order = rng.permutation(len(train))[: min(60, len(train))]
        d2: list[np.ndarray] = []
        for i in sorted(int(j) for j in order):
            H, _ = hidden_states(net, train[i])
            if H.shape[0] < 2:
                continue
            ii = rng.integers(0, H.shape[0], size=200)
            jj = rng.integers(0, H.shape[0], size=200)
            keep = ii != jj
            d2.append(((H[ii[keep]] - H[jj[keep]]) ** 2).sum(axis=1))
        med = float(np.median(np.concatenate(d2))) if d2 else 1.0
        self.ell2 = max(1e-6, float(self.config.get("bw", 1.0)) * med)
        return {"median_sq_dist": med, "ell2": self.ell2}

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        idx = np.flatnonzero(history.scored)
        if idx.size == 0:
            return state
        H, _ = hidden_states(self._require_net(), history)
        return (H[idx].copy(), history.actions[idx].copy(), idx.copy())

    def predict_with_states(self, seq: EncodedSeq, states: States) -> np.ndarray:
        H, out = hidden_states(self._require_net(), seq)
        wmix = float(self.config.get("w", 0.2))
        gamma = float(self.config.get("decay", 1.0))
        K = seq.k_max
        for s, e, st in _segments(states, seq.T):
            if st is None:
                continue
            Hk, ak, pos = st
            w = np.exp(-_sqdist(H[s:e], Hk) / (2.0 * self.ell2))
            t = np.arange(s, e)[:, None]
            w = w * gamma ** np.maximum(t - 1 - pos[None, :], 0)
            Y = np.zeros((ak.size, K))
            Y[np.arange(ak.size), ak] = 1.0
            av = seq.avail[s:e]
            counts = (w @ Y) * av
            kav = av.sum(axis=1, keepdims=True)
            pk = np.where(av, (counts + 1.0 / kav) / (counts.sum(axis=1, keepdims=True) + 1.0), 0.0)
            p = (1.0 - wmix) * np.exp(out[s:e]) + wmix * pk
            with np.errstate(divide="ignore"):
                out[s:e] = np.where(av, np.log(p), -np.inf)
        return out


# ============================================================================ Reptile
class ReptileGRU(FTFull):
    name = "reptile_gru"

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        meta = copy.deepcopy(self._require_net())
        meta.eval()
        n, lr, _ = self._hp()
        iters = int(self.config.get("meta_iters", 300))
        mb = int(self.config.get("meta_batch", 4))
        eps0 = float(self.config.get("meta_lr", 0.5))
        rng = stream(seed, "reptile_tasks")
        fit, _val = nets.split_train_val(train, int(self.config.get("backbone_seed", seed)), "pop_gru")
        t0 = time.process_time()
        done = 0
        for it in range(iters):
            if time.process_time() - t0 > budget_s:
                break
            eps = eps0 * (1.0 - it / iters)
            theta = [p.detach().clone() for p in meta.parameters()]
            delta = [torch.zeros_like(p) for p in theta]
            for _ in range(mb):
                s = fit[int(rng.integers(len(fit)))]
                phi = copy.deepcopy(meta)
                opt = torch.optim.Adam(phi.parameters(), lr=lr)
                b = nets.Batch([s])
                for j in range(n):
                    u = max(2, int(round(s.T * (j + 1) / n)))
                    mask = b.mask[:, :u]
                    k = int(mask.sum())
                    if k == 0:
                        continue
                    logp, _ = phi(b.x[:, :u], b.avail[:, :u])
                    lp = logp.gather(-1, b.a[:, :u].unsqueeze(-1)).squeeze(-1)
                    loss = -torch.where(mask, lp, torch.zeros_like(lp)).sum() / k
                    opt.zero_grad()
                    loss.backward()  # type: ignore[no-untyped-call]
                    nn_clip(phi)
                    opt.step()
                for d, p, q in zip(delta, phi.parameters(), theta, strict=True):
                    d += (p.detach() - q) / mb
            with torch.no_grad():
                for p, q, d in zip(meta.parameters(), theta, delta, strict=True):
                    p.copy_(q + eps * d)
            done += 1
        self.meta_net = meta
        return {"meta_iters_done": done, "meta_cpu_s": round(time.process_time() - t0, 2)}

    def _hp(self) -> tuple[int, float, float]:
        n, lr, _ = super()._hp()
        return n, lr, 0.0

    def _base(self) -> nets.PopRNN:
        if not hasattr(self, "meta_net"):
            raise RuntimeError("reptile_gru: predict before fit")
        return self.meta_net


# ============================================================================ CAVIA
class CaviaNet(nn.Module):
    """GRU over ``[x, c]`` and head over ``[h, c]``; ``c`` is a per-sequence context vector."""

    def __init__(self, d_in: int, hidden: int, k: int, c_dim: int) -> None:
        super().__init__()
        self.c_dim = c_dim
        self.rnn = nn.GRU(d_in + c_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden + c_dim, k)

    @classmethod
    def from_backbone(cls, bb: nets.PopRNN, d_in: int, k: int, c_dim: int, seed: int) -> CaviaNet:
        """Backbone weights copied; context weights drawn N(0, 0.1²) (exact zeros would be a saddle: ∇_c = 0).
        With ``c = 0`` the network reproduces the backbone exactly."""
        hidden = bb.rnn.hidden_size
        net = cls(d_in, hidden, k, c_dim)
        g = torch.Generator().manual_seed(int(seed) % (2**31 - 1))
        with torch.no_grad():
            net.rnn.weight_ih_l0[:, :d_in] = bb.rnn.weight_ih_l0
            net.rnn.weight_ih_l0[:, d_in:] = 0.1 * torch.randn(3 * hidden, c_dim, generator=g)
            net.rnn.weight_hh_l0.copy_(bb.rnn.weight_hh_l0)
            net.rnn.bias_ih_l0.copy_(bb.rnn.bias_ih_l0)
            net.rnn.bias_hh_l0.copy_(bb.rnn.bias_hh_l0)
            net.head.weight[:, :hidden] = bb.head.weight
            net.head.weight[:, hidden:] = 0.1 * torch.randn(k, c_dim, generator=g)
            net.head.bias.copy_(bb.head.bias)
        return net

    def forward(
        self, x: torch.Tensor, avail: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if c is None:
            c = torch.zeros(B, self.c_dim, dtype=x.dtype)
        cc = c.unsqueeze(1).expand(B, T, self.c_dim)
        h, _ = self.rnn(torch.cat([x, cc], dim=-1))
        return nets.masked_log_softmax(self.head(torch.cat([h, cc], dim=-1)), avail), h


def _masked_mean_nll(logp: torch.Tensor, b: nets.Batch, rows: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean NLL over scored rows selected by ``rows`` ([B, T] bool); 0 where none."""
    lp = logp.gather(-1, b.a.unsqueeze(-1)).squeeze(-1)
    m = b.mask & rows
    n = m.sum(dim=1).clamp(min=1)
    return -(torch.where(m, lp, torch.zeros_like(lp)).sum(dim=1) / n)


class CaviaGRU(_BackboneAdapter):
    name = "cavia_gru"

    def _inner(self, net: CaviaNet, b: nets.Batch, rows: torch.Tensor, create_graph: bool) -> torch.Tensor:
        alpha = float(self.config.get("inner_lr", 1.0))
        c = torch.zeros(b.B, net.c_dim, requires_grad=True)
        for _ in range(int(self.config.get("n_inner", 2))):
            logp, _ = net(b.x, b.avail, c)
            loss = _masked_mean_nll(logp, b, rows).sum()
            (g,) = torch.autograd.grad(loss, c, create_graph=create_graph)
            c = c - alpha * g
        return c

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        c_dim = int(self.config.get("c_dim", 4))
        net = CaviaNet.from_backbone(self._require_net(), self.d_in, self.k_max, c_dim, seed)
        rng = stream(seed, "cavia_splits")

        def split_rows(b: nets.Batch, fixed: bool) -> torch.Tensor:
            lengths = b.mask.cumsum(dim=1).argmax(dim=1) + 1  # one past the last scored row
            rows = torch.zeros(b.B, b.T, dtype=torch.bool)
            for i in range(b.B):
                Ti = int(lengths[i])
                u = Ti // 2 if fixed else int(rng.integers(1, max(2, Ti)))
                rows[i, :u] = True
            return rows

        def loss_fn(m: torch.nn.Module, b: nets.Batch, fixed: bool = False) -> tuple[torch.Tensor, int]:
            assert isinstance(m, CaviaNet)
            pre = split_rows(b, fixed)
            c = self._inner(m, b, pre, create_graph=m.training)
            logp, _ = m(b.x, b.avail, c)
            lp = logp.gather(-1, b.a.unsqueeze(-1)).squeeze(-1)
            post = b.mask & ~pre
            return -torch.where(post, lp, torch.zeros_like(lp)).sum(), int(post.sum())

        def val_fn(m: torch.nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            with torch.enable_grad():
                s, k = loss_fn(m, b, fixed=True)
            return s.detach(), k

        log = nets.train(
            net,
            loss_fn,
            train,
            seed=int(self.config.get("backbone_seed", seed)),
            key="pop_gru",
            lr=float(self.config.get("meta_lr", 0.003)),
            max_epochs=int(self.config.get("max_epochs", 40)),
            patience=int(self.config.get("patience", 5)),
            batch_size=int(self.config.get("batch_size", 8)),
            budget_s=budget_s,
            val_loss_fn=val_fn,
        )
        self.cnet = net
        return {"cavia_train": log}

    def _cnet(self) -> CaviaNet:
        if not hasattr(self, "cnet"):
            raise RuntimeError("cavia_gru: predict before fit")
        return self.cnet

    def init_state(self) -> Any:
        return np.zeros(int(self.config.get("c_dim", 4)), dtype=np.float32)

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        if not history.scored.any():
            return state
        b = nets.Batch([history])
        with torch.enable_grad():
            c = self._inner(self._cnet(), b, torch.ones(1, b.T, dtype=torch.bool), create_graph=False)
        return c.detach().numpy()[0].copy()

    def predict_with_states(self, seq: EncodedSeq, states: States) -> np.ndarray:
        net = self._cnet()
        out = np.full((seq.T, seq.k_max), -np.inf)
        for s, e, st in _segments(states, seq.T):
            c = torch.from_numpy(np.asarray(st, dtype=np.float32))[None, :]
            out[s:e] = _logp_rows(net, seq, s, e, c=c)
        return out

    def param_count(self) -> int:
        return nets.n_params(self._cnet())

    def per_person_params(self) -> int:
        return int(self.config.get("c_dim", 4))


# ============================================================================ registration
def _factory(cls: type[_BackboneAdapter]):  # type: ignore[no-untyped-def]
    def build(config: dict[str, Any], k: int, d: int) -> _BackboneAdapter:
        return cls(config, k, d)

    return build


_POOL: tuple[tuple[type[_BackboneAdapter], dict[str, list[Any]], str], ...] = (
    (FTHead, {"m": [10, 25], "n": [5, 20], "lr": [0.003, 0.01, 0.03]}, "pop_gru + online head fine-tuning"),
    (FTFull, {"n": [3, 10], "lr": [0.001, 0.003], "l2": [0.0, 1.0]}, "pop_gru + online full fine-tuning"),
    (
        BayesHead,
        {"prior_var": ["eb", 0.01, 0.03, 0.1, 0.3, 1.0], "m": [10, 25], "predictive": ["map", "laplace"]},
        "pop_gru + Bayesian head (Gaussian prior on the population head; EB or tuned variance)",
    ),
    (
        ExemplarMix,
        {"w": [0.05, 0.1, 0.2, 0.4], "bw": [0.25, 1.0, 4.0], "decay": [1.0, 0.98]},
        "pop_gru + exemplar-memory mixture",
    ),
    (ReptileGRU, {"lr": [0.001, 0.003], "n": [3, 10]}, "Reptile-meta-learned GRU + online fine-tuning"),
    (
        CaviaGRU,
        {"c_dim": [4, 8], "inner_lr": [0.3, 1.0]},
        "CAVIA context-latent GRU (gradient-adapted latent)",
    ),
)
for _cls, _grid, _note in _POOL:
    register_model(
        ModelSpec(_cls.name, _factory(_cls), "baselines", "comparator", _grid, adaptive=True, note=_note)
    )
