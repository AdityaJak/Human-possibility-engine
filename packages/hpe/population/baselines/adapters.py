"""Conventional individual-adaptation comparators on a population GRU backbone (thesis §4, pool items 4–7).

Every adapter predicts step ``t`` with parameters fitted only on the participant's steps ``< u(t)``, where
``u(t)`` is the last update point ``<= t`` (update every ``m`` steps; ``backbone.update_points``), and with the
recurrent state running over ``x[:t+1]``. Hence row ``t`` depends on ``x[:t+1]`` and ``actions[:t]`` only.
Each refit starts from the population (or meta-learned) parameters, so no state drifts between updates.

* ``ft_head``      — online fine-tuning of the output layer: ``n`` Adam steps (``lr``) on the mean NLL of the
                     participant's past scored steps (+ optional L2 pull ``l2`` to the population head).
* ``ft_full``      — the same on all weights, with an L2 pull ``l2/2 · ||θ − θ_pop||²``.
* ``bayes_head``   — Gaussian prior ``N(W_pop, σ² I)`` on the head ``W = [weights | bias]``; MAP by Newton's method
                     (warm-started) on the past scored steps; ``predictive="laplace"`` uses the Laplace posterior
                     and the mean-field softmax correction ``μ_k / sqrt(1 + π v_k / 8)``. ``prior_var="eb"``:
                     σ² chosen by type-II maximum likelihood (Laplace evidence) on the backbone's held-out
                     validation participants (dev), i.e. an empirical-Bayes population prior.
* ``exemplar_mix`` — ``p = (1 − w) p_pop + w p_kern``; ``p_kern(a) ∝ Σ_{s<t, scored} k(h_s, h_t) γ^{t−1−s} 1[a_s = a]
                     + 1/K`` with a Gaussian kernel on backbone hidden states (bandwidth ``bw`` × the median
                     squared within-person distance measured on dev).
* ``reptile_gru``  — first-order meta-learning (Reptile; Nichol, Achiam & Schulman, 2018) of the GRU weights,
                     starting from the population backbone: a task is one dev participant; the inner loop takes
                     ``n`` Adam steps on growing prefixes of their sequence (the online-adaptation regime);
                     ``θ ← θ + ε (φ − θ)``. At test time, online full fine-tuning from the meta-learned weights.

``mode="frozen"`` returns the (meta-learned) population predictions; ``donor=`` fits the individual parameters
on the donor's whole sequence, then holds them fixed while predicting ``seq`` (Gate A2 control).
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch

from hpe.core.rng import stream
from hpe.population.baselines.backbone import (
    backbone_config,
    hidden_states,
    masked_logsoftmax_np,
    train_backbone,
    update_points,
)
from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import ModelSpec, register_model


class _BackboneAdapter:
    adaptive = True
    name = "adapter"

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.m = int(config.get("m", 25))
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
        self.threads = threads
        extra = self._fit_extra(train, seed=seed, budget_s=max(1.0, budget_s - (time.process_time() - t0)))
        self.fit_info = {**log, **extra}
        return self.fit_info

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        return {}

    def _require_net(self) -> nets.PopRNN:
        if self.net is None:
            raise RuntimeError(f"{self.name}: predict before fit")
        return self.net

    # ------------------------------------------------------------------ prediction
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        net = self._require_net()
        if mode == "frozen":
            return hidden_states(net, seq)[1]
        if donor is not None:
            return self._predict_donor(seq, donor)
        return self._predict_online(seq)

    def _predict_online(self, seq: EncodedSeq) -> np.ndarray:
        raise NotImplementedError

    def _predict_donor(self, seq: EncodedSeq, donor: EncodedSeq) -> np.ndarray:
        raise NotImplementedError

    def param_count(self) -> int:
        return nets.n_params(self._require_net())

    def per_person_params(self) -> int:
        return 0

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": self.per_person_params(),
            "update_every_m_steps": self.m,
            "config": self.config,
            "backbone_cpu_s": self.fit_info.get("backbone_cpu_s"),
            "backbone_cached": self.fit_info.get("backbone_cached"),
        }


# ============================================================================ head fine-tuning
def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


class FTHead(_BackboneAdapter):
    name = "ft_head"

    def _fit_head(self, H: np.ndarray, a: np.ndarray, avail: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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
        Ht, at, avt = _tensor(H), _tensor(a), _tensor(avail)
        for _ in range(n):
            logp = nets.masked_log_softmax(Ht @ W.T + b, avt)
            loss = -logp.gather(1, at.unsqueeze(1)).mean()
            if l2:
                loss = loss + 0.5 * l2 * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        return W.detach(), b.detach()

    @staticmethod
    def _apply(H: np.ndarray, W: torch.Tensor, b: torch.Tensor, avail: np.ndarray) -> np.ndarray:
        logits = (_tensor(H) @ W.T + b).numpy()
        return masked_logsoftmax_np(logits, avail)

    def _predict_online(self, seq: EncodedSeq) -> np.ndarray:
        H, out = hidden_states(self._require_net(), seq)
        for u in update_points(seq.T, self.m):
            idx = np.flatnonzero(seq.scored[:u])
            if idx.size == 0:
                continue
            W, b = self._fit_head(H[idx], seq.actions[idx], seq.avail[idx])
            end = min(seq.T, u + self.m)
            out[u:end] = self._apply(H[u:end], W, b, seq.avail[u:end])
        return out

    def _predict_donor(self, seq: EncodedSeq, donor: EncodedSeq) -> np.ndarray:
        net = self._require_net()
        Hd, _ = hidden_states(net, donor)
        idx = np.flatnonzero(donor.scored)
        H, out = hidden_states(net, seq)
        if idx.size == 0:
            return out
        W, b = self._fit_head(Hd[idx], donor.actions[idx], donor.avail[idx])
        return self._apply(H, W, b, seq.avail)

    def per_person_params(self) -> int:
        net = self._require_net()
        return int(net.head.weight.numel() + net.head.bias.numel())


# ============================================================================ full fine-tuning
def _finetune_full(
    net0: nets.PopRNN, seq: EncodedSeq, u: int, *, n: int, lr: float, l2: float
) -> nets.PopRNN:
    """Copy of ``net0`` after ``n`` Adam steps on the mean NLL of ``seq``'s scored steps ``< u``."""
    net = copy.deepcopy(net0)
    net.eval()
    p0 = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    b = nets.Batch([seq])
    x, av, a = b.x[:, :u], b.avail[:, :u], b.a[:, :u]
    mask = b.mask[:, :u]
    k = int(mask.sum())
    if k == 0 or n <= 0:
        return net
    for _ in range(n):
        logp, _ = net(x, av)
        lp = logp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        loss = -torch.where(mask, lp, torch.zeros_like(lp)).sum() / k
        if l2:
            loss = loss + 0.5 * l2 * sum(
                ((p - q) ** 2).sum() for p, q in zip(net.parameters(), p0, strict=True)
            )
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        nn_clip(net)
        opt.step()
    return net


def nn_clip(net: torch.nn.Module, clip: float = 1.0) -> None:
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)


@torch.no_grad()
def _logp_rows(net: nets.PopRNN, seq: EncodedSeq, start: int, end: int) -> np.ndarray:
    b = nets.Batch([seq])
    logp, _ = net(b.x[:, :end], b.avail[:, :end])
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

    def _predict_online(self, seq: EncodedSeq) -> np.ndarray:
        base = self._base()
        n, lr, l2 = self._hp()
        out = hidden_states(base, seq)[1]
        for u in update_points(seq.T, self.m):
            if not seq.scored[:u].any():
                continue
            net_u = _finetune_full(base, seq, u, n=n, lr=lr, l2=l2)
            end = min(seq.T, u + self.m)
            out[u:end] = _logp_rows(net_u, seq, u, end)
        return out

    def _predict_donor(self, seq: EncodedSeq, donor: EncodedSeq) -> np.ndarray:
        n, lr, l2 = self._hp()
        net_d = _finetune_full(self._base(), donor, donor.T, n=n, lr=lr, l2=l2)
        return _logp_rows(net_d, seq, 0, seq.T)

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        if mode == "frozen":
            return hidden_states(self._base(), seq)[1]
        if donor is not None:
            return self._predict_donor(seq, donor)
        return self._predict_online(seq)

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
        # empirical Bayes: Laplace evidence summed over the backbone's validation participants (dev)
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
                logdet = np.linalg.slogdet(Hs)[1]
                ev[var] += -f - 0.5 * K * D * np.log(var) - 0.5 * logdet
        self.var = max(EB_GRID, key=lambda v: (ev[v], -v))
        return {"prior_var": self.var, "eb_log_evidence": {str(k): round(v, 3) for k, v in ev.items()}}

    def _predict_rows(self, Phi: np.ndarray, W: np.ndarray, Hs: np.ndarray, avail: np.ndarray) -> np.ndarray:
        mu = Phi @ W.T
        if self.config.get("predictive", "map") == "laplace":
            K, D = W.shape
            S = np.linalg.inv(Hs).reshape(K, D, K, D)
            Skk = np.stack([S[k, :, k, :] for k in range(K)])  # [K, D, D]
            v = np.stack([((Phi @ Skk[k]) * Phi).sum(axis=1) for k in range(K)], axis=1)
            mu = mu / np.sqrt(1.0 + np.pi * np.maximum(v, 0.0) / 8.0)
        return masked_logsoftmax_np(mu, avail)

    def _predict_online(self, seq: EncodedSeq) -> np.ndarray:
        H, out = hidden_states(self._require_net(), seq)
        Phi = _phi(H)
        W0 = self._W0()
        W = W0
        for u in update_points(seq.T, self.m):
            idx = np.flatnonzero(seq.scored[:u])
            if idx.size == 0:
                continue
            W, Hs, _ = map_head(Phi[idx], seq.actions[idx], seq.avail[idx], W0, self.var, W_init=W)
            end = min(seq.T, u + self.m)
            out[u:end] = self._predict_rows(Phi[u:end], W, Hs, seq.avail[u:end])
        return out

    def _predict_donor(self, seq: EncodedSeq, donor: EncodedSeq) -> np.ndarray:
        net = self._require_net()
        Hd, _ = hidden_states(net, donor)
        idx = np.flatnonzero(donor.scored)
        H, out = hidden_states(net, seq)
        if idx.size == 0:
            return out
        W, Hs, _ = map_head(_phi(Hd)[idx], donor.actions[idx], donor.avail[idx], self._W0(), self.var)
        return self._predict_rows(_phi(H), W, Hs, seq.avail)

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

    def _mix(self, out_pop: np.ndarray, p_kern: np.ndarray, avail: np.ndarray) -> np.ndarray:
        wmix = float(self.config.get("w", 0.2))
        p = (1.0 - wmix) * np.exp(out_pop) + wmix * p_kern
        with np.errstate(divide="ignore"):
            return np.where(avail, np.log(p), -np.inf)

    def _predict_online(self, seq: EncodedSeq) -> np.ndarray:
        H, out = hidden_states(self._require_net(), seq)
        return self._mix(out, self._kernel_probs(H, seq), seq.avail)

    def _kernel_probs(self, H: np.ndarray, seq: EncodedSeq) -> np.ndarray:
        """Row t: kernel-weighted frequencies of the person's scored choices at steps s < t (+ 1/K prior)."""
        gamma = float(self.config.get("decay", 1.0))
        T, K = H.shape[0], seq.avail.shape[1]
        d2 = _sqdist(H, H)
        w = np.exp(-d2 / (2.0 * self.ell2))
        t = np.arange(T)[:, None]
        s = np.arange(T)[None, :]
        w = np.where((s < t) & seq.scored[None, :], w * gamma ** np.maximum(t - 1 - s, 0), 0.0)
        Y = np.zeros((T, K))
        Y[np.arange(T), seq.actions] = 1.0
        counts = (w @ Y) * seq.avail
        kav = seq.avail.sum(axis=1, keepdims=True)
        return np.where(seq.avail, (counts + 1.0 / kav) / (counts.sum(axis=1, keepdims=True) + 1.0), 0.0)

    def _predict_donor(self, seq: EncodedSeq, donor: EncodedSeq) -> np.ndarray:
        net = self._require_net()
        Hd, _ = hidden_states(net, donor)
        idx = np.flatnonzero(donor.scored)
        H, out = hidden_states(net, seq)
        if idx.size == 0:
            return out
        d2 = _sqdist(H, Hd[idx])
        w = np.exp(-d2 / (2.0 * self.ell2))
        K = seq.avail.shape[1]
        Y = np.zeros((idx.size, K))
        Y[np.arange(idx.size), donor.actions[idx]] = 1.0
        counts = (w @ Y) * seq.avail
        kav = seq.avail.sum(axis=1, keepdims=True)
        p = np.where(seq.avail, (counts + 1.0 / kav) / (counts.sum(axis=1, keepdims=True) + 1.0), 0.0)
        return self._mix(out, p, seq.avail)

    def per_person_params(self) -> int:
        return 0  # non-parametric memory of the person's (hidden state, choice) pairs


# ============================================================================ Reptile
class ReptileGRU(FTFull):
    name = "reptile_gru"

    def _fit_extra(self, train: list[EncodedSeq], *, seed: int, budget_s: float) -> dict[str, Any]:
        base = self._require_net()
        meta = copy.deepcopy(base)
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

    def _base(self) -> nets.PopRNN:
        if not hasattr(self, "meta_net"):
            raise RuntimeError("reptile_gru: predict before fit")
        return self.meta_net


# ============================================================================ registration
def _factory(cls: type[_BackboneAdapter]):  # type: ignore[no-untyped-def]
    def build(config: dict[str, Any], k: int, d: int) -> _BackboneAdapter:
        return cls(config, k, d)

    return build


_POOL: tuple[tuple[type[_BackboneAdapter], dict[str, list[Any]], str], ...] = (
    (FTHead, {"m": [10, 25], "n": [5, 20], "lr": [0.003, 0.01, 0.03]}, "pop_gru + online head fine-tuning"),
    (
        FTFull,
        {"n": [3, 10], "lr": [0.0003, 0.001, 0.003], "l2": [0.0, 1.0]},
        "pop_gru + online full fine-tuning",
    ),
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
    (
        ReptileGRU,
        {"lr": [0.001, 0.003], "n": [3, 10], "meta_lr": [0.1, 0.5]},
        "Reptile-meta-learned GRU + online fine-tuning",
    ),
)
for _cls, _grid, _note in _POOL:
    register_model(
        ModelSpec(_cls.name, _factory(_cls), "baselines", "comparator", _grid, adaptive=True, note=_note)
    )
