"""F3 ``f3_arbiter`` — hierarchical episodic–parametric model with learned arbitration.

Definition (docs/phase3/POPULATION_ADAPTATION_ARCHITECTURES.md §F3). A population GRU (the shared
backbone, trained exactly like ``pop_gru``) gives p_pop(·|x_≤t) and a hidden state h_t. The person's
episodic memory holds (h_s, a_s) for their own past *free* choices s < t. Kernel retrieval

    w_ts = exp(−|h_t − h_s|² / (2 H b²)) · ρ^{t−1−s}
    p_epi(a) = (Σ_s w_ts 1[a_s = a] + ε/K_t) / (Σ_s w_ts 1[a_s avail] + ε)            (available a)

A gate g_t = σ(wᵀ r̃_t + w_0) arbitrates in probability space, p = g p_epi + (1 − g) p_pop, where r_t are
reliability features computed causally for this person: exponentially weighted running mean log-score
of p_pop and of p_epi on the person's past free choices (decay λ), log(1 + retrieval mass Σ_s w_ts),
normalised population entropy, and log(1 + number of past free choices); r̃ is r standardised with
statistics of the gate-fitting data. The gate (6 weights) is fitted by maximum likelihood on the
backbone's held-out early-stopping participants (dev only), so p_pop is not evaluated on the people it
was fitted to.

Arbiters (registered as separate models so ablations are paired on the same cached backbone):
``gate`` (F3), ``fixed`` (constant g: a number, or ``"learn"`` = fitted on the same split),
``fixedshare`` (Herbster & Warmuth 1998 fixed-share Bayesian mixture of the two experts, share α),
``gru`` (a second GRU initialised from the backbone that receives [p_epi, log(1+mass)] as extra input
and is trained on dev — learned arbitration by recurrence, matched information).
``mode="frozen"``: no episodic memory (p = p_pop; for ``gru``: memory empty). ``donor``: the memory and
the reliability state come from the donor's whole sequence and are held fixed.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from hpe.population.candidates import common
from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq

ARBITERS = ("gate", "fixed", "fixedshare", "gru")
N_REL = 5
LOGP_FLOOR = -10.0


def episodic(
    h: np.ndarray,
    a: np.ndarray,
    scored: np.ndarray,
    avail: np.ndarray,
    *,
    bw: float,
    rho: float,
    eps: float,
    mem_h: np.ndarray | None = None,
    mem_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Episodic prediction p_epi [T, K] and retrieval mass [T].

    Without ``mem_h`` the memory is the person's own free choices s < t (causal). With ``mem_h``/``mem_a``
    (donor control) every row retrieves from that fixed memory.
    """
    T, K = avail.shape
    H = h.shape[1]
    if mem_h is None:
        keys, vals = h, a
        S = T
        causal = np.tril(np.ones((T, S)), k=-1) * scored[None, :].astype(np.float64)
        age = np.arange(T)[:, None] - 1 - np.arange(S)[None, :]
    else:
        assert mem_a is not None
        keys, vals = mem_h, mem_a
        S = keys.shape[0]
        causal = np.ones((T, S))
        age = np.broadcast_to((S - 1 - np.arange(S))[None, :], (T, S))
    if S == 0:
        mass = np.zeros(T)
        p = avail / avail.sum(axis=1, keepdims=True)
        return p, mass
    d2 = (h**2).sum(1)[:, None] + (keys**2).sum(1)[None, :] - 2.0 * h @ keys.T
    d2 = np.maximum(d2, 0.0) / H
    rec = rho ** np.clip(age, 0, None).astype(np.float64)
    w = np.exp(-d2 / (2.0 * bw**2)) * rec * causal
    onehot = np.zeros((S, K))
    onehot[np.arange(S), vals] = 1.0
    counts = (w @ onehot) * avail
    mass = counts.sum(axis=1)
    k_t = avail.sum(axis=1)
    p = (counts + eps * avail / k_t[:, None]) / (mass + eps)[:, None]
    return p, mass


def reliability(
    lp_pop_a: np.ndarray, lp_epi_a: np.ndarray, scored: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """Causal EW mean log-scores before each trial; also the end state (for the donor control)."""
    T = lp_pop_a.shape[0]
    ls_pop, ls_epi, n_past = np.zeros(T), np.zeros(T), np.zeros(T)
    num_p = num_e = den = 0.0
    n = 0
    for t in range(T):
        if den > 0:
            ls_pop[t], ls_epi[t] = num_p / den, num_e / den
        n_past[t] = n
        if scored[t]:
            num_p = lam * num_p + max(LOGP_FLOOR, float(lp_pop_a[t]))
            num_e = lam * num_e + max(LOGP_FLOOR, float(lp_epi_a[t]))
            den = lam * den + 1.0
            n += 1
    end_p, end_e = (num_p / den, num_e / den) if den > 0 else (0.0, 0.0)
    return ls_pop, ls_epi, n_past, end_p, end_e, n


def rel_features(
    ls_pop: np.ndarray,
    ls_epi: np.ndarray,
    mass: np.ndarray,
    p_pop: np.ndarray,
    avail: np.ndarray,
    n_past: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(avail, p_pop * np.log(np.maximum(p_pop, 1e-300)), 0.0).sum(1)
    k_t = avail.sum(1)
    ent_n = np.where(k_t > 1, ent / np.log(np.maximum(k_t, 2)), 0.0)
    return np.stack([ls_pop, ls_epi, np.log1p(mass), ent_n, np.log1p(n_past)], axis=1)


class F3Arbiter:
    adaptive = True

    def __init__(
        self,
        config: dict[str, Any],
        k_max: int,
        d_in: int,
        *,
        arbiter: str = "gate",
        name: str = "f3_arbiter",
    ) -> None:
        if arbiter not in ARBITERS:
            raise ValueError(f"unknown arbiter {arbiter!r}")
        self.name, self.arbiter = name, arbiter
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        c = self.config
        self.hidden = int(c.get("hidden", 32))
        self.bw = float(c.get("bandwidth", 0.5))
        self.rho = float(0.5 ** (1.0 / float(c.get("epi_halflife", 1e9))))
        self.eps = float(c.get("epi_eps", 0.5))
        self.lam = float(c.get("rel_decay", 0.9))
        self.backbone = nets.PopRNN(d_in, self.hidden, k_max, dropout=float(c.get("dropout", 0.0)))
        self.gate_w = np.zeros(N_REL)
        self.gate_b = -2.0
        self.feat_mu = np.zeros(N_REL)
        self.feat_sd = np.ones(N_REL)
        self.fixed_g = 0.1
        self.fs_alpha = float(c.get("share", 0.05))
        self.fs_init = float(c.get("share_init", 0.5))
        self.gru: nets.PopRNN | None = None

    # ------------------------------------------------------------------ per-sequence quantities
    def _pop(self, seq: EncodedSeq) -> tuple[np.ndarray, np.ndarray]:
        b = nets.Batch([seq])
        with torch.no_grad():
            lp, h = self.backbone(b.x, b.avail)
        p = np.exp(nets.logp_numpy(lp[0], seq))
        return p, h[0].double().numpy()[: seq.T]

    def _state(self, seq: EncodedSeq, donor: EncodedSeq | None = None) -> dict[str, np.ndarray]:
        """p_pop, p_epi, mass and reliability features (causal; or from a fixed donor memory)."""
        p_pop, h = self._pop(seq)
        if donor is None:
            p_epi, mass = episodic(
                h, seq.actions, seq.scored, seq.avail, bw=self.bw, rho=self.rho, eps=self.eps
            )
            ar = np.arange(seq.T)
            ls_pop, ls_epi, n_past, *_ = reliability(
                np.log(np.maximum(p_pop[ar, seq.actions], 1e-300)),
                np.log(np.maximum(p_epi[ar, seq.actions], 1e-300)),
                seq.scored,
                self.lam,
            )
        else:
            dp, dh = self._pop(donor)
            dpe, _ = episodic(
                dh, donor.actions, donor.scored, donor.avail, bw=self.bw, rho=self.rho, eps=self.eps
            )
            ad = np.arange(donor.T)
            _, _, _, end_p, end_e, n_d = reliability(
                np.log(np.maximum(dp[ad, donor.actions], 1e-300)),
                np.log(np.maximum(dpe[ad, donor.actions], 1e-300)),
                donor.scored,
                self.lam,
            )
            keep = donor.scored
            p_epi, mass = episodic(
                h,
                seq.actions,
                seq.scored,
                seq.avail,
                bw=self.bw,
                rho=self.rho,
                eps=self.eps,
                mem_h=dh[keep],
                mem_a=donor.actions[keep],
            )
            ls_pop, ls_epi = np.full(seq.T, end_p), np.full(seq.T, end_e)
            n_past = np.full(seq.T, float(n_d))
        feats = rel_features(ls_pop, ls_epi, mass, p_pop, seq.avail, n_past)
        return {"p_pop": p_pop, "p_epi": p_epi, "mass": mass, "feats": feats}

    # ------------------------------------------------------------------ fitting
    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        t0 = time.process_time()
        self.backbone, blog = common.train_backbone(
            train, self.config, self.k_max, self.d_in, threads=threads, budget_s=budget_s
        )
        nets.setup_torch(seed, threads, self.name, repr(sorted(self.config.items())))
        _, val = common.backbone_val_split(train, self.config)
        log: dict[str, Any] = {"backbone": blog, "arbiter": self.arbiter, "n_gate_fit": len(val)}
        if self.arbiter in ("gate", "fixed"):
            states = [self._state(s) for s in val]
            pe, pp, ft = [], [], []
            for s, st in zip(val, states, strict=True):
                idx = np.flatnonzero(s.scored)
                pe.append(st["p_epi"][idx, s.actions[idx]])
                pp.append(st["p_pop"][idx, s.actions[idx]])
                ft.append(st["feats"][idx])
            p_e, p_p, f = np.concatenate(pe), np.concatenate(pp), np.concatenate(ft)
            if self.arbiter == "gate":
                self.feat_mu = f.mean(0)
                self.feat_sd = np.where(f.std(0) > 1e-8, f.std(0), 1.0)
                w, b0, nll = _fit_gate(
                    (f - self.feat_mu) / self.feat_sd, p_e, p_p, l2=float(self.config.get("gate_l2", 1e-3))
                )
                self.gate_w, self.gate_b = w, b0
                log.update(
                    gate_w=[round(v, 4) for v in w.tolist()], gate_b=round(b0, 4), gate_fit_nll=round(nll, 5)
                )
            else:
                mix = self.config.get("mix", "learn")
                if mix == "learn":
                    _, b0, nll = _fit_gate(np.zeros((p_e.shape[0], 0)), p_e, p_p, l2=0.0)
                    self.fixed_g = float(1.0 / (1.0 + math.exp(-b0)))
                    log["fixed_fit_nll"] = round(nll, 5)
                else:
                    self.fixed_g = float(mix)
                log["fixed_g"] = round(self.fixed_g, 4)
        elif self.arbiter == "gru":
            log["gru"] = self._fit_gru(train, seed, threads, max(30.0, budget_s - (time.process_time() - t0)))
        own = time.process_time() - t0
        log["cpu_s_incl_backbone"] = round(own + (blog["backbone_cpu_s"] if blog["cached"] else 0.0), 2)
        return log

    def _extra(self, seq: EncodedSeq, st: dict[str, np.ndarray] | None) -> np.ndarray:
        if st is None:  # empty memory
            p = seq.avail / seq.avail.sum(1, keepdims=True)
            return np.concatenate([p, np.zeros((seq.T, 1))], axis=1).astype(np.float32)
        return np.concatenate([st["p_epi"], np.log1p(st["mass"])[:, None]], axis=1).astype(np.float32)

    def _fit_gru(self, train: list[EncodedSeq], seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        extras = {s.key: self._extra(s, self._state(s)) for s in train}
        k = self.k_max
        gru = nets.PopRNN(
            self.d_in, self.hidden, k, extra_in=k + 1, dropout=float(self.config.get("dropout", 0.0))
        )
        with torch.no_grad():
            src = self.backbone.rnn
            gru.rnn.weight_ih_l0.zero_()
            gru.rnn.weight_ih_l0[:, : self.d_in].copy_(src.weight_ih_l0)
            gru.rnn.weight_hh_l0.copy_(src.weight_hh_l0)
            gru.rnn.bias_ih_l0.copy_(src.bias_ih_l0)
            gru.rnn.bias_hh_l0.copy_(src.bias_hh_l0)
            gru.head.load_state_dict(self.backbone.head.state_dict())

        def loss(m: torch.nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            ex = np.zeros((b.B, b.T, k + 1), dtype=np.float32)
            for i, key in enumerate(b.keys):
                e = extras[key]
                ex[i, : e.shape[0]] = e
            logp, _ = m(b.x, b.avail, torch.from_numpy(ex))
            return nets.nll_sum(logp, b)

        out = nets.train(
            gru,
            loss,
            train,
            seed=seed,
            key=self.name,
            budget_s=budget_s,
            **common.trainer_kwargs(self.config),
        )
        self.gru = gru
        return out

    # ------------------------------------------------------------------ prediction
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        self.backbone.eval()
        if self.arbiter == "gru":
            assert self.gru is not None
            st = None if mode == "frozen" else self._state(seq, donor)
            b = nets.Batch([seq])
            with torch.no_grad():
                lp, _ = self.gru(b.x, b.avail, torch.from_numpy(self._extra(seq, st))[None])
            return nets.logp_numpy(lp[0], seq)
        if mode == "frozen":
            p_pop, _ = self._pop(seq)
            return common.finalize_logp(p_pop, seq.avail)
        st = self._state(seq, donor)
        p_pop, p_epi = st["p_pop"], st["p_epi"]
        if self.arbiter == "gate":
            z = ((st["feats"] - self.feat_mu) / self.feat_sd) @ self.gate_w + self.gate_b
            g = 1.0 / (1.0 + np.exp(-z))
            p = g[:, None] * p_epi + (1.0 - g[:, None]) * p_pop
        elif self.arbiter == "fixed":
            p = self.fixed_g * p_epi + (1.0 - self.fixed_g) * p_pop
        else:
            p = self._fixed_share(seq, p_pop, p_epi, donor)
        return common.finalize_logp(p, seq.avail)

    def _fixed_share(
        self, seq: EncodedSeq, p_pop: np.ndarray, p_epi: np.ndarray, donor: EncodedSeq | None
    ) -> np.ndarray:
        a = self.fs_alpha
        if donor is not None:  # weights after the donor's whole sequence, held fixed
            dp, dh = self._pop(donor)
            dpe, _ = episodic(
                dh, donor.actions, donor.scored, donor.avail, bw=self.bw, rho=self.rho, eps=self.eps
            )
            v = _share_run(dp, dpe, donor.actions, donor.scored, a, self.fs_init)[1]
            return float(v[0]) * p_pop + float(v[1]) * p_epi
        wts, _ = _share_run(p_pop, p_epi, seq.actions, seq.scored, a, self.fs_init)
        return wts[:, :1] * p_pop + wts[:, 1:] * p_epi

    def param_count(self) -> int:
        n = nets.n_params(self.backbone)
        if self.arbiter == "gate":
            n += N_REL + 1
        elif self.arbiter == "fixed":
            n += 1
        elif self.arbiter == "gru" and self.gru is not None:
            n += nets.n_params(self.gru)
        return n

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": 0,
            "per_person_state": "episodic memory of (backbone hidden state, chosen action) + reliability statistics",
            "arbiter": self.arbiter,
            "gate_w": self.gate_w.tolist() if self.arbiter == "gate" else None,
            "gate_b": self.gate_b if self.arbiter == "gate" else None,
            "fixed_g": self.fixed_g if self.arbiter == "fixed" else None,
            "config": self.config,
        }


def _share_run(
    p_pop: np.ndarray, p_epi: np.ndarray, a: np.ndarray, scored: np.ndarray, alpha: float, init: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-share weights before each trial [T, 2] (pop, epi) and the final weights."""
    T = p_pop.shape[0]
    v = np.array([1.0 - init, init])
    out = np.zeros((T, 2))
    for t in range(T):
        out[t] = v
        if scored[t]:
            v = v * np.array([max(p_pop[t, a[t]], 1e-300), max(p_epi[t, a[t]], 1e-300)])
            v = v / v.sum()
            v = (1.0 - alpha) * v + alpha / 2.0
    return out, v


def _fit_gate(
    f: np.ndarray, p_e: np.ndarray, p_p: np.ndarray, *, l2: float, iters: int = 400
) -> tuple[np.ndarray, float, float]:
    """Maximum-likelihood gate weights for p = σ(fᵀw + b) p_e + (1 − σ) p_p (Adam, full batch)."""
    ft = torch.from_numpy(f.astype(np.float64))
    pe = torch.from_numpy(np.maximum(p_e, 1e-12).astype(np.float64))
    pp = torch.from_numpy(np.maximum(p_p, 1e-12).astype(np.float64))
    w = torch.zeros(f.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    nll = torch.zeros(())
    for _ in range(iters):
        g = torch.sigmoid(ft @ w + b) if f.shape[1] else torch.sigmoid(b).expand_as(pe)
        nll = -torch.log(g * pe + (1 - g) * pp).mean()
        loss = nll + l2 * (w**2).sum()
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    return w.detach().numpy().copy(), float(b.detach()), float(nll.detach())
