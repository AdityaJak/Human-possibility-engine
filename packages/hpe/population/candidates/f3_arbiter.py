"""F3 ``f3_arbiter`` — hierarchical episodic–parametric model with learned arbitration.

Definition (docs/phase3/POPULATION_ADAPTATION_ARCHITECTURES.md §F3). A population GRU (the shared
backbone, trained exactly like ``pop_gru``) gives p_pop(·|x_≤t) and a hidden state h_t. The person-level
state (harness-owned schedule, ``core.adaptation``; updated every ``update_every`` trials, default 1) is an
episodic memory of (h_s, a_s) for the person's own past *free* choices plus reliability statistics.
Kernel retrieval for row t from memory M:

    w_s = exp(−|h_t − h_s|² / (2 H b²)) · ρ^{t−1−pos_s}
    p_epi(a) = (Σ_s w_s 1[a_s = a] + ε/K_t) / (Σ_s w_s 1[a_s avail] + ε)            (available a)

A gate g_t = σ(wᵀ r̃_t + w_0) arbitrates in probability space, p = g p_epi + (1 − g) p_pop. Reliability
features r_t: exponentially weighted (decay λ) mean log-score of p_pop and of p_epi on the person's past
free choices (each scored with the memory it had *before* that choice), log(1 + retrieval mass),
normalised population entropy at t, log(1 + memory size); r̃ is r standardised with the gate-fitting
data. With an empty memory there is no episodic expert and p = p_pop (so ``frozen`` = the backbone).
The gate (6 weights) is fitted by maximum likelihood on the backbone's held-out early-stopping
participants (dev only), so p_pop is not evaluated on the people it was fitted to.

Arbiters (registered as separate models; ablations are paired on the same cached backbone):
``gate`` (F3), ``fixed`` (constant g: a number, or ``"learn"`` = fitted on the same split),
``fixedshare`` (Herbster & Warmuth 1998 fixed-share Bayesian mixture of the two experts, share α; the
weights are part of the person state), ``gru`` (a second GRU initialised from the backbone that receives
[p_epi, log(1+mass)] as extra input and is trained on dev — learned arbitration by recurrence).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from hpe.population.candidates import common
from hpe.population.core import nets
from hpe.population.core.adaptation import AdaptiveBase
from hpe.population.core.encoding import EncodedSeq

ARBITERS = ("gate", "fixed", "fixedshare", "gru")
N_REL = 5
LOGP_FLOOR = -10.0


def _kernel_probs(
    q: np.ndarray, t: np.ndarray, st: dict[str, Any], avail: np.ndarray, *, bw: float, rho: float, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """p_epi [R, K] and retrieval mass [R] for query hidden states q [R, H] at rows t [R]."""
    keys, acts, pos = st["keys"], st["acts"], st["pos"]
    K = avail.shape[1]
    H = q.shape[1]
    d2 = (q**2).sum(1)[:, None] + (keys**2).sum(1)[None, :] - 2.0 * q @ keys.T
    d2 = np.maximum(d2, 0.0) / H
    age = np.clip(t[:, None] - 1 - pos[None, :], 0, None).astype(np.float64)
    w = np.exp(-d2 / (2.0 * bw**2)) * rho**age
    onehot = np.zeros((acts.shape[0], K))
    onehot[np.arange(acts.shape[0]), acts] = 1.0
    counts = (w @ onehot) * avail
    mass = counts.sum(1)
    k_t = avail.sum(1)
    return (counts + eps * avail / k_t[:, None]) / (mass + eps)[:, None], mass


def _entropy_norm(p: np.ndarray, avail: np.ndarray) -> np.ndarray:
    ent = -np.where(avail, p * np.log(np.maximum(p, 1e-300)), 0.0).sum(1)
    k_t = avail.sum(1)
    return np.where(k_t > 1, ent / np.log(np.maximum(k_t, 2)), 0.0)


class F3Arbiter(AdaptiveBase):
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
        self.update_every = int(c.get("update_every", 1))
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

    # ------------------------------------------------------------------ person-level state
    def init_state(self) -> dict[str, Any]:
        return {
            "n_proc": 0,
            "h": np.zeros(self.hidden, dtype=np.float32),
            "keys": np.zeros((0, self.hidden)),
            "acts": np.zeros(0, dtype=np.int64),
            "pos": np.zeros(0, dtype=np.int64),
            "num_p": 0.0,
            "num_e": 0.0,
            "den": 0.0,
            "fs": np.array([1.0 - self.fs_init, self.fs_init]),
        }

    def update_state(self, state: Any, history: EncodedSeq) -> dict[str, Any]:
        st = state if int(state["n_proc"]) <= history.T else self.init_state()
        n0 = int(st["n_proc"])
        if history.T == n0:
            return st
        x = torch.from_numpy(history.x[n0:])[None]
        av = torch.from_numpy(history.avail[n0:])[None]
        with torch.no_grad():
            out, h_n = self.backbone.rnn(x, torch.from_numpy(st["h"])[None, None])
            lp = nets.masked_log_softmax(self.backbone.head(out), av)
        hs = out[0].double().numpy()
        p_pop = np.exp(lp[0].double().numpy())
        keys, acts, pos = [st["keys"]], [st["acts"]], [st["pos"]]
        cur = dict(st)
        num_p, num_e, den, fs = float(st["num_p"]), float(st["num_e"]), float(st["den"]), st["fs"].copy()
        for j in range(history.T - n0):
            s = n0 + j
            if not history.scored[s]:
                continue
            a = int(history.actions[s])
            if cur["acts"].shape[0] > 0:
                pe, _ = _kernel_probs(
                    hs[j : j + 1],
                    np.array([s]),
                    cur,
                    history.avail[s : s + 1],
                    bw=self.bw,
                    rho=self.rho,
                    eps=self.eps,
                )
                lp_p = max(LOGP_FLOOR, math.log(max(p_pop[j, a], 1e-300)))
                lp_e = max(LOGP_FLOOR, math.log(max(pe[0, a], 1e-300)))
                num_p, num_e, den = self.lam * num_p + lp_p, self.lam * num_e + lp_e, self.lam * den + 1.0
                fs = fs * np.array([max(p_pop[j, a], 1e-300), max(pe[0, a], 1e-300)])
                fs = (1.0 - self.fs_alpha) * fs / fs.sum() + self.fs_alpha / 2.0
            keys.append(hs[j : j + 1])
            acts.append(np.array([a], dtype=np.int64))
            pos.append(np.array([s], dtype=np.int64))
            cur = {"keys": np.concatenate(keys), "acts": np.concatenate(acts), "pos": np.concatenate(pos)}
            keys, acts, pos = [cur["keys"]], [cur["acts"]], [cur["pos"]]
        return {
            "n_proc": history.T,
            "h": h_n[0, 0].numpy().copy(),
            "keys": cur["keys"],
            "acts": cur["acts"],
            "pos": cur["pos"],
            "num_p": num_p,
            "num_e": num_e,
            "den": den,
            "fs": fs,
        }

    def _rows(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> dict[str, np.ndarray]:
        """Per-row p_pop, p_epi, mass, reliability features and fixed-share weights under the schedule."""
        b = nets.Batch([seq])
        with torch.no_grad():
            lp, h = self.backbone(b.x, b.avail)
        p_pop = np.exp(nets.logp_numpy(lp[0], seq))
        hs = h[0].double().numpy()[: seq.T]
        T = seq.T
        p_epi = (seq.avail / seq.avail.sum(1, keepdims=True)).astype(np.float64)
        mass, has = np.zeros(T), np.zeros(T, dtype=bool)
        feats, fsw = np.zeros((T, N_REL)), np.zeros((T, 2))
        ent = _entropy_norm(p_pop, seq.avail)
        bounds = [s for s, _ in states] + [T]
        for (s, st), e in zip(states, bounds[1:], strict=True):
            if e <= s:
                continue
            rows = np.arange(s, e)
            n_mem = int(st["acts"].shape[0])
            fsw[s:e] = st["fs"]
            if n_mem == 0:
                continue
            pe, ms = _kernel_probs(hs[s:e], rows, st, seq.avail[s:e], bw=self.bw, rho=self.rho, eps=self.eps)
            p_epi[s:e], mass[s:e], has[s:e] = pe, ms, True
            den = float(st["den"])
            ls_p = float(st["num_p"]) / den if den > 0 else 0.0
            ls_e = float(st["num_e"]) / den if den > 0 else 0.0
            feats[s:e, 0], feats[s:e, 1] = ls_p, ls_e
            feats[s:e, 2] = np.log1p(ms)
            feats[s:e, 4] = math.log1p(n_mem)
        feats[:, 3] = ent
        return {"p_pop": p_pop, "p_epi": p_epi, "mass": mass, "has": has, "feats": feats, "fs": fsw}

    def _own_states(self, seq: EncodedSeq) -> list[tuple[int, Any]]:
        """The adapt-mode schedule on the person's own prefixes (used when fitting on dev)."""
        from hpe.population.core.adaptation import schedule

        return schedule(self, seq, mode="adapt", donor=None)

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
            pe, pp, ft = [], [], []
            for s in val:
                r = self._rows(s, self._own_states(s))
                idx = np.flatnonzero(s.scored & r["has"])
                pe.append(r["p_epi"][idx, s.actions[idx]])
                pp.append(r["p_pop"][idx, s.actions[idx]])
                ft.append(r["feats"][idx])
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
            log["gru"] = self._fit_gru(train, seed, max(30.0, budget_s - (time.process_time() - t0)))
        own = time.process_time() - t0
        log["cpu_s_incl_backbone"] = round(own + (blog["backbone_cpu_s"] if blog["cached"] else 0.0), 2)
        return log

    def _extra(self, r: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([r["p_epi"], np.log1p(r["mass"])[:, None]], axis=1).astype(np.float32)

    def _fit_gru(self, train: list[EncodedSeq], seed: int, budget_s: float) -> dict[str, Any]:
        extras = [self._extra(self._rows(s, self._own_states(s))) for s in train]
        by_id = {id(s): e for s, e in zip(train, extras, strict=True)}
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
        # nets.Batch keeps only keys; map batches back to their sequences by key within this fit only
        key_to_extra: dict[str, np.ndarray] = {}
        for s in train:
            key_to_extra.setdefault(s.key, by_id[id(s)])

        def loss(m: torch.nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            ex = np.zeros((b.B, b.T, k + 1), dtype=np.float32)
            for i, key in enumerate(b.keys):
                e = key_to_extra[key]
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
    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        r = self._rows(seq, states)
        p_pop, p_epi, has = r["p_pop"], r["p_epi"], r["has"]
        if self.arbiter == "gru":
            assert self.gru is not None
            b = nets.Batch([seq])
            with torch.no_grad():
                lp, _ = self.gru(b.x, b.avail, torch.from_numpy(self._extra(r))[None])
            return nets.logp_numpy(lp[0], seq)
        if self.arbiter == "gate":
            z = ((r["feats"] - self.feat_mu) / self.feat_sd) @ self.gate_w + self.gate_b
            g = np.where(has, 1.0 / (1.0 + np.exp(-z)), 0.0)
            p = g[:, None] * p_epi + (1.0 - g[:, None]) * p_pop
        elif self.arbiter == "fixed":
            g = np.where(has, self.fixed_g, 0.0)
            p = g[:, None] * p_epi + (1.0 - g[:, None]) * p_pop
        else:
            w = np.where(has[:, None], r["fs"], np.array([1.0, 0.0]))
            p = w[:, :1] * p_pop + w[:, 1:] * p_epi
        return common.finalize_logp(p, seq.avail)

    def twin_config(self) -> dict[str, Any]:
        return common.twin_config(self.config)

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
            "update_every": self.update_every,
            "gate_w": self.gate_w.tolist() if self.arbiter == "gate" else None,
            "gate_b": self.gate_b if self.arbiter == "gate" else None,
            "fixed_g": self.fixed_g if self.arbiter == "fixed" else None,
            "config": self.config,
        }


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
