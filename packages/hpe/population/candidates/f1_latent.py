"""F1 ``f1_latent`` — population GRU with an explicit individual latent inferred online (CAVIA-style).

Definition (docs/phase3/POPULATION_ADAPTATION_ARCHITECTURES.md §F1). Shared weights θ; per-person latent
z = [z_trait (d_tr), z_ctx (d_ctx)]. z_trait enters the recurrent input and the read-out; z_ctx enters the
read-out only (additive + low-rank bilinear modulation of the hidden state), so it can change quickly
without rewriting the recurrent history.

    h_t = GRU([x_t, z_trait], h_{t-1});   logits_t = W [h_t, z] + b + Σ_j z_j V_j h_t

Person-level state (harness-owned schedule, ``core.adaptation``): the state is z. ``update_state`` is
called every ``m`` trials with the person's first t trials and re-infers z from the prior mean z = 0 by
``k_inner`` block Gauss–Newton (Fisher-scoring, i.e. Laplace-mode) steps on the negative log-posteriors

    J_trait(z) = Σ_{s<t} ℓ_s(z) + |z_trait|² / (2 σ_trait²)                         (all past trials)
    J_ctx(z)   = γ_t Σ_{s<t} ρ^{t-1-s} ℓ_s(z) + |z_ctx|² / (2 σ_ctx²)               (recency-weighted)
    z_b ← z_b − (Σ_s w_s J_sᵀ(diag p_s − p_s p_sᵀ) J_s + σ_b⁻² I)⁻¹ (Σ_s w_s J_sᵀ(p_s − y_s) + σ_b⁻² z_b)

with J_s = ∂logits_s/∂z_b (forward differences, no backward pass) and γ_t = clip(exp(β S_t), 1/4, 4)
scaling the context block's evidence relative to its prior, i.e. its effective step size (γ → 0: stay at
the prior; γ large: move towards the recent-data maximum-likelihood latent). S_t (surprise) is the
person's mean excess NLL over the last ``window`` trials under the population prediction (z = 0):
ℓ_s(0) − H[p_s(0)], i.e. how much more surprising the person is than the population model expects.

``predict_with_states`` re-runs the GRU from the first trial with each segment's z and reads out the
segment's rows (row t uses x[:t+1] only). ``inference="gd"`` (ablation ``f1_cavia``): one latent through
recurrence and read-out, ``k_inner`` plain gradient steps with a fixed inner learning rate on the mean
past NLL (no prior, no recency, no surprise) — the standard CAVIA context adapter.

Training: stage 1 is the shared population backbone (``common.train_backbone``, trained exactly like
``pop_gru``; z = 0); stage 2 (``meta=True``) is CAVIA-style meta-training of the latent pathway only (the
z columns of the GRU input and read-out and the bilinear modulation; the backbone weights stay frozen
unless ``meta_train_backbone``), so the z = 0 network *is* the population backbone. For each person in a
batch, ``train_cuts`` cut points c are sampled; z is inferred on the prefix [0, c) and the outer loss is
the NLL of trials [c, c+m). The meta-gradient is first-order (z is a constant for θ in the outer loss).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.core.rng import stream
from hpe.population.candidates import common
from hpe.population.core import nets
from hpe.population.core.adaptation import AdaptiveBase
from hpe.population.core.encoding import EncodedSeq

FD_STEP = 0.05


class LatentGRU(nn.Module):
    def __init__(self, d_in: int, hidden: int, k: int, d_tr: int, d_ctx: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_tr, self.d_ctx = d_tr, d_ctx
        dz = d_tr + d_ctx
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(d_in + d_tr, hidden, batch_first=True)
        self.head = nn.Linear(hidden + dz, k)
        self.mod = nn.Parameter(torch.randn(max(dz, 1), k, hidden) * (0.1 / math.sqrt(hidden)))

    def encode(self, x: torch.Tensor, z_tr: torch.Tensor) -> torch.Tensor:
        inp = self.drop(x)
        if self.d_tr:
            inp = torch.cat([inp, z_tr[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        h, _ = self.rnn(inp)
        return h

    def logits(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not z.shape[1]:
            return self.head(h)
        out = self.head(torch.cat([h, z[:, None, :].expand(-1, h.shape[1], -1)], dim=-1))
        m = torch.einsum("rj,jkh->rkh", z, self.mod[: z.shape[1]])
        return out + torch.einsum("rkh,rth->rtk", m, h)

    def readout(self, h: torch.Tensor, z: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        return nets.masked_log_softmax(self.logits(h, z), avail)

    def freeze_backbone_part(self) -> list[Any]:
        """Zero the gradients of every z-independent weight (meta-train only the latent pathway).

        With these hooks the z = 0 network is exactly the loaded backbone. Adam leaves a parameter with
        identically zero gradients (and zero moments) unchanged.
        """
        d = self.rnn.weight_ih_l0.shape[1] - self.d_tr
        hsz = self.head.weight.shape[1] - self.d_tr - self.d_ctx
        m_ih = torch.zeros_like(self.rnn.weight_ih_l0)
        m_ih[:, d:] = 1.0
        m_head = torch.zeros_like(self.head.weight)
        m_head[:, hsz:] = 1.0
        hooks = [
            self.rnn.weight_ih_l0.register_hook(lambda g: g * m_ih),
            self.head.weight.register_hook(lambda g: g * m_head),
        ]
        for prm in (self.rnn.weight_hh_l0, self.rnn.bias_ih_l0, self.rnn.bias_hh_l0, self.head.bias):
            hooks.append(prm.register_hook(lambda g: torch.zeros_like(g)))
        return hooks

    def load_backbone(self, pop: nets.PopRNN) -> None:
        """Copy a trained ``PopRNN`` into the z-independent part (z columns keep their random init)."""
        src = pop.rnn
        d = src.weight_ih_l0.shape[1]
        with torch.no_grad():
            self.rnn.weight_ih_l0[:, :d].copy_(src.weight_ih_l0)
            self.rnn.weight_hh_l0.copy_(src.weight_hh_l0)
            self.rnn.bias_ih_l0.copy_(src.bias_ih_l0)
            self.rnn.bias_hh_l0.copy_(src.bias_hh_l0)
            hsz = pop.head.weight.shape[1]
            self.head.weight[:, :hsz].copy_(pop.head.weight)
            self.head.bias.copy_(pop.head.bias)


@dataclass(frozen=True, slots=True)
class F1Hyper:
    d_tr: int = 2
    d_ctx: int = 2
    sigma_tr: float = 0.5
    sigma_ctx: float = 1.0
    k_inner: int = 2
    m: int = 10
    halflife: float = 20.0
    beta: float = 2.0
    window: int = 20
    inference: str = "gn"  # "gn" (Gauss–Newton with prior) | "gd" (fixed-step CAVIA gradient descent)
    inner_lr: float = 1.0

    @property
    def rho(self) -> float:
        return float(0.5 ** (1.0 / self.halflife))


def _chosen(lp: torch.Tensor, a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ll = lp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, ll, torch.zeros_like(ll))


def _gn_block(
    jac: torch.Tensor, p: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted gradient Σ_t w_t J_tᵀ(p_t − y_t) and Fisher Σ_t w_t J_tᵀ(diag p_t − p_t p_tᵀ)J_t per row."""
    g = torch.einsum("rtkd,rtk,rt->rd", jac, p - y, w)
    jp = torch.einsum("rtkd,rtk->rtd", jac, p)
    fis = torch.einsum("rtkd,rtk,rtke,rt->rde", jac, p, jac, w) - torch.einsum("rtd,rte,rt->rde", jp, jp, w)
    return g, fis


def _newton(z: torch.Tensor, g: torch.Tensor, fis: torch.Tensor, prec: float) -> torch.Tensor:
    d = z.shape[1]
    eye = torch.eye(d, dtype=torch.float64)
    step = torch.linalg.solve(fis + prec * eye, (g + prec * z.double()).unsqueeze(-1)).squeeze(-1)
    return (z.double() - step).to(z.dtype)


@torch.no_grad()
def _infer_gn(
    net: LatentGRU,
    hp: F1Hyper,
    x: torch.Tensor,
    a: torch.Tensor,
    avail: torch.Tensor,
    scored: torch.Tensor,
    cut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, T = a.shape
    t_idx = torch.arange(T)
    past_b = scored & (t_idx[None, :] < cut[:, None])
    past = past_b.double()
    age = (cut[:, None] - 1 - t_idx[None, :]).clamp(min=0).double()
    w_c = past * hp.rho**age
    z_tr = torch.zeros(R, hp.d_tr)
    z_c = torch.zeros(R, hp.d_ctx)
    gamma = torch.ones(R, 1, dtype=torch.float64)
    if hp.k_inner <= 0 or not bool(past_b.any()):
        return z_tr, z_c, gamma
    y = torch.nn.functional.one_hot(a, avail.shape[-1]).double()
    for step in range(hp.k_inner):
        z = torch.cat([z_tr, z_c], dim=1)
        h = net.encode(x, z_tr)
        lg = net.logits(h, z)
        lp = nets.masked_log_softmax(lg, avail)
        p = lp.exp().double()
        if step == 0 and hp.d_ctx and hp.beta != 0.0:  # population (z = 0) surprise
            ll = _chosen(lp, a, past_b).double()
            ent = -torch.where(avail, p * lp.double(), torch.zeros_like(p)).sum(-1)
            win = past_b & (t_idx[None, :] >= cut[:, None] - hp.window)
            exc = torch.where(win, -ll - ent, torch.zeros_like(ent)).sum(1, keepdim=True)
            s = exc / win.sum(1, keepdim=True).clamp(min=1)
            gamma = torch.exp(hp.beta * s).clamp(0.25, 4.0)
        new_tr, new_c = z_tr, z_c
        if hp.d_tr:
            cols = []
            for j in range(hp.d_tr):
                e = torch.zeros_like(z)
                e[:, j] = FD_STEP
                cols.append(
                    (net.logits(net.encode(x, z_tr + e[:, : hp.d_tr]), z + e) - lg).double() / FD_STEP
                )
            g, fis = _gn_block(torch.stack(cols, dim=-1), p, y, past)
            new_tr = _newton(z_tr, g, fis, hp.sigma_tr**-2)
        if hp.d_ctx:
            cols = []
            for j in range(hp.d_ctx):
                e = torch.zeros_like(z)
                e[:, hp.d_tr + j] = FD_STEP
                cols.append((net.logits(h, z + e) - lg).double() / FD_STEP)
            g, fis = _gn_block(torch.stack(cols, dim=-1), p, y, w_c * gamma)
            new_c = _newton(z_c, g, fis, hp.sigma_ctx**-2)
        z_tr, z_c = new_tr, new_c
    return z_tr, z_c, gamma


def _infer_gd(
    net: LatentGRU,
    hp: F1Hyper,
    x: torch.Tensor,
    a: torch.Tensor,
    avail: torch.Tensor,
    scored: torch.Tensor,
    cut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CAVIA inner loop: ``k_inner`` fixed-step gradient steps on the mean past NLL from z = 0."""
    R, T = a.shape
    past_b = scored & (torch.arange(T)[None, :] < cut[:, None])
    n = past_b.sum(1).clamp(min=1).to(x.dtype)
    dz = hp.d_tr + hp.d_ctx
    z = torch.zeros(R, dz)
    if hp.k_inner > 0 and bool(past_b.any()):
        with torch.enable_grad():
            for _ in range(hp.k_inner):
                zz = z.detach().requires_grad_(True)
                lp = net.readout(net.encode(x, zz[:, : hp.d_tr]), zz, avail)
                loss = -(_chosen(lp, a, past_b).sum(1) / n).sum()
                (g,) = torch.autograd.grad(loss, zz)
                z = (zz - hp.inner_lr * g).detach()
    return z[:, : hp.d_tr], z[:, hp.d_tr :], torch.ones(R, 1, dtype=torch.float64)


def infer_latent(
    net: LatentGRU,
    hp: F1Hyper,
    x: torch.Tensor,
    a: torch.Tensor,
    avail: torch.Tensor,
    scored: torch.Tensor,
    cut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer (z_trait, z_ctx) per row from that row's free choices s < cut[row]. Returns (z_tr, z_ctx, γ)."""
    fn = _infer_gd if hp.inference == "gd" else _infer_gn
    z_tr, z_c, g = fn(net, hp, x, a, avail, scored, cut)
    return z_tr.detach(), z_c.detach(), g.detach()


def rows_logp(
    net: LatentGRU,
    hp: F1Hyper,
    x: torch.Tensor,
    a: torch.Tensor,
    avail: torch.Tensor,
    scored: torch.Tensor,
    cut: torch.Tensor,
) -> torch.Tensor:
    """Log-probabilities per row with the latent inferred at that row's cut (z is a constant for θ)."""
    z_tr, z_c, _ = infer_latent(net, hp, x, a, avail, scored, cut)
    h = net.encode(x, z_tr)
    return net.readout(h, torch.cat([z_tr, z_c], dim=1), avail)


def _last_scored(b: nets.Batch) -> list[int]:
    m = b.mask.numpy()
    return [int(np.flatnonzero(row).max()) + 1 if row.any() else 1 for row in m]


def adapt_nll(net: LatentGRU, hp: F1Hyper, b: nets.Batch, max_rows: int = 96) -> tuple[torch.Tensor, int]:
    """Adapt-mode NLL sum and count for a batch (same schedule as the harness: cuts 0, m, 2m, …)."""
    lengths = _last_scored(b)
    rows = [(i, c) for i, n in enumerate(lengths) for c in range(0, n, hp.m)]
    tot = torch.zeros(())
    cnt = 0
    with torch.no_grad():
        for r0 in range(0, len(rows), max_rows):
            chunk = rows[r0 : r0 + max_rows]
            idx = torch.tensor([i for i, _ in chunk])
            cut = torch.tensor([c for _, c in chunk])
            tc = min(b.T, int(cut.max()) + hp.m)
            x, a, av, sc = b.x[idx, :tc], b.a[idx, :tc], b.avail[idx, :tc], b.mask[idx, :tc]
            lp = rows_logp(net, hp, x, a, av, sc, cut)
            t_idx = torch.arange(tc)
            outer = sc & (t_idx[None, :] >= cut[:, None]) & (t_idx[None, :] < cut[:, None] + hp.m)
            tot = tot - _chosen(lp, a, outer).sum()
            cnt += int(outer.sum())
    return tot, cnt


class F1Latent(AdaptiveBase):
    adaptive = True

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int, *, name: str = "f1_latent") -> None:
        self.name = name
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        c = self.config
        self.hp = F1Hyper(
            d_tr=int(c.get("d_trait", 2)),
            d_ctx=int(c.get("d_ctx", 2)),
            sigma_tr=float(c.get("sigma_trait", 0.5)),
            sigma_ctx=float(c.get("sigma_ctx", 1.0)),
            k_inner=int(c.get("k_inner", 2)),
            m=int(c.get("m", 10)),
            halflife=float(c.get("ctx_halflife", 20.0)),
            beta=float(c.get("beta", 2.0)),
            window=int(c.get("surprise_window", 20)),
            inference=str(c.get("inference", "gn")),
            inner_lr=float(c.get("inner_lr", 1.0)),
        )
        self.update_every = self.hp.m
        self.meta = bool(c.get("meta", True))
        self.net = LatentGRU(
            d_in, int(c.get("hidden", 32)), k_max, self.hp.d_tr, self.hp.d_ctx, float(c.get("dropout", 0.0))
        )

    # ------------------------------------------------------------------ fitting
    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        t0 = time.process_time()
        pop, blog = common.train_backbone(
            train, self.config, self.k_max, self.d_in, threads=threads, budget_s=budget_s
        )
        nets.setup_torch(seed, threads, self.name, repr(sorted(self.config.items())))
        hp, c = self.hp, self.config
        self.net = LatentGRU(
            self.d_in, int(c.get("hidden", 32)), self.k_max, hp.d_tr, hp.d_ctx, float(c.get("dropout", 0.0))
        )
        self.net.load_backbone(pop)
        log: dict[str, Any] = {"backbone": blog, "meta": self.meta, "inner_loop": hp.inference}
        if not self.meta:
            self.net.eval()
            log["cpu_s_incl_backbone"] = round(
                time.process_time() - t0 + (blog["backbone_cpu_s"] if blog["cached"] else 0.0), 2
            )
            return log
        rng = stream(seed, "f1_cuts", self.name)
        n_cuts = int(c.get("train_cuts", 2))
        train_backbone = bool(c.get("meta_train_backbone", False))
        w_pop = float(c.get("pop_weight", 0.5)) if train_backbone else 0.0
        hooks = [] if train_backbone else self.net.freeze_backbone_part()
        log["meta_train_backbone"] = train_backbone
        log["meta_gradient"] = "first-order"

        def meta_loss(net: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            assert isinstance(net, LatentGRU)
            lengths = _last_scored(b)
            rows: list[tuple[int, int]] = []
            for i, n in enumerate(lengths):
                cuts = list(range(0, n, hp.m))
                pick = rng.choice(len(cuts), size=min(n_cuts, len(cuts)), replace=False)
                rows += [(i, cuts[int(j)]) for j in sorted(pick)]
            idx = torch.tensor([i for i, _ in rows])
            cut = torch.tensor([cc for _, cc in rows])
            tc = min(b.T, int(cut.max()) + hp.m)
            x, a, av, sc = b.x[idx, :tc], b.a[idx, :tc], b.avail[idx, :tc], b.mask[idx, :tc]
            lp = rows_logp(net, hp, x, a, av, sc, cut)
            t_idx = torch.arange(tc)
            outer = sc & (t_idx[None, :] >= cut[:, None]) & (t_idx[None, :] < cut[:, None] + hp.m)
            k_out = int(outer.sum())
            loss = -_chosen(lp, a, outer).sum() / max(k_out, 1)
            if w_pop > 0:
                zt = torch.zeros(b.B, hp.d_tr)
                z = torch.zeros(b.B, hp.d_tr + hp.d_ctx)
                lp0 = net.readout(net.encode(b.x, zt), z, b.avail)
                s0, k0 = nets.nll_sum(lp0, b)
                loss = loss + w_pop * s0 / max(k0, 1)
            return loss * max(k_out, 1), max(k_out, 1)

        def val_loss(net: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            assert isinstance(net, LatentGRU)
            return adapt_nll(net, hp, b)

        tk = common.trainer_kwargs(c)
        tk.update(
            lr=float(c.get("meta_lr", 0.01 if not train_backbone else 0.003)),
            max_epochs=int(c.get("meta_epochs", 30)),
            patience=int(c.get("meta_patience", 6)),
            batch_size=int(c.get("meta_batch", 8)),
        )
        spent = time.process_time() - t0
        mlog = nets.train(
            self.net,
            meta_loss,
            train,
            seed=seed,
            key=self.name,
            budget_s=max(30.0, budget_s - spent),
            val_loss_fn=val_loss,
            plateau=int(c.get("meta_plateau", 3)),
            **tk,
        )
        for hk in hooks:
            hk.remove()
        self.net.eval()
        log["meta_train"] = mlog
        own = time.process_time() - t0
        log["cpu_s_incl_backbone"] = round(own + (blog["backbone_cpu_s"] if blog["cached"] else 0.0), 2)
        return log

    # ------------------------------------------------------------------ person-level state
    def init_state(self) -> dict[str, Any]:
        return {
            "z_trait": np.zeros(self.hp.d_tr, dtype=np.float32),
            "z_ctx": np.zeros(self.hp.d_ctx, dtype=np.float32),
            "gamma": 1.0,
            "n": 0,
        }

    def update_state(self, state: Any, history: EncodedSeq) -> dict[str, Any]:
        if history.T == 0 or not history.scored.any():
            return self.init_state()
        b = nets.Batch([history])
        z_tr, z_c, g = infer_latent(self.net, self.hp, b.x, b.a, b.avail, b.mask, torch.tensor([history.T]))
        return {
            "z_trait": z_tr[0].numpy().copy(),
            "z_ctx": z_c[0].numpy().copy(),
            "gamma": float(g[0, 0]),
            "n": int(history.n_scored),
        }

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        bounds = [s for s, _ in states] + [seq.T]
        segs = [(s, e, st) for (s, st), e in zip(states, bounds[1:], strict=True) if e > s]
        out = np.full((seq.T, seq.k_max), -np.inf)
        b = nets.Batch([seq])
        with torch.no_grad():
            for r0 in range(0, len(segs), 32):
                chunk = segs[r0 : r0 + 32]
                n = len(chunk)
                tc = max(e for _, e, _ in chunk)
                z_tr = torch.from_numpy(
                    np.stack([np.asarray(st["z_trait"], dtype=np.float32) for _, _, st in chunk])
                )
                z_c = torch.from_numpy(
                    np.stack([np.asarray(st["z_ctx"], dtype=np.float32) for _, _, st in chunk])
                )
                x = b.x[:, :tc].expand(n, -1, -1)
                av = b.avail[:, :tc].expand(n, -1, -1)
                lp = self.net.readout(self.net.encode(x, z_tr), torch.cat([z_tr, z_c], dim=1), av)
                lpn = lp.double().numpy()
                for r, (s, e, _) in enumerate(chunk):
                    out[s:e] = lpn[r, s:e]
        return np.where(seq.avail, out, -np.inf)

    def twin_config(self) -> dict[str, Any]:
        return common.twin_config(self.config)

    def param_count(self) -> int:
        return nets.n_params(self.net)

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": self.hp.d_tr + self.hp.d_ctx,
            "per_person_state": "latent z = [z_trait, z_ctx] re-inferred every m trials",
            "inner_loop": (
                "k_inner fixed-step gradient steps (CAVIA)"
                if self.hp.inference == "gd"
                else "k_inner Gauss-Newton steps (forward-difference Jacobian)"
            ),
            "meta_gradient": "first-order",
            "meta_trained": self.meta,
            "update_every": self.update_every,
            "config": self.config,
        }
