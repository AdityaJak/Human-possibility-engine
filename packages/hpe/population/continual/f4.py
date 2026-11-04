"""F4 — continual meta-learner (RL²-style; Wang et al., 2016; Duan et al., 2016). Requires torch.

A GRU policy is meta-trained by advantage actor-critic across many switching sequences drawn from the
**meta-train development seeds** (with option-relabelling augmentation, a symmetry of the task). At test its
weights are frozen: all adaptation inside a sequence happens in the recurrent state, which receives only
the previous action and its realised reward (no context cue). Checkpoints are selected on meta-validation
development seeds by mean expected reward, the same criterion used for every Gate E learner.

Input at trial t: [one-hot(a_{t−1}), one-hot(a_{t−1})·r_{t−1}, r_{t−1}, first-trial flag] (the product
term makes "repeat a rewarded option" linearly available, as for the comparators' network input).
Outputs: policy logits over K options and a state value. Loss per update (B sequences × T trials):
−mean log π(a_t)·Â_t + c_v·mean(G_t − V_t)² − c_e·mean H(π_t), with generalised advantage estimates
(Schulman et al., 2016; discount ``gamma``, ``gae_lambda``, V = 0 after the last trial) normalised per batch,
value targets G_t = Â_t + V_t, and an entropy weight annealed linearly from ``ent_start`` to ``ent_end``.
``gamma = 0`` is the myopic (immediate-reward) special case.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.core.rng import derive_seed, stream
from hpe.population.continual.suite import (
    K_OPTIONS,
    SwitchingSequence,
    Trajectory,
    average_metrics,
    permute_options,
    sequence_metrics,
)


class F4Net(nn.Module):
    def __init__(self, K: int = K_OPTIONS, hidden: int = 48) -> None:
        super().__init__()
        self.K, self.hidden = K, hidden
        self.d_in = 2 * K + 2
        self.cell = nn.GRUCell(self.d_in, hidden)
        self.pi = nn.Linear(hidden, K)
        self.v = nn.Linear(hidden, 1)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.cell(x, h)
        return self.pi(h), self.v(h).squeeze(-1), h

    def first_input(self, B: int) -> torch.Tensor:
        x = torch.zeros(B, self.d_in)
        x[:, -1] = 1.0
        return x

    def next_input(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        n = a.shape[0]
        x = torch.zeros(n, self.d_in)
        x[torch.arange(n), a] = 1.0
        x[torch.arange(n), self.K + a] = r
        x[:, 2 * self.K] = r
        return x

    @torch.no_grad()
    def policy_logprobs(self, actions: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Teacher-forced log π(· | history before t) for one fixed action/reward history ([T] each)."""
        T = len(actions)
        h = torch.zeros(1, self.hidden)
        x = self.first_input(1)
        out = np.zeros((T, self.K))
        for t in range(T):
            logits, _v, h = self.step(x, h)
            out[t] = torch.log_softmax(logits, -1)[0].double().numpy()
            x = self.next_input(
                torch.tensor([int(actions[t])]), torch.tensor([float(rewards[t])], dtype=torch.float32)
            )
        return out


def _reward_tensor(seqs: Sequence[SwitchingSequence]) -> tuple[torch.Tensor, torch.Tensor]:
    rew = torch.tensor(np.stack([s.rewards for s in seqs]), dtype=torch.float32)  # [B, T, K]
    exp = torch.tensor(np.stack([s.probs[s.contexts] for s in seqs]), dtype=torch.float32)  # [B, T, K]
    return rew, exp


def _rollout_batch(
    net: F4Net, seqs: Sequence[SwitchingSequence], gen: torch.Generator, *, train: bool
) -> dict[str, torch.Tensor]:
    rew, exp = _reward_tensor(seqs)
    B, T, _K = rew.shape
    h = torch.zeros(B, net.hidden)
    x = net.first_input(B)
    logps, vals, ents, rs, acts, exps = [], [], [], [], [], []
    ar = torch.arange(B)
    for t in range(T):
        logits, v, h = net.step(x, h)
        logp_all = torch.log_softmax(logits, -1)
        a = torch.multinomial(logp_all.detach().exp(), 1, generator=gen).squeeze(-1)
        r = rew[ar, t, a]
        if train:
            logps.append(logp_all[ar, a])
            vals.append(v)
            ents.append(-(logp_all.exp() * logp_all).sum(-1))
        rs.append(r)
        acts.append(a)
        exps.append(exp[ar, t, a])
        x = net.next_input(a, r)
    out = {"rewards": torch.stack(rs, 1), "actions": torch.stack(acts, 1), "expected": torch.stack(exps, 1)}
    if train:
        out.update(
            {"logp": torch.stack(logps, 1), "values": torch.stack(vals, 1), "ent": torch.stack(ents, 1)}
        )
    return out


def rollout_f4(
    net: F4Net, seqs: Sequence[SwitchingSequence], *, seed: int, repeats: int = 1, tag: str = "eval"
) -> list[list[Trajectory]]:
    """Frozen-weight rollouts: ``out[i][rep]`` for ``seqs[i]``. Weights are never updated here."""
    net.eval()
    gen = torch.Generator().manual_seed(derive_seed(seed, "gate_e_f4_policy", tag) % (2**63 - 1))
    batch = [s for s in seqs for _ in range(repeats)]
    with torch.no_grad():
        res = _rollout_batch(net, batch, gen, train=False)
    out: list[list[Trajectory]] = []
    for i, s in enumerate(seqs):
        row = []
        for rep in range(repeats):
            j = i * repeats + rep
            row.append(
                Trajectory(
                    s.seed,
                    res["actions"][j].numpy().astype(np.int64),
                    res["rewards"][j].double().numpy(),
                    res["expected"][j].double().numpy(),
                )
            )
        out.append(row)
    return out


def evaluate_f4(
    net: F4Net, seqs: Sequence[SwitchingSequence], *, seed: int, repeats: int = 1, tag: str = "eval"
) -> dict[int, dict[str, float]]:
    trajs = rollout_f4(net, seqs, seed=seed, repeats=repeats, tag=tag)
    return {
        s.seed: average_metrics([sequence_metrics(s, tr) for tr in row])
        for s, row in zip(seqs, trajs, strict=True)
    }


def gae(r: torch.Tensor, v: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Generalised advantage estimates over the time axis ([B, T]); V after the last trial is 0."""
    adv = torch.zeros_like(r)
    acc = torch.zeros(r.shape[0])
    T = r.shape[1]
    for t in range(T - 1, -1, -1):
        v_next = v[:, t + 1] if t + 1 < T else torch.zeros(r.shape[0])
        delta = r[:, t] + gamma * v_next - v[:, t]
        acc = delta + gamma * lam * acc
        adv[:, t] = acc
    return adv


def train_f4(
    train_seqs: Sequence[SwitchingSequence],
    val_seqs: Sequence[SwitchingSequence],
    *,
    seed: int,
    hidden: int = 48,
    lr: float = 1e-3,
    gamma: float = 0.9,
    gae_lambda: float = 0.5,
    batch: int = 32,
    updates: int = 1500,
    ent_start: float = 0.05,
    ent_end: float = 0.005,
    value_coef: float = 0.5,
    eval_every: int = 100,
    val_repeats: int = 2,
    budget_s: float = 2700.0,
    threads: int = 1,
    log_fn: Any = None,
) -> tuple[F4Net, dict[str, Any]]:
    """Meta-train F4 with A2C; keep the checkpoint with the best meta-validation mean expected reward."""
    torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(derive_seed(seed, "gate_e_f4_init", hidden) % (2**31 - 1))
    torch.use_deterministic_algorithms(True)
    net = F4Net(K_OPTIONS, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(derive_seed(seed, "gate_e_f4_train_policy") % (2**63 - 1))
    rng = stream(seed, "gate_e_f4_batches")
    t0 = time.process_time()
    history: list[dict[str, float]] = []
    best: tuple[float, int, dict[str, torch.Tensor]] | None = None
    stopped = "updates"

    def validate(u: int) -> None:
        nonlocal best
        m = evaluate_f4(net, val_seqs, seed=seed, repeats=val_repeats, tag=f"val{u}")
        agg = {k: float(np.nanmean([v[k] for v in m.values()])) for k in next(iter(m.values()))}
        rec = {
            "update": u,
            "cpu_s": round(time.process_time() - t0, 1),
            **{k: round(v, 4) for k, v in agg.items()},
        }
        history.append(rec)
        if log_fn is not None:
            log_fn(rec)
        if best is None or agg["reward_fraction"] > best[0] + 1e-9:
            best = (agg["reward_fraction"], u, copy.deepcopy(net.state_dict()))
        net.train()

    validate(0)
    for u in range(1, updates + 1):
        idx = rng.integers(0, len(train_seqs), size=batch)
        perms = [rng.permutation(K_OPTIONS) for _ in range(batch)]
        seqs = [permute_options(train_seqs[int(i)], p) for i, p in zip(idx, perms, strict=True)]
        net.train()
        out = _rollout_batch(net, seqs, gen, train=True)
        v_det = out["values"].detach()
        adv = gae(out["rewards"], v_det, gamma, gae_lambda)
        G = adv + v_det
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        frac = (u - 1) / max(1, updates - 1)
        c_e = ent_start + (ent_end - ent_start) * frac
        loss = (
            -(out["logp"] * adv).mean()
            + value_coef * ((G - out["values"]) ** 2).mean()
            - c_e * out["ent"].mean()
        )
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if u % eval_every == 0 or u == updates:
            validate(u)
        if time.process_time() - t0 > budget_s:
            stopped = "budget"
            if u % eval_every != 0:
                validate(u)
            break
    assert best is not None
    net.load_state_dict(best[2])
    net.eval()
    return net, {
        "best_update": best[1],
        "best_val_reward_fraction": best[0],
        "history": history,
        "stopped": stopped,
        "cpu_s": round(time.process_time() - t0, 1),
        "params": int(sum(p.numel() for p in net.parameters())),
        "config": {
            "hidden": hidden,
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "batch": batch,
            "updates": updates,
            "ent_start": ent_start,
            "ent_end": ent_end,
            "value_coef": value_coef,
        },
    }
