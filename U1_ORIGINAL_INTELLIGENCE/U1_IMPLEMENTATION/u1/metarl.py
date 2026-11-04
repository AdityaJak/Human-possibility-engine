"""Meta-reinforcement-learning comparator (RL²-style: Duan et al. 2016; Wang et al. 2016).

A GRU policy receives, at every step, all entity appearances, its previous action, the previous outcome and reward
and the time; it is trained by advantage actor-critic over many training-condition worlds (fresh worlds every
episode, far more than IRA sees). At evaluation its weights are frozen; adaptation within a world happens only in
its hidden state. Same information as IRA, no access to roles, laws, cue validity or the condition.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .env import (
    BIT_HI,
    BIT_LO,
    D_APP,
    HORIZON,
    N_ACT,
    N_BITS,
    N_ENT,
    P_HI,
    P_LO,
    PAIRS,
    Universe,
    _sample_roles,
    decode_bits,
    is_pair,
)

OBS_DIM = (
    N_ENT * D_APP + N_ACT + 1 + N_BITS + 1 + 1 + 1
)  # appearance, last action, pair outcome, bits, r, t, first


@dataclass
class MetaRLConfig:
    hidden: int = 128
    lr: float = 1e-3
    batch: int = 128
    updates: int = 12000
    ent_start: float = 0.05
    ent_end: float = 0.003
    gamma: float = 1.0
    lam: float = 0.95
    value_coef: float = 0.5
    clip: float = 1.0
    seed: int = 0


class Net(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACT)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.gru(self.inp(obs), h)
        return self.pi(h), self.v(h).squeeze(-1), h


def make_obs(x: np.ndarray, last_a: np.ndarray, last_o: np.ndarray, last_r: np.ndarray, t: int) -> np.ndarray:
    """Batch observation. x [B, E, D]; last_a [B] (-1 = none); last_o [B] (pair 0/1 or probe code 0..7)."""
    b = len(x)
    obs = np.zeros((b, OBS_DIM), dtype=np.float32)
    obs[:, : N_ENT * D_APP] = x.reshape(b, -1)
    off = N_ENT * D_APP
    has = last_a >= 0
    obs[np.flatnonzero(has), off + last_a[has]] = 1.0
    off += N_ACT
    pair = has & (last_a >= N_ENT)
    obs[pair, off] = last_o[pair]
    probe = has & (last_a < N_ENT)
    for bit in range(N_BITS):
        obs[probe, off + 1 + bit] = ((last_o[probe] >> bit) & 1) * 2.0 - 1.0
    off += 1 + N_BITS
    obs[:, off] = last_r
    obs[:, off + 1] = t / HORIZON
    obs[:, off + 2] = 0.0 if t > 0 else 1.0
    return obs


class TrainWorlds:
    """Vectorized sampler of training-condition worlds with the same distribution as ``env.make_world(u, 'train')``."""

    def __init__(self, u: Universe, rng: np.random.Generator, pool: int = 50000) -> None:
        self.u, self.rng = u, rng
        self.roles = np.stack([_sample_roles(rng, u, (2, 3)) for _ in range(pool)])
        pi, pj = np.array([p[0] for p in PAIRS]), np.array([p[1] for p in PAIRS])
        self.pair_p = np.where(u.law[self.roles[:, pi], self.roles[:, pj]], P_HI, P_LO)  # [pool, 10]
        self.bit_p = np.where(u.sig[self.roles] == 1, BIT_HI, BIT_LO)  # [pool, E, 3]

    def sample(self, b: int, fixed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = fixed if fixed is not None else self.rng.integers(len(self.roles), size=b)
        roles = self.roles[idx]
        x = self.rng.normal(size=(b, N_ENT, D_APP))
        bits = self.u.code[roles]  # [B, E, 3]
        flip = self.rng.random((b, N_ENT, 3)) >= 0.9
        bits = np.where(flip, 1 - bits, bits)
        x[:, :, :3] = np.where(bits == 1, 2.0, -2.0) + 0.3 * self.rng.normal(size=(b, N_ENT, 3))
        return x, self.pair_p[idx], self.bit_p[idx]


def train_metarl(
    u: Universe,
    cfg: MetaRLConfig,
    fixed_worlds: np.ndarray | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Net, dict]:
    """``fixed_worlds``: optional array of pool indices to restrict training to (data-matched variant)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, u.seed, 505])
    worlds = TrainWorlds(u, rng)
    net = Net(cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    hist = []
    t0 = time.time()
    for upd in range(cfg.updates):
        frac = upd / max(cfg.updates - 1, 1)
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * frac
        idx = rng.choice(fixed_worlds, size=cfg.batch) if fixed_worlds is not None else None
        x, pair_p, bit_p = worlds.sample(cfg.batch, idx)
        h = torch.zeros(cfg.batch, cfg.hidden)
        last_a = np.full(cfg.batch, -1)
        last_o = np.zeros(cfg.batch, dtype=int)
        last_r = np.zeros(cfg.batch, dtype=np.float32)
        logps, vals, ents, rews = [], [], [], []
        for t in range(HORIZON):
            obs = torch.from_numpy(make_obs(x, last_a, last_o, last_r, t))
            logits, v, h = net(obs, h)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            an = a.numpy()
            r = np.zeros(cfg.batch, dtype=np.float32)
            o = np.zeros(cfg.batch, dtype=int)
            pm = an >= N_ENT
            if pm.any():
                succ = rng.random(int(pm.sum())) < pair_p[np.flatnonzero(pm), an[pm] - N_ENT]
                o[pm] = succ.astype(int)
                r[pm] = succ.astype(np.float32)
            qm = ~pm
            if qm.any():
                bp = bit_p[np.flatnonzero(qm), an[qm]]  # [n, 3]
                bb = (rng.random(bp.shape) < bp).astype(int)
                o[qm] = bb[:, 0] + 2 * bb[:, 1] + 4 * bb[:, 2]
            logps.append(dist.log_prob(a))
            vals.append(v)
            ents.append(dist.entropy())
            rews.append(torch.from_numpy(r))
            last_a, last_o, last_r = an, o, r
        values = torch.stack(vals)  # [T, B]
        rewards = torch.stack(rews)
        with torch.no_grad():
            adv = torch.zeros_like(rewards)
            gae = torch.zeros(cfg.batch)
            for t in reversed(range(HORIZON)):
                nv = values[t + 1] if t + 1 < HORIZON else torch.zeros(cfg.batch)
                delta = rewards[t] + cfg.gamma * nv - values[t]
                gae = delta + cfg.gamma * cfg.lam * gae
                adv[t] = gae
            ret = adv + values
        logp = torch.stack(logps)
        ent = torch.stack(ents)
        loss = -(logp * adv).mean() + cfg.value_coef * ((values - ret) ** 2).mean() - ent_coef * ent.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        opt.step()
        hist.append(float(rewards.sum(0).mean()))
        if log is not None and (upd + 1) % 1000 == 0:
            log(
                f"update {upd + 1}: mean return (last 1000) {np.mean(hist[-1000:]):.3f} ({time.time() - t0:.0f}s)"
            )
    info = {
        "config": asdict(cfg),
        "train_return_last1000": float(np.mean(hist[-1000:])),
        "seconds": time.time() - t0,
        "parameters": sum(p.numel() for p in net.parameters()),
    }
    return net, info


class MetaRLAgent:
    """Frozen meta-trained policy acting in one world (greedy or sampled actions)."""

    def __init__(
        self, net: Net, hidden: int, greedy: bool = True, rng: np.random.Generator | None = None
    ) -> None:
        self.net, self.hidden, self.greedy = net, hidden, greedy
        self.rng = rng or np.random.default_rng(0)

    def begin(self, x: np.ndarray) -> None:
        self.x = x[None].astype(np.float32)
        self.h = torch.zeros(1, self.hidden)
        self.last = (np.array([-1]), np.array([0]), np.array([0.0], dtype=np.float32))

    def act(self, t: int) -> int:
        with torch.no_grad():
            obs = torch.from_numpy(make_obs(self.x, *self.last, t))
            logits, _, self.h = self.net(obs, self.h)
        if self.greedy:
            return int(torch.argmax(logits[0]))
        p = torch.softmax(logits[0], -1).numpy().astype(float)
        return int(self.rng.choice(N_ACT, p=p / p.sum()))

    def observe(self, t: int, a: int, o: int) -> None:
        r = float(o) if is_pair(a) else 0.0
        self.last = (np.array([a]), np.array([o]), np.array([r], dtype=np.float32))


__all__ = ["MetaRLAgent", "MetaRLConfig", "Net", "TrainWorlds", "decode_bits", "make_obs", "train_metarl"]
