"""Meta-reinforcement-learning comparators (RL²-style: Duan et al. 2016; Wang et al. 2016).

Two architectures, both trained by advantage actor-critic on many training worlds and frozen at evaluation
(adaptation within a world happens only in the recurrent state):

* ``gru`` — a flat GRU over [all appearances, previous action, outcome, reward, time] (attempt 1's comparator);
* ``rel`` — a permutation-equivariant relational agent (R1, critique F5): one shared GRU cell per entity fed with
  that entity's appearance and its own interaction history (including the partner's appearance for pair actions),
  self-attention across entities, probe logits from each entity state and symmetric pair logits from each pair of
  states.

Two training distributions: ``train`` (the narrow distribution IRA trains on) and ``train_broad`` (R1, critique F3:
the kinds of variation IRA's priors admit — cue validity ρ ~ U(0.3, 1), switches at random times in 25% of worlds,
training-only extra roles in 25% of worlds — while the tested values stay held out: ρ = 0.1, the unseen role 5, and
4–5 distinct roles).
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
    KNOWN_ROLES,
    N_ACT,
    N_BITS,
    N_ENT,
    P_HI,
    P_LO,
    PAIRS,
    Universe,
    is_pair,
    make_world,
)

OBS_DIM = (
    N_ENT * D_APP + N_ACT + 1 + N_BITS + 1 + 1 + 1
)  # appearance, last action, pair outcome, bits, r, t, first
ENT_DIM = (
    D_APP + 1 + 1 + N_BITS + 1 + 1 + D_APP + 1 + 1
)  # own x, probed, paired, bits, reaction, r, partner x, t, first
PI = np.array([p[0] for p in PAIRS])
PJ = np.array([p[1] for p in PAIRS])


@dataclass
class MetaRLConfig:
    arch: str = "gru"  # "gru" | "rel"
    dist: str = "train"  # "train" | "train_broad"
    hidden: int = 128
    lr: float = 1e-3
    batch: int = 128
    updates: int = 20000
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
        self.hidden = hidden
        self.inp = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACT)
        self.v = nn.Linear(hidden, 1)

    def init_state(self, b: int) -> torch.Tensor:
        return torch.zeros(b, self.hidden)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.gru(self.inp(obs), h)
        return self.pi(h), self.v(h).squeeze(-1), h


class RelNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Sequential(nn.Linear(ENT_DIM, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.att = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.probe = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.pair = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.pi_idx = torch.from_numpy(PI)
        self.pj_idx = torch.from_numpy(PJ)

    def init_state(self, b: int) -> torch.Tensor:
        return torch.zeros(b, N_ENT, self.hidden)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = obs.shape[0]
        h = self.gru(self.inp(obs).reshape(b * N_ENT, -1), h.reshape(b * N_ENT, -1)).reshape(b, N_ENT, -1)
        a, _ = self.att(h, h, h, need_weights=False)
        z = self.ln(h + a)
        lp = self.probe(z).squeeze(-1)
        zi, zj = z[:, self.pi_idx], z[:, self.pj_idx]
        lq = self.pair(torch.cat([zi + zj, zi * zj], -1)).squeeze(-1)
        return torch.cat([lp, lq], -1), self.v(z.mean(1)).squeeze(-1), h


def make_net(arch: str, hidden: int) -> nn.Module:
    return {"gru": Net, "rel": RelNet}[arch](hidden)


def make_obs(x: np.ndarray, last_a: np.ndarray, last_o: np.ndarray, last_r: np.ndarray, t: int) -> np.ndarray:
    """Flat observation [B, OBS_DIM]. x [B, E, D]; last_a [B] (-1 = none); last_o [B] (pair 0/1 or probe code)."""
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


def make_ent_obs(
    x: np.ndarray, last_a: np.ndarray, last_o: np.ndarray, last_r: np.ndarray, t: int
) -> np.ndarray:
    """Per-entity observation [B, E, ENT_DIM] (permutation-equivariant)."""
    b = len(x)
    obs = np.zeros((b, N_ENT, ENT_DIM), dtype=np.float32)
    obs[:, :, :D_APP] = x
    off = D_APP
    rows = np.arange(b)
    probe = (last_a >= 0) & (last_a < N_ENT)
    pr = rows[probe]
    obs[pr, last_a[probe], off] = 1.0
    for bit in range(N_BITS):
        obs[pr, last_a[probe], off + 2 + bit] = ((last_o[probe] >> bit) & 1) * 2.0 - 1.0
    pair = last_a >= N_ENT
    qr = rows[pair]
    if len(qr):
        pi, pj = PI[last_a[pair] - N_ENT], PJ[last_a[pair] - N_ENT]
        for me, other in ((pi, pj), (pj, pi)):
            obs[qr, me, off + 1] = 1.0
            obs[qr, me, off + 2 + N_BITS] = last_o[pair]
            obs[qr, me, off + 3 + N_BITS] = last_r[pair]
            obs[qr, me, off + 4 + N_BITS : off + 4 + N_BITS + D_APP] = x[qr, other]
    obs[:, :, -2] = t / HORIZON
    obs[:, :, -1] = 0.0 if t > 0 else 1.0
    return obs


def observe_fn(arch: str) -> Callable[..., np.ndarray]:
    return make_obs if arch == "gru" else make_ent_obs


class TrainWorlds:
    """Pool of worlds from ``env.make_world(u, dist, seed)`` (roles, switch, ρ); appearance and outcomes are
    resampled every episode with the same rules as the environment."""

    def __init__(
        self, u: Universe, dist: str, rng: np.random.Generator, pool: int = 20000, seed0: int = 500000
    ) -> None:
        self.u, self.rng = u, rng
        worlds = [make_world(u, dist, seed0 + i) for i in range(pool)]
        self.roles = np.stack([w.roles for w in worlds])
        after = np.stack([w.roles_at(HORIZON) for w in worlds])
        self.t_switch = np.array([w.switch[0] if w.switch else HORIZON for w in worlds])
        self.rho = np.array([w.meta["rho"] for w in worlds])
        self.known = np.isin(self.roles, KNOWN_ROLES)
        self.pair_p = np.stack(
            [np.where(u.law[r[:, PI], r[:, PJ]], P_HI, P_LO) for r in (self.roles, after)]
        )  # [2, pool, 10]
        self.bit_p = np.stack(
            [np.where(u.sig[r] == 1, BIT_HI, BIT_LO) for r in (self.roles, after)]
        )  # [2,pool,E,3]

    def sample(self, b: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(len(self.roles), size=b)
        roles = self.roles[idx]
        x = self.rng.normal(size=(b, N_ENT, D_APP))
        bits = np.where(
            self.known[idx][..., None], self.u.code[roles], self.rng.integers(0, 2, size=(b, N_ENT, 3))
        )
        flip = self.rng.random((b, N_ENT, 3)) >= self.rho[idx][:, None, None]
        bits = np.where(flip, 1 - bits, bits)
        x[:, :, :3] = np.where(bits == 1, 2.0, -2.0) + 0.3 * self.rng.normal(size=(b, N_ENT, 3))
        return x, self.pair_p[:, idx], self.bit_p[:, idx], self.t_switch[idx]


def train_metarl(
    u: Universe, cfg: MetaRLConfig, log: Callable[[str], None] | None = None
) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, u.seed, 515, 0 if cfg.dist == "train" else 1])
    worlds = TrainWorlds(u, cfg.dist, rng)
    net = make_net(cfg.arch, cfg.hidden)
    obs_fn = observe_fn(cfg.arch)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    hist = []
    t0 = time.time()
    rows = np.arange(cfg.batch)
    for upd in range(cfg.updates):
        frac = upd / max(cfg.updates - 1, 1)
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * frac
        x, pair_p, bit_p, t_sw = worlds.sample(cfg.batch)
        h = net.init_state(cfg.batch)
        last_a = np.full(cfg.batch, -1)
        last_o = np.zeros(cfg.batch, dtype=int)
        last_r = np.zeros(cfg.batch, dtype=np.float32)
        logps, vals, ents, rews = [], [], [], []
        for t in range(HORIZON):
            phase = (t >= t_sw).astype(int)
            obs = torch.from_numpy(obs_fn(x, last_a, last_o, last_r, t))
            logits, v, h = net(obs, h)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            an = a.numpy()
            r = np.zeros(cfg.batch, dtype=np.float32)
            o = np.zeros(cfg.batch, dtype=int)
            pm = an >= N_ENT
            if pm.any():
                succ = rng.random(int(pm.sum())) < pair_p[phase[pm], rows[pm], an[pm] - N_ENT]
                o[pm] = succ.astype(int)
                r[pm] = succ.astype(np.float32)
            qm = ~pm
            if qm.any():
                bp = bit_p[phase[qm], rows[qm], an[qm]]  # [n, 3]
                bb = (rng.random(bp.shape) < bp).astype(int)
                o[qm] = bb[:, 0] + 2 * bb[:, 1] + 4 * bb[:, 2]
            logps.append(dist.log_prob(a))
            vals.append(v)
            ents.append(dist.entropy())
            rews.append(torch.from_numpy(r))
            last_a, last_o, last_r = an, o, r
        values = torch.stack(vals)
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
        self, net: nn.Module, arch: str, greedy: bool = True, rng: np.random.Generator | None = None
    ) -> None:
        self.net, self.greedy = net, greedy
        self.obs_fn = observe_fn(arch)
        self.rng = rng or np.random.default_rng(0)

    def begin(self, x: np.ndarray) -> None:
        self.x = x[None].astype(np.float32)
        self.h = self.net.init_state(1)
        self.last = (np.array([-1]), np.array([0]), np.array([0.0], dtype=np.float32))

    def act(self, t: int) -> int:
        with torch.no_grad():
            obs = torch.from_numpy(self.obs_fn(self.x, *self.last, t))
            logits, _, self.h = self.net(obs, self.h)
        if self.greedy:
            return int(torch.argmax(logits[0]))
        p = torch.softmax(logits[0], -1).numpy().astype(float)
        return int(self.rng.choice(N_ACT, p=p / p.sum()))

    def observe(self, t: int, a: int, o: int) -> None:
        r = float(o) if is_pair(a) else 0.0
        self.last = (np.array([a]), np.array([o]), np.array([r], dtype=np.float32))
