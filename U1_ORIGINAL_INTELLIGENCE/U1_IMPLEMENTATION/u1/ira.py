"""IRA — interventional role abstraction (MATHEMATICAL_FORMULATION.md §2).

Exact Bayesian inference over concept assignments ``z`` (one label per entity) and cue reliability ``w`` by
enumeration, with Beta-Bernoulli laws collapsed per concept and probe bit, and per concept pair (reaction). Fresh
labels are exchangeable, so assignments are enumerated in canonical form (fresh labels appear in order of first use)
and the prior of a canonical assignment carries its number of equivalent labelings.

The agent sees appearances ``x``, its own actions, outcomes and rewards — nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np

from .env import D_APP, HORIZON, N_BITS, N_ENT, PAIRS, action_entities, decode_bits, is_pair

W_GRID = np.array([0.0, 0.5, 1.0])
PRUNE = 14.0
EPS = 1e-12


def _hb(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _h(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis)


@lru_cache(maxsize=64)
def _space(c_known: int, m_fresh: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical assignments [Z, E]; law index per pair [Z, 10]; per probe bit [Z, E, 3]; log #labelings [Z]."""
    c = c_known + m_fresh
    rows = []
    used_list = []
    for z in product(range(c), repeat=N_ENT):
        nxt, ok, used = c_known, True, 0
        for v in z:
            if v >= c_known:
                if v > nxt:
                    ok = False
                    break
                if v == nxt:
                    nxt += 1
                    used += 1
        if ok:
            rows.append(z)
            used_list.append(used)
    digits = np.array(rows, dtype=np.int64)
    lut = np.zeros((c, c), dtype=np.int64)
    k = N_BITS * c
    for i in range(c):
        for j in range(i, c):
            lut[i, j] = lut[j, i] = k
            k += 1
    law_pair = np.stack([lut[digits[:, i], digits[:, j]] for i, j in PAIRS], 1)
    law_bit = N_BITS * digits[:, :, None] + np.arange(N_BITS)[None, None, :]
    logperm = np.array([sum(np.log(m_fresh - q) for q in range(u)) for u in used_list])
    return digits, law_pair, law_bit, logperm


@dataclass
class Library:
    """Persistent concept knowledge: additive sufficient statistics (never retrained)."""

    bits: list[np.ndarray] = field(default_factory=list)  # per concept [N_BITS, 2] (on, off) counts
    react: dict[tuple[int, int], list[float]] = field(default_factory=dict)  # (k<=l) -> [S, F]
    app: list[np.ndarray] = field(default_factory=list)  # per concept [D_APP, 2] sign counts (neg, pos)
    n: list[float] = field(default_factory=list)  # membership counts
    pooled: list[float] = field(default_factory=lambda: [0.0, 0.0])  # pooled reaction [S, F]
    worlds: int = 0

    @property
    def k(self) -> int:
        return len(self.n)

    def rbar(self) -> float:
        s, f = self.pooled
        return (s + 0.5) / (s + f + 1.0)

    def add_concept(self) -> int:
        self.bits.append(np.zeros((N_BITS, 2)))
        self.app.append(np.zeros((D_APP, 2)))
        self.n.append(0.0)
        return self.k - 1

    def copy(self) -> Library:
        return Library(
            [b.copy() for b in self.bits],
            {k: v[:] for k, v in self.react.items()},
            [a.copy() for a in self.app],
            self.n[:],
            self.pooled[:],
            self.worlds,
        )

    def to_json(self) -> dict:
        return {
            "bits": [b.tolist() for b in self.bits],
            "react": {f"{k[0]},{k[1]}": v for k, v in sorted(self.react.items())},
            "app": [a.tolist() for a in self.app],
            "n": self.n,
            "pooled": self.pooled,
            "worlds": self.worlds,
        }

    @staticmethod
    def from_json(d: dict) -> Library:
        return Library(
            [np.array(b) for b in d["bits"]],
            {(int(k.split(",")[0]), int(k.split(",")[1])): list(v) for k, v in d["react"].items()},
            [np.array(a) for a in d["app"]],
            list(d["n"]),
            list(d["pooled"]),
            int(d["worlds"]),
        )


@dataclass
class IRAConfig:
    beta: float = 1.0  # information-gain weight (tuned on the training-like condition only)
    tau: float = 4.0  # surprise threshold (nats over an entity's last two observations)
    alpha: float = 1.0  # concentration for new concepts
    k_top: int = 7
    m_fresh: int = 2
    active: bool = True  # False -> Thompson sampling of z (ira_noactive)
    open_set: bool = True  # False -> no fresh labels when the library is non-empty (ira_closed)
    cue_track: bool = True  # False -> w fixed at 1 (ira_nocuetrack)
    change: bool = True  # False -> no re-identification (ira_nochange)


class IRA:
    """One world's inference and action selection on top of a library."""

    def __init__(self, lib: Library, cfg: IRAConfig, rng: np.random.Generator) -> None:
        self.lib, self.cfg, self.rng = lib, cfg, rng

    # ----------------------------------------------------------------------------------------------- world start
    def begin(self, x: np.ndarray) -> None:
        lib, cfg = self.lib, self.cfg
        order = sorted(range(lib.k), key=lambda k: (-lib.n[k], k))[: cfg.k_top]
        order = [k for k in order if lib.n[k] > 0]
        self.known = order
        ck = len(order)
        m = cfg.m_fresh + (1 if ck < 2 else 0) if (cfg.open_set or ck == 0) else 0
        self.ck, self.m = ck, m
        c = ck + m
        self.digits, self.law_pair, self.law_bit, logperm = _space(ck, m)
        n_law = N_BITS * c + c * (c + 1) // 2
        rb = lib.rbar()
        a = np.empty(n_law)
        b = np.empty(n_law)
        for lab in range(c):
            for bit in range(N_BITS):
                on, off = lib.bits[order[lab]][bit] if lab < ck else (0.0, 0.0)
                a[N_BITS * lab + bit], b[N_BITS * lab + bit] = 0.5 + on, 0.5 + off
        k = N_BITS * c
        for i in range(c):
            for j in range(i, c):
                s, f = 0.0, 0.0
                if i < ck and j < ck:
                    s, f = lib.react.get((min(order[i], order[j]), max(order[i], order[j])), [0.0, 0.0])
                a[k], b[k] = rb + s, 1 - rb + f
                k += 1
        self.a_law, self.b_law = a, b
        ntot = sum(lib.n[k] for k in order)
        logpz = np.empty(c)
        for lab in range(c):
            if lab < ck:
                logpz[lab] = np.log(lib.n[order[lab]] / (ntot + cfg.alpha))
            else:
                logpz[lab] = np.log(cfg.alpha / m / (ntot + cfg.alpha))
        self.logprior = logpz[self.digits].sum(1) + logperm
        s_app = (x > 0).astype(int)  # [E, D]
        q = np.full((c, D_APP), 0.5)
        for lab in range(ck):
            cnt = lib.app[order[lab]]
            q[lab] = (1 + cnt[:, 1]) / (2 + cnt.sum(1))
        w = W_GRID if cfg.cue_track else np.array([1.0])
        self.w = w
        pq = np.where(s_app[:, None, :] == 1, q[None], 1 - q[None])  # [E, C, D]
        like = np.log(w[None, None, None, :] * pq[..., None] + (1 - w) * 0.5).sum(2)  # [E, C, W]
        self.logapp = like[np.arange(N_ENT)[None, :], self.digits].sum(1)  # [Z, W]
        nz = len(self.digits)
        self.sz = np.zeros((nz, n_law), dtype=np.int32)
        self.fz = np.zeros((nz, n_law), dtype=np.int32)
        self.loglik = np.zeros(nz)
        self.obs: list[tuple[int, int, int]] = []
        self.active_obs: list[bool] = []
        self.surp: dict[int, list[float]] = {e: [] for e in range(N_ENT)}
        self.n_resets = 0
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ----------------------------------------------------------------------------------------------- inference
    def _logpost(self) -> np.ndarray:
        return self.logprior[:, None] + self.logapp + self.loglik[:, None]

    def posterior(self) -> np.ndarray:
        lp = self._logpost()
        pz = np.exp(lp - lp.max()).sum(1)
        return pz / pz.sum()

    def w_posterior(self) -> np.ndarray:
        lp = self._logpost()
        p = np.exp(lp - lp.max()).sum(0)
        return p / p.sum()

    def _rate(self, rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
        r = rows.reshape((-1,) + (1,) * (idx.ndim - 1))
        s, f = self.sz[r, idx], self.fz[r, idx]
        a, b = self.a_law[idx], self.b_law[idx]
        return (a + s) / (a + b + s + f)

    def _predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior over the active assignments [Z']; pair success [Z', 10]; probe-bit rates [Z', E, 3].

        Assignments more than ``PRUNE`` nats below the best (< 1e-6 of the mass each) are skipped when predicting;
        they keep being updated, so they can return.
        """
        if self._cache is None:
            lp = self._logpost()
            lz = np.log(np.exp(lp - lp.max()).sum(1))
            top = lz.max()
            rows = np.flatnonzero(lz > top - PRUNE)
            pz = np.exp(lz[rows] - top)
            pz = pz / pz.sum()
            pp = self._rate(rows, self.law_pair[rows])
            pb = self._rate(rows, self.law_bit[rows])
            self._cache = (pz, pp, pb)
        return self._cache

    def act(self, t: int) -> int:
        pz, pp, pb = self._predict()
        phat = pz @ pp  # [10]
        if not self.cfg.active:
            zi = int(self.rng.choice(len(pz), p=pz))
            return int(N_ENT + np.argmax(pp[zi]))
        ig_pair = _hb(phat) - pz @ _hb(pp)
        # a probe reads 3 bits that are independent given z, so H(o | z) is the sum of the bit entropies
        b = [np.stack([1 - pb[..., i], pb[..., i]], -1) for i in range(N_BITS)]  # each [Z', E, 2]
        pout = np.einsum("z,zea,zeb,zec->ecba", pz, b[0], b[1], b[2], optimize=True).reshape(N_ENT, 8)
        ig_probe = _h(pout) - pz @ _hb(pb).sum(-1)
        beta_t = self.cfg.beta * (1 - t / HORIZON)
        score = np.concatenate([beta_t * ig_probe, phat + beta_t * ig_pair])
        return int(np.argmax(score))

    def _apply(self, a: int, o: int) -> None:
        r = np.arange(len(self.digits))
        if is_pair(a):
            lws = [(self.law_pair[:, a - N_ENT], o)]
        else:
            bits = decode_bits(o)
            lws = [(self.law_bit[:, a, b], int(bits[b])) for b in range(N_BITS)]
        for lw, ob in lws:
            s, f = self.sz[r, lw], self.fz[r, lw]
            aa, bb = self.a_law[lw], self.b_law[lw]
            p = (aa + s) / (aa + bb + s + f)
            self.loglik += np.log(p if ob else 1 - p)
            self.sz[r, lw] += ob
            self.fz[r, lw] += 1 - ob
        self._cache = None

    def predictive(self, a: int, o: int) -> float:
        pz, pp, pb = self._predict()
        if is_pair(a):
            ph = float(pz @ pp[:, a - N_ENT])
            return ph if o else 1 - ph
        bits = decode_bits(o)
        q = np.where(bits[None, :] == 1, pb[:, a, :], 1 - pb[:, a, :]).prod(-1)
        return float(pz @ q)

    def observe(self, t: int, a: int, o: int) -> None:
        u = -np.log(max(self.predictive(a, o), EPS))
        self._apply(a, o)
        self.obs.append((t, a, o))
        self.active_obs.append(True)
        if not self.cfg.change:
            return
        for e in action_entities(a):
            self.surp[e].append(u)
            if len(self.surp[e]) >= 2 and sum(self.surp[e][-2:]) > self.cfg.tau:
                self._reset(e)

    def _reset(self, e: int) -> None:
        """Contradiction: discard the entity's evidence before its last two observations and re-infer."""
        idx = [i for i, (_, a, _) in enumerate(self.obs) if self.active_obs[i] and e in action_entities(a)]
        for i in idx[:-2]:
            self.active_obs[i] = False
        self.surp[e] = []
        self.n_resets += 1
        self.sz[:] = 0
        self.fz[:] = 0
        self.loglik[:] = 0
        for i, (_, a, o) in enumerate(self.obs):
            if self.active_obs[i]:
                self._apply(a, o)

    # ----------------------------------------------------------------------------------------------- after a world
    def active_observations(self) -> list[tuple[int, int]]:
        return [(a, o) for i, (_, a, o) in enumerate(self.obs) if self.active_obs[i]]

    def map_assignment(self) -> tuple[np.ndarray, np.ndarray]:
        pz = self.posterior()
        zstar = self.digits[int(np.argmax(pz))]
        conf = np.array([pz[self.digits[:, e] == zstar[e]].sum() for e in range(N_ENT)])
        return zstar, conf

    def consolidate(self, x: np.ndarray) -> list[int]:
        """Add the world's (non-discarded) evidence to the library under the MAP assignment.

        Entities with at least one observation are consolidated; fresh labels they use become new concepts.
        Returns the library concept per entity (-1 = not consolidated).
        """
        lib = self.lib
        zstar, _ = self.map_assignment()
        concept: dict[int, int] = {}

        def cid(e: int) -> int:
            lab = int(zstar[e])
            if lab not in concept:
                concept[lab] = self.known[lab] if lab < self.ck else lib.add_concept()
            return concept[lab]

        touched: set[int] = set()
        for i, (_, a, o) in enumerate(self.obs):
            if not self.active_obs[i]:
                continue
            ents = action_entities(a)
            touched.update(ents)
            if is_pair(a):
                lib.pooled[0 if o else 1] += 1
                ki, kj = cid(ents[0]), cid(ents[1])
                lib.react.setdefault((min(ki, kj), max(ki, kj)), [0.0, 0.0])[0 if o else 1] += 1
            else:
                bits = decode_bits(o)
                k = cid(ents[0])
                lib.bits[k][np.arange(N_BITS), 1 - bits] += 1
        for e in sorted(touched):
            k = cid(e)
            lib.n[k] += 1
            lib.app[k][np.arange(D_APP), (x[e] > 0).astype(int)] += 1
        lib.worlds += 1
        return [concept[int(zstar[e])] if e in touched else -1 for e in range(N_ENT)]
