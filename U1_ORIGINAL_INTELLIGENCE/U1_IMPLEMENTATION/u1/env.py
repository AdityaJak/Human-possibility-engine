"""Hidden-law entity worlds (MATHEMATICAL_FORMULATION.md §1).

A universe fixes hidden causal laws (role glow propensities and a symmetric role-pair reaction law). A world is one
episode with five entities whose roles are hidden; appearances are fresh noise except for a cue dimension whose
validity depends on the condition. Agents see appearances, their own actions, outcomes and rewards only.

Outcome noise uses common random numbers: ``u[t, a]`` is drawn once per world, so every agent that takes action ``a``
at step ``t`` receives the same outcome — paired comparisons between agents are therefore exact on the noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

N_ENT = 5
D_APP = 4
HORIZON = 20
PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(N_ENT), 2))  # 10 pairs
N_ACT = N_ENT + len(PAIRS)  # 5 pokes + 10 pairs
P_HI, P_LO = 0.9, 0.05
GLOW_HI, GLOW_LO = 0.9, 0.1
N_ROLES = 6
KNOWN_ROLES = (0, 1, 2, 3, 4)
NOVEL_ROLE = 5
CUE_GROUP = (0, 1)  # roles whose cue sign is + (training correlation)
SWITCH_T = 10
CONDITIONS = ("train", "iid", "cue_reversed", "recombined", "novel_role", "role_switch")
SHIFT_CONDITIONS = ("cue_reversed", "recombined", "novel_role", "role_switch")


def action_entities(a: int) -> tuple[int, ...]:
    return (a,) if a < N_ENT else PAIRS[a - N_ENT]


def is_pair(a: int) -> bool:
    return a >= N_ENT


@dataclass(frozen=True)
class Universe:
    seed: int
    glow: np.ndarray  # [N_ROLES] glow probability per role
    law: np.ndarray  # [N_ROLES, N_ROLES] bool, symmetric: pair reacts

    def pair_prob(self, r: int, s: int) -> float:
        return P_HI if self.law[r, s] else P_LO


def make_universe(seed: int) -> Universe:
    """Random laws subject to the constraints every condition needs (documented in EXPERIMENTS_AND_BASELINES.md)."""
    rng = np.random.default_rng([seed, 101])
    for _ in range(100000):
        glow = np.where(rng.random(N_ROLES) < 0.5, GLOW_HI, GLOW_LO)
        law = np.zeros((N_ROLES, N_ROLES), dtype=bool)
        for a in range(N_ROLES):
            for b in range(a, N_ROLES):
                law[a, b] = law[b, a] = bool(rng.random() < 0.3)
        k = law[:5, :5]
        ok_known = all(k[r].any() for r in KNOWN_ROLES) and len(set(glow[:5].tolist())) == 2
        # a non-reacting pair of known roles that the novel role reacts with (needed for the novel_role condition)
        ok_novel = any(
            not law[a, a] and not law[b, b] and not law[a, b] and (law[NOVEL_ROLE, a] or law[NOVEL_ROLE, b])
            for a in KNOWN_ROLES
            for b in KNOWN_ROLES
            if a < b
        )
        if ok_known and ok_novel and law[NOVEL_ROLE, :5].any():
            return Universe(seed, glow, law)
    raise RuntimeError("no universe satisfied the constraints")


@dataclass
class World:
    condition: str
    roles: np.ndarray  # [N_ENT]
    x: np.ndarray  # [N_ENT, D_APP]
    switch: tuple[int, int, int] | None = None  # (t, entity, new_role)
    meta: dict = field(default_factory=dict)

    def roles_at(self, t: int) -> np.ndarray:
        r = self.roles.copy()
        if self.switch is not None and t >= self.switch[0]:
            r[self.switch[1]] = self.switch[2]
        return r


def true_probs(u: Universe, roles: np.ndarray) -> np.ndarray:
    p = np.zeros(N_ACT)
    for e in range(N_ENT):
        p[e] = u.glow[roles[e]]
    for k, (i, j) in enumerate(PAIRS):
        p[N_ENT + k] = u.pair_prob(int(roles[i]), int(roles[j]))
    return p


def _reacting(u: Universe, roles: np.ndarray) -> list[int]:
    return [k for k, (i, j) in enumerate(PAIRS) if u.law[roles[i], roles[j]]]


def _appearance(rng: np.random.Generator, roles: np.ndarray, rho: float) -> np.ndarray:
    x = rng.normal(size=(N_ENT, D_APP))
    for e, r in enumerate(roles):
        if r == NOVEL_ROLE:
            sign = 1.0 if rng.random() < 0.5 else -1.0
        else:
            sign = 1.0 if r in CUE_GROUP else -1.0
            if rng.random() >= rho:
                sign = -sign
        x[e, 0] = 2.0 * sign + 0.3 * rng.normal()
    return x


def _distinct_roles_world(rng: np.random.Generator, u: Universe, n_distinct: tuple[int, ...]) -> np.ndarray:
    for _ in range(100000):
        k = int(rng.choice(n_distinct))
        subset = rng.choice(np.array(KNOWN_ROLES), size=k, replace=False)
        roles = np.concatenate([subset, rng.choice(subset, size=N_ENT - k)])
        rng.shuffle(roles)
        if _reacting(u, roles):
            return roles.astype(int)
    raise RuntimeError("could not sample a world")


def make_world(u: Universe, condition: str, seed: int) -> World:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    rng = np.random.default_rng([u.seed, seed, CONDITIONS.index(condition), 202])
    rho = 0.9
    switch = None
    if condition in ("train", "iid", "cue_reversed", "role_switch"):
        roles = _distinct_roles_world(rng, u, (2, 3))
        if condition == "cue_reversed":
            rho = 0.1
    elif condition == "recombined":
        roles = _distinct_roles_world(rng, u, (4, 5))
    else:  # novel_role: the only reacting pairs involve the unseen role
        for _ in range(100000):
            n_new = int(rng.choice([1, 2]))
            k = int(rng.choice([1, 2]))
            subset = rng.choice(np.array(KNOWN_ROLES), size=k, replace=False)
            known = np.concatenate([subset, rng.choice(subset, size=N_ENT - n_new - k)])
            roles = np.concatenate([known, np.full(n_new, NOVEL_ROLE)]).astype(int)
            rng.shuffle(roles)
            react = _reacting(u, roles)
            if react and all(NOVEL_ROLE in (roles[PAIRS[p][0]], roles[PAIRS[p][1]]) for p in react):
                break
        else:
            raise RuntimeError("could not sample a novel_role world")
    x = _appearance(rng, roles, rho)
    if condition == "role_switch":
        for _ in range(100000):
            react = _reacting(u, roles)
            p = int(rng.choice(react))
            e = int(rng.choice(PAIRS[p]))
            new = int(rng.choice([r for r in KNOWN_ROLES if r != roles[e]]))
            after = roles.copy()
            after[e] = new
            ok = (not u.law[after[PAIRS[p][0]], after[PAIRS[p][1]]]) and bool(_reacting(u, after))
            if ok:
                switch = (SWITCH_T, e, new)
                break
    return World(condition, roles, x, switch, meta={"rho": rho})


class Episode:
    """One world with common random numbers for outcomes."""

    def __init__(self, u: Universe, world: World, seed: int) -> None:
        self.u, self.world = u, world
        rng = np.random.default_rng([u.seed, seed, CONDITIONS.index(world.condition), 303])
        self.unif = rng.random((HORIZON, N_ACT))

    def probs(self, t: int) -> np.ndarray:
        return true_probs(self.u, self.world.roles_at(t))

    def step(self, t: int, a: int) -> tuple[int, float]:
        o = int(self.unif[t, a] < self.probs(t)[a])
        return o, float(o) if is_pair(a) else 0.0

    def oracle_expected_return(self, t0: int = 0) -> float:
        return float(sum(self.probs(t)[N_ENT:].max() for t in range(t0, HORIZON)))
