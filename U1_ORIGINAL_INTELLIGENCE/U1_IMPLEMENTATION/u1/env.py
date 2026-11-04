"""Hidden-law entity worlds, version 2 (MATHEMATICAL_FORMULATION.md §1; v1 rejected as too easy, see
``results/dev/env_v1_findings.md``).

A universe fixes hidden laws: for each role a 3-bit probe signature (each bit observed with probability 0.9 of being
correct), a 3-bit appearance code, and a symmetric role-pair reaction law. A world is one episode with five entities
whose roles are hidden. Agents see appearances, their own actions, outcomes and rewards only.

Outcome noise uses common random numbers: ``unif[t, a, :]`` is drawn once per world, so every agent that takes
action ``a`` at step ``t`` receives the same outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

N_ENT = 5
D_APP = 4  # dims 0-2 carry the role code, dim 3 is noise
N_BITS = 3  # probe outcome bits
HORIZON = 20
PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(N_ENT), 2))  # 10 pairs
N_ACT = N_ENT + len(PAIRS)  # 5 probes + 10 pairs
P_HI, P_LO = 0.9, 0.05
BIT_HI, BIT_LO = 0.9, 0.1
N_ROLES = 6
KNOWN_ROLES = (0, 1, 2, 3, 4)
NOVEL_ROLE = 5
SWITCH_T = 10
CONDITIONS = ("train", "iid", "cue_reversed", "recombined", "novel_role", "role_switch")
SHIFT_CONDITIONS = ("cue_reversed", "recombined", "novel_role", "role_switch")
CODES = np.array([[(c >> b) & 1 for b in range(3)] for c in range(8)])  # the 8 three-bit patterns


def action_entities(a: int) -> tuple[int, ...]:
    return (a,) if a < N_ENT else PAIRS[a - N_ENT]


def is_pair(a: int) -> bool:
    return a >= N_ENT


@dataclass(frozen=True)
class Universe:
    seed: int
    sig: np.ndarray  # [N_ROLES, N_BITS] probe signature bits
    code: np.ndarray  # [N_ROLES, 3] appearance code bits (novel role: unused, appearance is random)
    law: np.ndarray  # [N_ROLES, N_ROLES] bool, symmetric

    def bit_probs(self, r: int) -> np.ndarray:
        return np.where(self.sig[r] == 1, BIT_HI, BIT_LO)

    def pair_prob(self, r: int, s: int) -> float:
        return P_HI if self.law[r, s] else P_LO


def make_universe(seed: int) -> Universe:
    rng = np.random.default_rng([seed, 101])
    sig = CODES[rng.permutation(8)[:N_ROLES]]
    code = CODES[rng.permutation(8)[:N_ROLES]]
    for _ in range(100000):
        law = np.zeros((N_ROLES, N_ROLES), dtype=bool)
        for a in range(N_ROLES):
            for b in range(a, N_ROLES):
                law[a, b] = law[b, a] = bool(rng.random() < (0.12 if a == b else 0.25))
        k = law[:5, :5]
        n_known = int(np.triu(k).sum())
        if not (all(k[r].any() for r in KNOWN_ROLES) and 3 <= n_known <= 5):
            continue
        if law[NOVEL_ROLE, NOVEL_ROLE] or not (1 <= int(law[NOVEL_ROLE, :5].sum()) <= 2):
            continue
        return Universe(seed, sig, code, law)
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


def reacting(u: Universe, roles: np.ndarray) -> list[int]:
    return [k for k, (i, j) in enumerate(PAIRS) if u.law[roles[i], roles[j]]]


def _appearance(rng: np.random.Generator, u: Universe, roles: np.ndarray, rho: float) -> np.ndarray:
    x = rng.normal(size=(N_ENT, D_APP))
    for e, r in enumerate(roles):
        bits = rng.integers(0, 2, size=3) if r == NOVEL_ROLE else u.code[r].copy()
        flip = rng.random(3) >= rho
        bits = np.where(flip, 1 - bits, bits)
        x[e, :3] = np.where(bits == 1, 2.0, -2.0) + 0.3 * rng.normal(size=3)
    return x


def _sample_roles(rng: np.random.Generator, u: Universe, n_distinct: tuple[int, ...]) -> np.ndarray:
    for _ in range(100000):
        k = int(rng.choice(n_distinct))
        subset = rng.choice(np.array(KNOWN_ROLES), size=k, replace=False)
        roles = np.concatenate([subset, rng.choice(subset, size=N_ENT - k)])
        rng.shuffle(roles)
        if 1 <= len(reacting(u, roles)) <= 2:
            return roles.astype(int)
    raise RuntimeError("could not sample a world")


def _sample_switch(rng: np.random.Generator, u: Universe, roles: np.ndarray) -> tuple[int, int, int] | None:
    """One entity of a reacting pair changes role at SWITCH_T: every pair reacting before stops reacting, and
    1-2 pairs react after. None if no such change exists for these roles."""
    react = reacting(u, roles)
    options = []
    for e in sorted({i for p in react for i in PAIRS[p]}):
        for new in KNOWN_ROLES:
            if new == roles[e]:
                continue
            after = roles.copy()
            after[e] = new
            if (
                not any(u.law[after[i], after[j]] for i, j in (PAIRS[q] for q in react))
                and 1 <= len(reacting(u, after)) <= 2
            ):
                options.append((SWITCH_T, e, new))
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def make_world(u: Universe, condition: str, seed: int) -> World:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    rng = np.random.default_rng([u.seed, seed, CONDITIONS.index(condition), 202])
    rho = 0.9
    switch = None
    if condition in ("train", "iid", "cue_reversed", "role_switch"):
        roles = _sample_roles(rng, u, (2, 3))
        if condition == "cue_reversed":
            rho = 0.1
    elif condition == "recombined":
        roles = _sample_roles(rng, u, (4, 5))
    else:  # novel_role: 1-2 entities of the unseen role; every reacting pair involves it
        for _ in range(100000):
            n_new = int(rng.choice([1, 2]))
            k = int(rng.choice([2, 3]))
            subset = rng.choice(np.array(KNOWN_ROLES), size=k, replace=False)
            known = np.concatenate([subset, rng.choice(subset, size=N_ENT - n_new - k)])[: N_ENT - n_new]
            roles = np.concatenate([known, np.full(n_new, NOVEL_ROLE)]).astype(int)
            rng.shuffle(roles)
            react = reacting(u, roles)
            if 1 <= len(react) <= 2 and all(
                NOVEL_ROLE in (roles[PAIRS[p][0]], roles[PAIRS[p][1]]) for p in react
            ):
                break
        else:
            raise RuntimeError("could not sample a novel_role world")
    if condition == "role_switch":
        for _ in range(100000):
            switch = _sample_switch(rng, u, roles)
            if switch is not None:
                break
            roles = _sample_roles(rng, u, (2, 3))
        if switch is None:
            raise RuntimeError("could not sample a switch")
    x = _appearance(rng, u, roles, rho)
    return World(condition, roles, x, switch, meta={"rho": rho})


def encode_bits(bits: np.ndarray) -> int:
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def decode_bits(o: int) -> np.ndarray:
    return np.array([(o >> i) & 1 for i in range(N_BITS)])


class Episode:
    """One world with common random numbers for outcomes.

    ``step`` returns ``(o, r)``: for a pair, ``o`` ∈ {0, 1} (reaction) and ``r = o``; for a probe, ``o`` ∈ 0..7 is the
    3-bit signature reading (bit ``b`` = ``(o >> b) & 1``) and ``r = 0``.
    """

    def __init__(self, u: Universe, world: World, seed: int) -> None:
        self.u, self.world = u, world
        rng = np.random.default_rng([u.seed, seed, CONDITIONS.index(world.condition), 303])
        self.unif = rng.random((HORIZON, N_ACT, N_BITS))

    def pair_probs(self, t: int) -> np.ndarray:
        roles = self.world.roles_at(t)
        return np.array([self.u.pair_prob(int(roles[i]), int(roles[j])) for i, j in PAIRS])

    def expected_reward(self, t: int, a: int) -> float:
        return float(self.pair_probs(t)[a - N_ENT]) if is_pair(a) else 0.0

    def step(self, t: int, a: int) -> tuple[int, float]:
        roles = self.world.roles_at(t)
        if is_pair(a):
            o = int(self.unif[t, a, 0] < self.pair_probs(t)[a - N_ENT])
            return o, float(o)
        bits = (self.unif[t, a] < self.u.bit_probs(int(roles[a]))).astype(int)
        return encode_bits(bits), 0.0

    def oracle_expected_return(self, t0: int = 0) -> float:
        return float(sum(self.pair_probs(t).max() for t in range(t0, HORIZON)))
