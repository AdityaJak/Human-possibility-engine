"""Hidden-law entity worlds, version 3 (revision R1; v1 rejected as too easy, ``results/dev/env_v1_findings.md``;
v2 = attempt 1, whose generator could produce universes in which a condition is impossible).

A universe fixes hidden laws over eight roles — known roles 0-4 (training and test), the unseen role 5 (test only),
and extra roles 6-7 (used only in the *broad* training distribution of one comparator): a 3-bit probe signature per
role (all eight distinct; each bit read correctly with probability 0.9), a 3-bit appearance code per role (all
distinct), and a symmetric reaction law. Every universe is checked to make every condition samplable.

A world is one episode with five entities whose roles are hidden. Agents see appearances, their own actions,
outcomes and rewards only. Outcome noise uses common random numbers (``unif[t, a, :]`` drawn once per world).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, product

import numpy as np

N_ENT = 5
D_APP = 4  # dims 0-2 carry the role code, dim 3 is noise
N_BITS = 3  # probe outcome bits
HORIZON = 20
PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(N_ENT), 2))  # 10 pairs
N_ACT = N_ENT + len(PAIRS)  # 5 probes + 10 pairs
P_HI, P_LO = 0.9, 0.05
BIT_HI, BIT_LO = 0.9, 0.1
N_ROLES = 8
KNOWN_ROLES = (0, 1, 2, 3, 4)
NOVEL_ROLE = 5
EXTRA_ROLES = (6, 7)  # broad training distribution only
SWITCH_T = 10
CONDITIONS = ("train", "iid", "cue_reversed", "recombined", "novel_role", "role_switch", "train_broad")
EVAL_CONDITIONS = ("iid", "cue_reversed", "recombined", "novel_role", "role_switch")
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
    code: np.ndarray  # [N_ROLES, 3] appearance code bits (unseen/extra roles in test: appearance random)
    law: np.ndarray  # [N_ROLES, N_ROLES] bool, symmetric

    def bit_probs(self, r: int) -> np.ndarray:
        return np.where(self.sig[r] == 1, BIT_HI, BIT_LO)

    def pair_prob(self, r: int, s: int) -> float:
        return P_HI if self.law[r, s] else P_LO


def reacting(u: Universe, roles: np.ndarray) -> list[int]:
    return [k for k, (i, j) in enumerate(PAIRS) if u.law[roles[i], roles[j]]]


def _n_react(law: np.ndarray, roles: tuple[int, ...]) -> int:
    return sum(bool(law[roles[i], roles[j]]) for i, j in PAIRS)


@lru_cache(maxsize=1)
def _known_tuples() -> tuple[tuple[int, ...], ...]:
    return tuple(product(KNOWN_ROLES, repeat=N_ENT))


def feasible(law: np.ndarray) -> bool:
    """Every condition must be samplable: train (2-3 distinct known roles, 1-2 reacting pairs, and a valid switch
    for role_switch), recombined (4-5 distinct, 1-2 reacting), novel_role, and the extra-role worlds."""
    train_ok = switch_ok = recomb_ok = False
    for r in _known_tuples():
        k = len(set(r))
        n = _n_react(law, r)
        if not 1 <= n <= 2:
            continue
        if k in (2, 3):
            train_ok = True
            if not switch_ok and _switch_options(law, np.array(r)):
                switch_ok = True
        elif k >= 4:
            recomb_ok = True
        if train_ok and switch_ok and recomb_ok:
            break
    special_ok = all(_special_ok(law, s) for s in (NOVEL_ROLE, *EXTRA_ROLES))
    return train_ok and switch_ok and recomb_ok and special_ok


def _special_ok(law: np.ndarray, special: int) -> bool:
    for n_new in (1, 2):
        for known in product(KNOWN_ROLES, repeat=N_ENT - n_new):
            roles = (*known, *([special] * n_new))
            react = [p for p, (i, j) in enumerate(PAIRS) if law[roles[i], roles[j]]]
            if 1 <= len(react) <= 2 and all(
                special in (roles[PAIRS[p][0]], roles[PAIRS[p][1]]) for p in react
            ):
                return True
    return False


def make_universe(seed: int) -> Universe:
    rng = np.random.default_rng([seed, 111])
    sig = CODES[rng.permutation(8)]
    code = CODES[rng.permutation(8)]
    for _ in range(100000):
        law = np.zeros((N_ROLES, N_ROLES), dtype=bool)
        for a in range(N_ROLES):
            for b in range(a, N_ROLES):
                law[a, b] = law[b, a] = bool(rng.random() < (0.12 if a == b else 0.25))
        k = law[:5, :5]
        if not (all(k[r].any() for r in KNOWN_ROLES) and 3 <= int(np.triu(k).sum()) <= 5):
            continue
        if any(law[s, s] or not (1 <= int(law[s, :5].sum()) <= 2) for s in (NOVEL_ROLE, *EXTRA_ROLES)):
            continue
        law[np.ix_((NOVEL_ROLE, *EXTRA_ROLES), (NOVEL_ROLE, *EXTRA_ROLES))] = False
        if not feasible(law):
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


def _appearance(rng: np.random.Generator, u: Universe, roles: np.ndarray, rho: float) -> np.ndarray:
    """Known roles show their code (each bit flipped with probability 1 - rho); other roles look random."""
    x = rng.normal(size=(N_ENT, D_APP))
    for e, r in enumerate(roles):
        bits = u.code[r].copy() if r in KNOWN_ROLES else rng.integers(0, 2, size=3)
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


def _sample_special(rng: np.random.Generator, u: Universe, special: int) -> np.ndarray:
    for _ in range(100000):
        n_new = int(rng.choice([1, 2]))
        k = int(rng.choice([2, 3]))
        subset = rng.choice(np.array(KNOWN_ROLES), size=k, replace=False)
        known = np.concatenate([subset, rng.choice(subset, size=N_ENT - n_new - k)])[: N_ENT - n_new]
        roles = np.concatenate([known, np.full(n_new, special)]).astype(int)
        rng.shuffle(roles)
        react = reacting(u, roles)
        if 1 <= len(react) <= 2 and all(special in (roles[PAIRS[p][0]], roles[PAIRS[p][1]]) for p in react):
            return roles
    raise RuntimeError("could not sample a special-role world")


def _switch_options(law: np.ndarray, roles: np.ndarray) -> list[tuple[int, int]]:
    react = [p for p, (i, j) in enumerate(PAIRS) if law[roles[i], roles[j]]]
    options = []
    for e in sorted({i for p in react for i in PAIRS[p]}):
        for new in KNOWN_ROLES:
            if new == roles[e]:
                continue
            after = roles.copy()
            after[e] = new
            n_after = sum(bool(law[after[i], after[j]]) for i, j in PAIRS)
            if not any(law[after[i], after[j]] for i, j in (PAIRS[q] for q in react)) and 1 <= n_after <= 2:
                options.append((e, new))
    return options


def make_world(u: Universe, condition: str, seed: int) -> World:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    rng = np.random.default_rng([u.seed, seed, CONDITIONS.index(condition), 212])
    rho = 0.9
    switch = None
    if condition == "train_broad":
        rho = float(rng.uniform(0.3, 1.0))
        kind = rng.random()
        if kind < 0.25:
            roles = _sample_special(rng, u, int(rng.choice(EXTRA_ROLES)))
        else:
            roles = _sample_roles(rng, u, (2, 3))
            if kind < 0.5:
                while not (opts := _switch_options(u.law, roles)):
                    roles = _sample_roles(rng, u, (2, 3))
                e, new = opts[int(rng.integers(len(opts)))]
                switch = (int(rng.integers(5, 16)), e, new)
    elif condition in ("train", "iid", "cue_reversed", "role_switch"):
        roles = _sample_roles(rng, u, (2, 3))
        if condition == "cue_reversed":
            rho = 0.1
        if condition == "role_switch":
            while not (opts := _switch_options(u.law, roles)):
                roles = _sample_roles(rng, u, (2, 3))
            e, new = opts[int(rng.integers(len(opts)))]
            switch = (SWITCH_T, e, new)
    elif condition == "recombined":
        roles = _sample_roles(rng, u, (4, 5))
    else:  # novel_role
        roles = _sample_special(rng, u, NOVEL_ROLE)
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
        rng = np.random.default_rng([u.seed, seed, CONDITIONS.index(world.condition), 313])
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
