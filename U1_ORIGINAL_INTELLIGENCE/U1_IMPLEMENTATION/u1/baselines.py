"""Non-neural comparators. All act on the same information as IRA (pair outcomes; they ignore pokes and appearance)."""

from __future__ import annotations

import numpy as np

from .env import N_ACT, N_ENT


class RandomPairs:
    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def begin(self, x: np.ndarray) -> None:
        pass

    def act(self, t: int) -> int:
        return int(N_ENT + self.rng.integers(N_ACT - N_ENT))

    def observe(self, t: int, a: int, o: int) -> None:
        pass


class UCBPairs:
    """UCB1 over the ten pair actions (each pair is its own arm; nothing transfers between pairs or worlds)."""

    name = "ucb"

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def begin(self, x: np.ndarray) -> None:
        self.n = np.zeros(N_ACT - N_ENT)
        self.s = np.zeros(N_ACT - N_ENT)

    def act(self, t: int) -> int:
        if (self.n == 0).any():
            return int(N_ENT + np.argmax(self.n == 0))
        bonus = self.c * np.sqrt(np.log(t + 1) / self.n)
        return int(N_ENT + np.argmax(self.s / self.n + bonus))

    def observe(self, t: int, a: int, o: int) -> None:
        self.n[a - N_ENT] += 1
        self.s[a - N_ENT] += o


class EntityBayes:
    """No concepts: an independent Beta-Bernoulli law per pair, prior mean = pooled training reaction rate,
    Thompson sampling. Learns across worlds only the pooled rate."""

    name = "entity_bayes"

    def __init__(self, rng: np.random.Generator, rbar: float = 0.5, strength: float = 1.0) -> None:
        self.rng, self.rbar, self.strength = rng, rbar, strength

    def begin(self, x: np.ndarray) -> None:
        self.a = np.full(N_ACT - N_ENT, self.rbar * self.strength)
        self.b = np.full(N_ACT - N_ENT, (1 - self.rbar) * self.strength)

    def act(self, t: int) -> int:
        return int(N_ENT + np.argmax(self.rng.beta(self.a, self.b)))

    def observe(self, t: int, a: int, o: int) -> None:
        self.a[a - N_ENT] += o
        self.b[a - N_ENT] += 1 - o
