"""Small, reusable measurement helpers for the developmental learner.

These only orchestrate ``hpe.tasks.practice`` / ``hpe.tasks.assess``; they never measure skill
any other way. Formal, preregistered experiments live in ``hpe.evaluation`` (other lane).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hpe.contracts.runtime import FeedbackMode, Policy
from hpe.core.rng import stream
from hpe.tasks.assessment import assess, practice
from hpe.tasks.symbols import SymbolWorld


@dataclass(slots=True)
class LearningCurve:
    family: str
    episodes: list[int]
    accuracy: list[float]
    log_loss: list[float]

    @property
    def gain(self) -> float:
        return self.accuracy[-1] - self.accuracy[0]


def learning_curve(
    policy: Policy,
    world: SymbolWorld,
    family: str,
    *,
    checkpoints: Sequence[int] = (0, 25, 50, 100, 200, 400),
    mode: FeedbackMode = "full",
    age_years: float = 20.0,
    noise: float = 0.05,
    seed: int = 0,
) -> LearningCurve:
    """Practise ``policy`` (in place) and assess on held-out items at each checkpoint."""
    rng = stream(seed, "development", "diagnostics", family, mode)
    done = 0
    accs: list[float] = []
    losses: list[float] = []
    for n in checkpoints:
        if n > done:
            practice(
                policy,
                world,
                family,
                n - done,
                rng,
                agent_id="diag",
                tick=0,
                age_years=age_years,
                mode=mode,
                noise=noise,
            )
            done = n
        r = assess(policy, world, family, age_years)
        accs.append(r.accuracy)
        losses.append(r.log_loss)
    return LearningCurve(family, list(checkpoints), accs, losses)


def episode_cost_us(
    policy: Policy, world: SymbolWorld, family: str = "sort", n: int = 2000, seed: int = 0
) -> float:
    """Median-of-3 wall-clock microseconds for one ``decide`` + ``learn`` pair (policy only).

    Observations and feedback are pre-generated so task-world sampling cost is excluded.
    Reported for compute accounting; never used in simulation logic.
    """
    rng = stream(seed, "development", "timing")
    items = [world.sample_item(family, rng) for _ in range(n)]
    obs = [world.observe(it, "timing", 0, 20.0) for it in items]
    fbs = [world.feedback(it, 0, "bandit", rng) for it in items]
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        for o, fb in zip(obs, fbs, strict=True):
            d = policy.decide(o, rng)
            policy.learn(o, d.action, fb)
        runs.append((time.perf_counter() - t0) / n * 1e6)
    return float(np.median(runs))
