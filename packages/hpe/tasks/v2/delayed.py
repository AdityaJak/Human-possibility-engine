"""Delayed-feedback practice loop.

Feedback for episode ``t`` is delivered to the policy at episode ``t + k`` together with the original
observation, so credit assignment is correct but learning is delayed and interleaved with ``k``
intervening decisions. Episodes left in the buffer at the end are flushed. This tests learning under
delayed, interleaved feedback within the frozen contracts; it is not a temporal-credit-assignment task.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from hpe.contracts.runtime import FeedbackMode, FeedbackSource, Policy
from hpe.tasks.assessment import PracticeStats


def practice_delayed(
    policy: Policy,
    world,  # V2World or SymbolWorld (duck-typed)
    family: str,
    n: int,
    rng: np.random.Generator,
    *,
    k: int,
    agent_id: str,
    tick: int,
    age_years: float,
    mode: FeedbackMode = "bandit",
    source: FeedbackSource = "environment",
    noise: float = 0.0,
) -> PracticeStats:
    stats = PracticeStats(family=family)
    buf: deque = deque()
    for i in range(n):
        item = world.sample_item(family, rng, split="train")
        obs = world.observe(item, agent_id, tick + i, age_years, True)
        dec = policy.decide(obs, rng, explore=True)
        fb = world.feedback(item, dec.action, mode, rng, noise=noise, source=source)
        stats.n += 1
        stats.correct += int(dec.action == item.correct_action)
        if fb.mode != "none":
            stats.rewards += fb.reward
        buf.append((obs, dec.action, fb))
        if len(buf) > k:
            o, a, f = buf.popleft()
            upd = policy.learn(o, a, f)
            if upd is not None:
                stats.loss_sum += upd.loss
                stats.delta_norm_sum += upd.param_delta_norm
                stats.n_updates += 1
    while buf:
        o, a, f = buf.popleft()
        upd = policy.learn(o, a, f)
        if upd is not None:
            stats.loss_sum += upd.loss
            stats.delta_norm_sum += upd.param_delta_norm
            stats.n_updates += 1
    return stats
