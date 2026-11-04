"""Play one world with any agent exposing ``begin(x)``, ``act(t)``, ``observe(t, a, o)``."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .env import HORIZON, N_ACT, SWITCH_T, Episode, Universe, make_world


class Agent(Protocol):
    def begin(self, x: np.ndarray) -> None: ...
    def act(self, t: int) -> int: ...
    def observe(self, t: int, a: int, o: int) -> None: ...


def play(agent: Agent, u: Universe, condition: str, seed: int) -> dict[str, Any]:
    world = make_world(u, condition, seed)
    ep = Episode(u, world, seed)
    agent.begin(world.x.copy())
    ret = 0.0
    exp_ret = 0.0
    first = -1
    exp_post = 0.0
    actions = []
    for t in range(HORIZON):
        a = agent.act(t)
        if not (0 <= a < N_ACT):
            raise ValueError(f"illegal action {a}")
        o, r = ep.step(t, a)
        agent.observe(t, a, o)
        p = ep.expected_reward(t, a)
        ret += r
        exp_ret += p
        if t >= SWITCH_T:
            exp_post += p
        if r > 0 and first < 0:
            first = t
        actions.append(a)
    oracle = ep.oracle_expected_return()
    return {
        "seed": seed,
        "condition": condition,
        "return": ret,
        "expected_return": exp_ret,
        "oracle": oracle,
        "norm": exp_ret / oracle,
        "norm_post_switch": exp_post / ep.oracle_expected_return(SWITCH_T),
        "first_reaction": first if first >= 0 else HORIZON,
        "actions": actions,
    }
