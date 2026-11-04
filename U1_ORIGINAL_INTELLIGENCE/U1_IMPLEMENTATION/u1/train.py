"""Training IRA's library on training-condition worlds (the only phase in which the library changes, except in
the sequential accumulation test)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .env import Universe, make_world
from .ira import IRA, IRAConfig, Library
from .play import play
from .recluster import WorldRecord, recluster

TRAIN_SEED0 = 100000


def train_library(
    u: Universe,
    cfg: IRAConfig,
    n_worlds: int = 300,
    refit_every: int = 50,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> tuple[Library, list[WorldRecord], list[float], dict[tuple[int, int], int]]:
    """Play ``n_worlds`` training worlds; consolidate additively after each and re-cluster every ``refit_every``."""
    lib = Library()
    records: list[WorldRecord] = []
    norms: list[float] = []
    assign: dict[tuple[int, int], int] = {}
    rng = np.random.default_rng([seed, u.seed, 404])
    for s in range(n_worlds):
        ws = TRAIN_SEED0 + s
        ag = IRA(lib, cfg, np.random.default_rng([seed, u.seed, ws]))
        m = play(ag, u, "train", ws)
        world = make_world(u, "train", ws)
        records.append(WorldRecord(world.x.copy(), ag.active_observations()))
        ag.consolidate(world.x)
        norms.append(m["norm"])
        if (s + 1) % refit_every == 0 or s + 1 == n_worlds:
            lib, assign = recluster(records, rng, alpha=cfg.alpha)
            if log is not None:
                log(
                    f"train {s + 1}: mean norm (last {refit_every}) {np.mean(norms[-refit_every:]):.3f}; K={lib.k} sizes={[round(v) for v in lib.n]}"
                )
    return lib, records, norms, assign
