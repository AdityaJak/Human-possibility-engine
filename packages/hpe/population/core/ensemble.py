"""``adaptive_ens``: a generic probability-averaged seed ensemble of any registered adaptive model.

Purpose (X3-E01, matched-compute control): the strongest restless4 comparator on tune is a 5-seed ensemble of
population GRUs (``pop_gru_ens``). To ask whether explicit individual adaptation adds value *on top of* the
variance reduction an ensemble buys, a candidate or an adaptive comparator must be compared at equal ensemble
size. The wrapper does not alter the member model: member i is built from ``base_config`` (with any
``backbone_seed`` shifted by 7919·i so members differ) and fitted with a derived seed and 1/n of the budget.
Its person-level state is the tuple of the members' states, so the harness owns adapt / frozen / donor modes
exactly as for a single model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hpe.population.core.adaptation import AdaptiveBase
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import ModelSpec, get_model, register_model


class AdaptiveEnsemble(AdaptiveBase):
    name = "adaptive_ens"
    adaptive = True

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.base = str(config["base"])
        self.n = int(config.get("members", 5))
        self.base_config = dict(config.get("base_config", {}))
        self.k_max, self.d_in = k_max, d_in
        self.members: list[Any] = []
        self.update_every = 10

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        spec = get_model(self.base)
        self.members, logs = [], []
        for i in range(self.n):
            cfg = dict(self.base_config)
            if "backbone_seed" in cfg:
                cfg["backbone_seed"] = (int(cfg["backbone_seed"]) + 7919 * i) % (2**31 - 1)
            m = spec.factory(cfg, self.k_max, self.d_in)
            if not isinstance(m, AdaptiveBase):
                raise TypeError(f"adaptive_ens needs an AdaptiveBase member, got {type(m).__name__}")
            logs.append(
                m.fit(train, seed=(seed * 31 + i) % (2**31 - 1), threads=threads, budget_s=budget_s / self.n)
            )
            self.members.append(m)
        steps = {int(m.update_every) for m in self.members}
        if len(steps) != 1:
            raise ValueError(f"members disagree on update_every: {sorted(steps)}")
        self.update_every = steps.pop()
        stopped = "budget" if any((lg or {}).get("stopped") == "budget" for lg in logs) else "ok"
        return {"members": logs, "stopped": stopped}

    def init_state(self) -> Any:
        return tuple(m.init_state() for m in self.members)

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        return tuple(m.update_state(s, history) for m, s in zip(self.members, state, strict=True))

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        probs = np.zeros((seq.T, seq.k_max))
        for i, m in enumerate(self.members):
            lp = m.predict_with_states(seq, [(start, st[i]) for start, st in states])
            probs += np.exp(lp)
        probs /= len(self.members)
        with np.errstate(divide="ignore"):
            return np.where(seq.avail, np.log(probs), -np.inf)

    def twin_config(self) -> dict[str, Any] | None:
        return self.members[0].twin_config() if self.members else None

    def param_count(self) -> int:
        return int(sum(m.param_count() for m in self.members))

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "base": self.base,
            "members": self.n,
            "base_config": self.base_config,
        }


def _build(config: dict[str, Any], k: int, d: int) -> AdaptiveEnsemble:
    return AdaptiveEnsemble(config, k, d)


register_model(
    ModelSpec(
        "adaptive_ens",
        _build,
        "core",
        "reference",
        {},
        adaptive=True,
        note="seed ensemble of an adaptive model",
    )
)
