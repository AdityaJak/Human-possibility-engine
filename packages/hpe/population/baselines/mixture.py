"""``gru_hb_mix`` — fixed-weight mixture of the population GRU and the hierarchical Bayesian cognitive model
(critic finding F-5: "a pop_gru + hierarchical-cognitive-model mixture with a weight fitted on tune").

``p_t = (1 − w) p_gru,t + w p_hb,t``. The GRU is the backbone (trained exactly like ``pop_gru``; configuration and
seed from the YAML, normally the tune-selected ``pop_gru``); ``hb_cog`` is fitted on dev as in ``hb_cog.py``; the
weight ``w`` is a grid hyper-parameter, i.e. chosen on tune by the runner. Person-level state (harness-owned
modes): ``hb_cog``'s posterior log-weights over its particles; the GRU part adapts only in context.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hpe.population.baselines.backbone import backbone_config, hidden_states, train_backbone
from hpe.population.baselines.hb_cog import HBCog
from hpe.population.core import nets
from hpe.population.core.adaptation import AdaptiveBase
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import ModelSpec, register_model


class GruHbMix(AdaptiveBase):
    name = "gru_hb_mix"
    adaptive = True

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.hb = HBCog(config, k_max, d_in)
        self.update_every = self.hb.update_every
        self.net: nets.PopRNN | None = None
        self.fit_info: dict[str, Any] = {}

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        bb = backbone_config(self.config)
        self.net, log = train_backbone(
            train,
            bb,
            int(self.config.get("backbone_seed", seed)),
            d_in=self.d_in,
            k_max=self.k_max,
            threads=threads,
            budget_s=budget_s,
        )
        self.fit_info = {**log, "hb_cog": self.hb.fit(train, seed=seed, threads=threads, budget_s=budget_s)}
        return self.fit_info

    def init_state(self) -> Any:
        return self.hb.init_state()

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        return self.hb.update_state(state, history)

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("gru_hb_mix: predict before fit")
        w = float(self.config.get("w", 0.2))
        _, lp_gru = hidden_states(self.net, seq)
        lp_hb = self.hb.predict_with_states(seq, states)
        p = (1.0 - w) * np.exp(lp_gru) + w * np.exp(lp_hb)
        with np.errstate(divide="ignore"):
            return np.where(seq.avail, np.log(p), -np.inf)

    def twin_config(self) -> dict[str, Any] | None:
        return backbone_config(self.config)

    def param_count(self) -> int:
        if self.net is None:
            raise RuntimeError("gru_hb_mix: param_count before fit")
        return nets.n_params(self.net) + self.hb.param_count() + 1

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": self.hb.param_count(),
            "update_every": self.update_every,
            "config": self.config,
            "backbone_cpu_s": self.fit_info.get("backbone_cpu_s"),
        }


def _build(config: dict[str, Any], k: int, d: int) -> GruHbMix:
    return GruHbMix(config, k, d)


register_model(
    ModelSpec(
        "gru_hb_mix",
        _build,
        "baselines",
        "comparator",
        {"w": [0.05, 0.1, 0.2, 0.3, 0.5], "c": [0.1, 1.0]},
        adaptive=True,
        note="(1-w) pop_gru + w hb_cog, w chosen on tune",
    )
)
