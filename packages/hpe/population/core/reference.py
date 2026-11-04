"""Reference population sequence models owned by the core: ``pop_gru`` and ``pop_lstm``.

Population-only: trained on the fitting participants; individual adaptation happens only in context
(through the recurrent state). ``mode`` and ``donor`` are ignored (``adaptive = False``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import ModelSpec, register_model


class PopRNNPredictor:
    adaptive = False

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int, cell: str) -> None:
        self.name = f"pop_{cell}"
        self.config = dict(config)
        self.k_max, self.d_in, self.cell = k_max, d_in, cell
        self.net = nets.PopRNN(
            d_in, int(config.get("hidden", 32)), k_max, cell=cell, layers=int(config.get("layers", 1))
        )

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        nets.setup_torch(seed, threads, self.name, repr(sorted(self.config.items())))
        self.net = nets.PopRNN(
            self.d_in,
            int(self.config.get("hidden", 32)),
            self.k_max,
            cell=self.cell,
            layers=int(self.config.get("layers", 1)),
            dropout=float(self.config.get("dropout", 0.0)),
        )

        def loss(m: torch.nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            logp, _ = m(b.x, b.avail)
            return nets.nll_sum(logp, b)

        return nets.train(
            self.net,
            loss,
            train,
            seed=seed,
            key=self.name,
            lr=float(self.config.get("lr", 0.01)),
            max_epochs=int(self.config.get("max_epochs", 150)),
            patience=int(self.config.get("patience", 10)),
            batch_size=int(self.config.get("batch_size", 8)),
            budget_s=budget_s,
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    @torch.no_grad()
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        b = nets.Batch([seq])
        logp, _ = self.net(b.x, b.avail)
        return nets.logp_numpy(logp[0], seq)

    def param_count(self) -> int:
        return nets.n_params(self.net)

    def describe(self) -> dict[str, Any]:
        return {"population_params": self.param_count(), "per_person_params": 0, "config": self.config}


def _gru(config: dict[str, Any], k: int, d: int) -> PopRNNPredictor:
    return PopRNNPredictor(config, k, d, "gru")


def _lstm(config: dict[str, Any], k: int, d: int) -> PopRNNPredictor:
    return PopRNNPredictor(config, k, d, "lstm")


register_model(
    ModelSpec("pop_gru", _gru, "core", "comparator", {"hidden": [16, 32, 64], "lr": [0.003, 0.01]})
)
register_model(
    ModelSpec("pop_lstm", _lstm, "core", "comparator", {"hidden": [16, 32, 64], "lr": [0.003, 0.01]})
)


class PopGRUEnsemble:
    """``pop_gru_ens``: probability average of ``members`` population GRUs trained from different seeds
    (critic finding F-5: a seed ensemble is a standard, strong population baseline)."""

    adaptive = False

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.name = "pop_gru_ens"
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.members: list[PopRNNPredictor] = []

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        n = int(self.config.get("members", 5))
        logs = []
        self.members = []
        for i in range(n):
            m = PopRNNPredictor(
                {k: v for k, v in self.config.items() if k != "members"}, self.k_max, self.d_in, "gru"
            )
            logs.append(
                m.fit(train, seed=(seed * 7919 + i) % (2**31 - 1), threads=threads, budget_s=budget_s / n)
            )
            self.members.append(m)
        return {
            "members": logs,
            "stopped": "budget" if any(lg.get("stopped") == "budget" for lg in logs) else "ok",
        }

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        probs = np.mean([np.exp(m.predict_logprobs(seq)) for m in self.members], axis=0)
        with np.errstate(divide="ignore"):
            return np.where(seq.avail, np.log(probs), -np.inf)

    def param_count(self) -> int:
        return int(sum(m.param_count() for m in self.members))

    def describe(self) -> dict[str, Any]:
        return {"population_params": self.param_count(), "per_person_params": 0, "config": self.config}


def _ens(config: dict[str, Any], k: int, d: int) -> PopGRUEnsemble:
    return PopGRUEnsemble(config, k, d)


register_model(
    ModelSpec(
        "pop_gru_ens", _ens, "core", "comparator", {"hidden": [32, 64], "lr": [0.003, 0.01], "members": [5]}
    )
)
