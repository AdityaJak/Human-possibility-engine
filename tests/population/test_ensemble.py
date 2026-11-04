"""adaptive_ens: averages member probabilities, keeps the harness-owned modes, passes the strict check."""

from __future__ import annotations

from typing import Any

import numpy as np

from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import encode, input_dim
from hpe.population.core.interface import ModelSpec, get_model, register_model

from .test_adaptation import CountAdapter
from .test_core import _participant


class SeededCount(CountAdapter):
    name = "toy_seeded_count"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.prior = 1.0

    def fit(self, train, *, seed, threads, budget_s):  # type: ignore[no-untyped-def]
        self.prior = 1.0 + (seed % 5)  # members differ only through their seed
        return {"stopped": "ok"}

    def init_state(self) -> Any:
        return np.full(4, self.prior)


register_model(
    ModelSpec("toy_seeded_count", lambda c, k, d: SeededCount(c), "tests", "reference", {}, adaptive=True)
)


def test_adaptive_ensemble_averages_members_and_is_causal():
    seqs = [encode(_participant(30, i), "toy", 4) for i in range(4)]
    for i, s in enumerate(seqs):
        s.key = f"toy#{i}"
    ens = get_model("adaptive_ens").factory({"base": "toy_seeded_count", "members": 3}, 4, input_dim(4))
    ens.fit(seqs, seed=2, threads=1, budget_s=10)
    priors = [m.prior for m in ens.members]
    assert len(set(priors)) > 1
    lp = ens.predict_logprobs(seqs[0])
    manual = np.mean([np.exp(m.predict_logprobs(seqs[0])) for m in ens.members], axis=0)
    assert np.allclose(np.exp(lp), manual)
    assert check_causal(ens, seqs) == []
