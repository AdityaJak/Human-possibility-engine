"""Harness-owned adaptation modes: adapt uses only the own prefix, frozen uses the prior, donor is length-matched."""

from __future__ import annotations

from typing import Any

import numpy as np

from hpe.population.core.adaptation import AdaptiveBase, overrides_harness, schedule
from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import EncodedSeq, encode

from .test_core import _participant


class CountAdapter(AdaptiveBase):
    """Per-person choice-frequency state, updated every 3 trials; prediction = state frequencies."""

    name = "count_adapter"
    update_every = 3

    def __init__(self) -> None:
        self.seen_lengths: list[int] = []

    def fit(self, train, *, seed, threads, budget_s):  # type: ignore[no-untyped-def]
        return {}

    def init_state(self) -> Any:
        return np.ones(4)

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        self.seen_lengths.append(history.T)
        c = np.ones(4)
        np.add.at(c, history.actions[history.scored], 1.0)
        return c

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        out = np.full((seq.T, seq.k_max), -np.inf)
        starts = [s for s, _ in states] + [seq.T]
        for (s, st), e in zip(states, starts[1:], strict=False):
            p = np.where(seq.avail[s:e], st, 0.0)
            p = p / p.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore"):
                out[s:e] = np.where(seq.avail[s:e], np.log(p), -np.inf)
        return out

    def param_count(self) -> int:
        return 0


def test_schedules_are_causal_and_length_matched():
    s = encode(_participant(20, 1), "toy", 4)
    d = encode(_participant(8, 2), "toy", 4)
    m = CountAdapter()
    ad = schedule(m, s, mode="adapt", donor=None)
    assert [st for st, _ in ad] == [0, 3, 6, 9, 12, 15, 18]
    assert m.seen_lengths == [3, 6, 9, 12, 15, 18]  # own prefix strictly before each start row
    assert len(schedule(m, s, mode="frozen", donor=None)) == 1
    m.seen_lengths.clear()
    schedule(m, s, mode="adapt", donor=d)
    assert m.seen_lengths == [3, 6, 8, 8, 8, 8]  # donor prefix of equal length, capped at the donor's length
    assert not overrides_harness(m)
    assert check_causal(m, s) == []  # type: ignore[arg-type]
