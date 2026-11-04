"""Harness-owned individual adaptation (critic finding G-1, docs/phase3/INDEPENDENT_SCIENTIFIC_AUDIT.md).

Adaptive models (candidates and adaptive comparators) subclass ``AdaptiveBase`` and implement

* ``update_every`` (m >= 1),
* ``init_state()`` — the person-level state before any of the person's data (the prior),
* ``update_state(state, history)`` — ``history`` is a truncated, re-keyed copy holding only the trials
  the harness allows,
* ``predict_with_states(seq, states)`` — ``[T, k]`` log-probabilities where ``states[i] = (start, state)``
  applies to rows ``[start_i, start_{i+1})``; in-context dynamics inside it must stay causal,
* ``twin_config()`` — the ``pop_gru`` configuration of a separately trained no-adaptation twin (Gate B1).

The harness — not the model — implements the three modes used by the gates:

* ``adapt``: state at row ``j*m`` = update(previous state, the person's own first ``j*m`` trials);
* ``frozen``: ``init_state()`` for every row (diagnostic only; Gate B1 uses the twin);
* ``donor`` (Gate A2): state at row ``j*m`` = update(previous, the donor's first ``min(j*m, T_donor)``
  trials) — a length-matched online schedule, so own and donor histories have equal length at every row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hpe.population.core.encoding import EncodedSeq, truncate


class AdaptiveBase:
    name: str = "adaptive"
    adaptive: bool = True
    update_every: int = 10

    # ---- to implement --------------------------------------------------------------------------------
    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        raise NotImplementedError

    def init_state(self) -> Any:
        raise NotImplementedError

    def update_state(self, state: Any, history: EncodedSeq) -> Any:
        raise NotImplementedError

    def predict_with_states(self, seq: EncodedSeq, states: list[tuple[int, Any]]) -> np.ndarray:
        raise NotImplementedError

    def param_count(self) -> int:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"population_params": self.param_count()}

    def twin_config(self) -> dict[str, Any] | None:
        return None

    # ---- harness (do not override) --------------------------------------------------------------------
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        return harness_predict(self, seq, mode=mode, donor=donor)


def schedule(
    model: AdaptiveBase, seq: EncodedSeq, *, mode: str, donor: EncodedSeq | None
) -> list[tuple[int, Any]]:
    m = max(1, int(model.update_every))
    state = model.init_state()
    states: list[tuple[int, Any]] = [(0, state)]
    if mode == "frozen":
        return states
    source = donor if donor is not None else seq
    for start in range(m, seq.T, m):
        n = min(start, source.T)
        state = model.update_state(state, truncate(source, n, key="history"))
        states.append((start, state))
    return states


def harness_predict(
    model: AdaptiveBase, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
) -> np.ndarray:
    if mode not in ("adapt", "frozen"):
        raise ValueError(f"unknown mode {mode!r}")
    states = schedule(model, seq, mode=mode, donor=donor)
    out = model.predict_with_states(seq, states)
    if out.shape != (seq.T, seq.k_max):
        raise ValueError(
            f"{model.name}: predict_with_states returned {out.shape}, expected {(seq.T, seq.k_max)}"
        )
    return out


def overrides_harness(model: object) -> bool:
    """True if an adaptive model replaced the harness-owned ``predict_logprobs`` (not allowed)."""
    return (
        isinstance(model, AdaptiveBase) and type(model).predict_logprobs is not AdaptiveBase.predict_logprobs
    )
