"""The ``SequencePredictor`` interface every Phase III model (candidate or comparator) implements.

Contract:
* ``fit(train, seed=, threads=, budget_s=)`` sees only fitting-partition sequences (dev). Any internal
  early-stopping split is drawn from them. Returns a JSON-serialisable fit log.
* ``predict_logprobs(seq, mode=, donor=)`` returns a ``[T, k_max]`` array of log-probabilities for the
  participant's action at every step, **causally**: row t may depend only on ``seq.x[:t+1]`` and
  ``seq.actions[:t]`` (enforced by ``conformance.check_causal``). Unavailable actions get ``-inf``.
  ``mode="adapt"`` (default) enables the model's individual adaptation; ``mode="frozen"`` disables it
  (population weights fixed; recurrent state still runs) — the Gate B1 comparison. With ``donor`` set,
  the individual-level state is learned from the donor's whole sequence and then held fixed while
  predicting ``seq`` — the Gate A2 mismatched-personalisation control. Non-adaptive models ignore both.
* ``param_count()`` counts population parameters; ``describe()`` adds per-person adaptive parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from hpe.population.core.encoding import EncodedSeq

Mode = Literal["adapt", "frozen"]


@runtime_checkable
class SequencePredictor(Protocol):
    name: str
    adaptive: bool

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]: ...

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: Mode = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray: ...

    def param_count(self) -> int: ...

    def describe(self) -> dict[str, Any]: ...


Factory = Callable[[dict[str, Any], int, int], SequencePredictor]  # (config, k_max, input_dim)


@dataclass(slots=True)
class ModelSpec:
    name: str
    factory: Factory
    owner: str  # "core" | "baselines" | "candidates" | ...
    kind: str  # "comparator" | "candidate" | "reference"
    default_grid: dict[str, list[Any]] = field(default_factory=dict)
    adaptive: bool = False
    note: str = ""


MODELS: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    if spec.name in MODELS and MODELS[spec.name].factory is not spec.factory:
        raise ValueError(f"model {spec.name!r} already registered")
    MODELS[spec.name] = spec


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        _import_plugins()
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name]


def _import_plugins() -> None:
    import importlib

    for mod in (
        "hpe.population.core.reference",
        "hpe.population.core.ensemble",
        "hpe.population.baselines",
        "hpe.population.candidates",
        "hpe.population.continual",
        "hpe.population.affect",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:  # lane package not present yet, or torch missing
            if e.name and (e.name == mod or e.name.startswith("torch")):
                continue
            raise
