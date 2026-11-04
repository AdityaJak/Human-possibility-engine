"""Policy registry: the single seam where lanes plug their policy classes in.

The world never imports a learner directly; experiments and the API resolve a
``PolicyClass`` to a factory here. Each lane registers its class from its own package
``__init__`` via ``register`` (lazy import below keeps the dependency graph acyclic).

    developmental   -> hpe.development   (research/development lane)
    static_persona  -> hpe.evaluation.baselines (science/evaluation lane)
    memory_only     -> hpe.evaluation.baselines
    plain_rl        -> hpe.evaluation.baselines
    random          -> hpe.contracts.runtime.RandomPolicy (reference)
"""

from __future__ import annotations

import importlib
from typing import Any

from hpe.contracts.protocols import PolicyFactoryBuilder
from hpe.contracts.runtime import PolicyFactory, random_policy_factory

_REGISTRY: dict[str, PolicyFactoryBuilder] = {}

_PROVIDERS: dict[str, str] = {
    "developmental": "hpe.development",
    "hybrid_exemplar": "hpe.development",
    "static_persona": "hpe.evaluation.baselines",
    "memory_only": "hpe.evaluation.baselines",
    "plain_rl": "hpe.evaluation.baselines",
}


def register(policy_class: str, builder: PolicyFactoryBuilder) -> None:
    _REGISTRY[policy_class] = builder


def _random_builder(params: dict[str, Any]) -> PolicyFactory:
    return random_policy_factory


register("random", _random_builder)


def get_factory(policy_class: str, params: dict[str, Any] | None = None) -> PolicyFactory:
    key = str(getattr(policy_class, "value", policy_class))
    if key not in _REGISTRY and key in _PROVIDERS:
        importlib.import_module(_PROVIDERS[key])
    if key not in _REGISTRY:
        raise KeyError(f"no policy registered for {key!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](dict(params or {}))


def registered() -> list[str]:
    for mod in set(_PROVIDERS.values()):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    return sorted(_REGISTRY)
