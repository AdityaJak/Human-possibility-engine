"""Developmental learning (lane: research/development).

Importing this package registers ``policy_class = "developmental"`` with
``hpe.contracts.registry``::

    from hpe.contracts.registry import get_factory
    factory = get_factory("developmental", {"use_shared_representation": False})
    policy = factory(agent_init)

``params`` are fields of :class:`DevConfig` (ablation flags + hyperparameters). Unknown keys
raise ``ValueError``.
"""

from __future__ import annotations

from typing import Any

from hpe.contracts import registry
from hpe.contracts.runtime import AgentInit, Policy, PolicyFactory
from hpe.development.config import ABLATION_FLAGS, DevConfig
from hpe.development.learner import DevelopmentalLearner
from hpe.development.schedules import forgetting_multiplier, plasticity, working_memory


def build_factory(params: dict[str, Any] | None = None) -> PolicyFactory:
    cfg = DevConfig.from_params(params)

    def factory(init: AgentInit) -> Policy:
        return DevelopmentalLearner(init, cfg)  # type: ignore[return-value]

    return factory


registry.register("developmental", build_factory)

__all__ = [
    "ABLATION_FLAGS",
    "DevConfig",
    "DevelopmentalLearner",
    "build_factory",
    "forgetting_multiplier",
    "plasticity",
    "working_memory",
]
