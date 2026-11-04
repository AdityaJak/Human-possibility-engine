"""Frozen shared contracts (v0.1). Owned by the orchestrator; changes require an ADR.

- ``runtime``: hot-path dataclasses + the ``Policy`` protocol used by world and learners.
- ``schemas``: persisted/API Pydantic records (exported to JSON Schema and TypeScript).
- ``api``: HTTP request/response models.
- ``protocols``: World runner and event-sink protocols.
"""

from hpe.contracts.runtime import (
    CONTRACT_VERSION,
    AgentInit,
    BasePolicy,
    ComputeCounter,
    Decision,
    Demonstration,
    Feedback,
    LearningUpdateRecord,
    Observation,
    Policy,
    PolicyFactory,
    RandomPolicy,
    masked_softmax,
    random_policy_factory,
)

__all__ = [
    "CONTRACT_VERSION",
    "AgentInit",
    "BasePolicy",
    "ComputeCounter",
    "Decision",
    "Demonstration",
    "Feedback",
    "LearningUpdateRecord",
    "Observation",
    "Policy",
    "PolicyFactory",
    "RandomPolicy",
    "masked_softmax",
    "random_policy_factory",
]
