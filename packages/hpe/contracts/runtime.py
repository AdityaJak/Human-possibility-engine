"""Hot-path runtime contracts between the World and Policies (contract v0.1).

These are lightweight ``dataclasses`` holding numpy arrays because they are created
hundreds of thousands of times per run. Persisted/API records live in ``schemas.py``.

Data flow (ARCHITECTURE.md §3)::

    World --Observation--> Policy.decide() --Decision--> World
    World --Feedback-----> Policy.learn()  --LearningUpdate--> event log
    World --Demonstration (peer/tutor)--> Policy.learn_from_demonstration()

Invariants every Policy implementation MUST satisfy (enforced by
``tests/contracts/test_policy_conformance.py``):

1. ``decide(obs, rng, explore=False)`` and ``action_probabilities*`` never mutate learnable
   state (only compute counters). Assessment relies on this to measure skill without
   training on held-out items.
2. ``action_probabilities(obs)`` returns a length-``n_actions`` probability vector that sums
   to 1, assigns 0 to masked actions, and equals the distribution ``decide`` samples from
   when ``explore=True``.
3. ``state_dict()`` is JSON-serialisable and ``load_state_dict(state_dict())`` round-trips
   exactly: decisions after restore are identical given the same rng.
4. The policy only sees what the Observation contains. It must not read world internals,
   task rules, held-out sets, or other agents' state.
5. No network or language-model calls inside ``decide``/``learn`` (Tier 1/2 must run
   without API keys).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

CONTRACT_VERSION = "0.1.0"

DecisionKind = Literal["task", "activity", "practice", "social"]
"""
- ``task``: solve one item of a cognitive task family (``obs.family`` = family id).
- ``activity``: choose how to spend the current tick (``obs.meta['actions']`` names them).
- ``practice``: choose which task family to practise during self-study.
- ``social``: cooperate (1) or not (0) with a partner in a repeated trust game.
"""

FeedbackMode = Literal["none", "bandit", "full"]
"""
- ``none``: no outcome revealed (e.g. unsupervised exposure, or the no-feedback condition).
- ``bandit``: only the reward of the chosen action is revealed.
- ``full``: instructional feedback; the correct action (and every action's reward) is revealed.
"""

ACTIVITY_ACTIONS: tuple[str, ...] = ("rest", "study", "work", "socialize")
ACTIVITY_CONTEXT_FEATURES: tuple[str, ...] = (
    "energy",
    "stress",
    "social_connection",
    "resources_norm",
    "age_norm",
    "employed",
    "enrolled",
    "bias",
)
SOCIAL_CONTEXT_FEATURES: tuple[str, ...] = (
    "trust_in_partner",
    "partner_is_family",
    "partner_is_friend",
    "bias",
)

FeedbackSource = Literal["environment", "school", "tutor", "peer", "self", "job", "human_data"]


@dataclass(slots=True)
class Observation:
    """What an agent perceives at a decision point.

    ``context`` is a 2-D float32 array of shape ``(n_slots, n_features)``. For task decisions
    each slot is one perceived symbol (already age-gated by the world: masked features are 0).
    For activity decisions there is one slot holding the agent's need/affect state vector.
    """

    agent_id: str
    tick: int
    age_years: float
    kind: DecisionKind
    family: str
    context: np.ndarray
    n_actions: int
    action_mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def mask(self) -> np.ndarray:
        if self.action_mask is None:
            return np.ones(self.n_actions, dtype=bool)
        return self.action_mask


@dataclass(slots=True)
class Feedback:
    """Outcome information the world reveals after a decision."""

    mode: FeedbackMode
    reward: float = float("nan")
    correct_action: int | None = None
    rewards_all: np.ndarray | None = None
    source: FeedbackSource = "environment"


@dataclass(slots=True)
class Decision:
    action: int
    probs: np.ndarray
    policy_class: str
    policy_version: str
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Demonstration:
    """A socially observed episode (peer, parent, tutor, or misinformation source)."""

    observation: Observation
    action: int
    feedback: Feedback
    demonstrator_id: str
    trust: float


@dataclass(slots=True)
class LearningUpdateRecord:
    """Hot-path record of one learning step; converted to ``schemas.LearningUpdate`` for logs."""

    agent_id: str
    tick: int
    policy_class: str
    family: str
    source: FeedbackSource
    feedback_mode: FeedbackMode
    n_samples: int
    loss: float
    param_delta_norm: float
    n_updates_total: int


@dataclass(slots=True)
class ComputeCounter:
    """Counts work so baselines can be compared under matched compute budgets."""

    decisions: int = 0
    learn_calls: int = 0
    param_updates: int = 0
    flops_estimate: float = 0.0
    memory_items: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "decisions": self.decisions,
            "learn_calls": self.learn_calls,
            "param_updates": self.param_updates,
            "flops_estimate": self.flops_estimate,
            "memory_items": self.memory_items,
        }


@dataclass(slots=True)
class AgentInit:
    """Initial conditions handed to a policy factory when an agent is created.

    ``traits`` are abstract model parameters in [0, 1] (curiosity, diligence, sociability,
    risk_tolerance, patience). They are *not* personality measurements of any real person.
    ``inherited`` lists which init values came from parents (for Experiment 5).
    """

    agent_id: str
    birth_tick: int
    seed: int
    traits: dict[str, float]
    base_params: dict[str, float]
    families: dict[str, dict[str, int]]
    activity_actions: list[str]
    inherited: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    policy_class: ClassVar[str]
    version: str
    compute: ComputeCounter

    def decide(self, obs: Observation, rng: np.random.Generator, *, explore: bool = True) -> Decision: ...

    def action_probabilities(self, obs: Observation) -> np.ndarray: ...

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray: ...

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None: ...

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None: ...

    def tick_lifecycle(self, age_years: float, dt_years: float) -> None: ...

    def self_estimates(self) -> dict[str, dict[str, float]]: ...

    def parameter_summary(self) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def clone(self) -> Policy: ...


PolicyFactory = Callable[[AgentInit], Policy]


def masked_softmax(logits: np.ndarray, mask: np.ndarray | None, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax over the last axis that gives masked actions probability 0."""
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class BasePolicy(ABC):
    """Convenience base class. Implementations override ``_logits`` (and learning hooks).

    Provides contract-conformant ``decide``, ``action_probabilities``, batch scoring, and a
    default ``clone``. Subclasses must keep all learnable state JSON-serialisable through
    ``state_dict``/``load_state_dict``.
    """

    policy_class: ClassVar[str] = "base"
    version: str = "0.0.0"

    def __init__(self, init: AgentInit) -> None:
        self.init = init
        self.agent_id = init.agent_id
        self.compute = ComputeCounter()

    # ---- to implement -------------------------------------------------------------
    @abstractmethod
    def _probs(self, obs: Observation) -> np.ndarray:
        """Return the action distribution for ``obs`` without mutating learnable state."""

    def _explain(self, obs: Observation, probs: np.ndarray, action: int) -> dict[str, Any]:
        return {}

    # ---- contract surface ---------------------------------------------------------
    def action_probabilities(self, obs: Observation) -> np.ndarray:
        p = np.asarray(self._probs(obs), dtype=np.float64)
        mask = obs.mask()
        p = np.where(mask, p, 0.0)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            p = mask.astype(np.float64) / mask.sum()
        else:
            p = p / s
        return p

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray:
        out = np.empty((contexts.shape[0], n_actions), dtype=np.float64)
        for i in range(contexts.shape[0]):
            obs = Observation(self.agent_id, -1, age_years, kind, family, contexts[i], n_actions)
            out[i] = self.action_probabilities(obs)
        return out

    def decide(self, obs: Observation, rng: np.random.Generator, *, explore: bool = True) -> Decision:
        p = self.action_probabilities(obs)
        self.compute.decisions += 1
        if explore:
            action = int(rng.choice(len(p), p=p))
        else:
            action = int(np.argmax(p))
        return Decision(action, p, self.policy_class, self.version, self._explain(obs, p, action))

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        return None

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None:
        return None

    def tick_lifecycle(self, age_years: float, dt_years: float) -> None:
        return None

    def self_estimates(self) -> dict[str, dict[str, float]]:
        return {}

    def parameter_summary(self) -> dict[str, Any]:
        return {"policy_class": self.policy_class, "version": self.version, "compute": self.compute.as_dict()}

    def state_dict(self) -> dict[str, Any]:
        return {"policy_class": self.policy_class, "version": self.version, "compute": self.compute.as_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        c = state.get("compute", {})
        self.compute = ComputeCounter(**{k: c[k] for k in c})

    def clone(self) -> Policy:
        return copy.deepcopy(self)  # type: ignore[return-value]


class RandomPolicy(BasePolicy):
    """Reference policy for tests: uniform over available actions. Never learns."""

    policy_class: ClassVar[str] = "random"
    version = "1.0.0"

    def _probs(self, obs: Observation) -> np.ndarray:
        return np.ones(obs.n_actions) / obs.n_actions


def random_policy_factory(init: AgentInit) -> Policy:
    return RandomPolicy(init)  # type: ignore[return-value]
