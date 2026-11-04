"""Mutable internal state records of the society kernel.

These are plain dataclasses (fast to mutate in the tick loop) with exact JSON round-trips for
snapshots. The public, persisted view is ``hpe.contracts.schemas.AgentState`` and friends,
built on demand by ``World.agent(...)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hpe.contracts.schemas import LifeStage


@dataclass(slots=True)
class AgentRec:
    id: str
    name: str
    household_id: str
    birth_tick: int
    birth_cohort_year: int
    traits: dict[str, float]
    base_params: dict[str, float]
    values: dict[str, float]
    persona: str
    seed: int
    inherited: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    partner_id: str | None = None
    alive: bool = True
    death_tick: int | None = None
    stage: str = LifeStage.INFANT.value
    place_id: str | None = None
    activity: str | None = None
    job_id: str | None = None
    education_years: float = 0.0
    enrolled: bool = False
    graduated: bool = False
    tutored: bool = False
    # affect proxies (modelled control signals, NOT experienced emotions)
    energy: float = 0.8
    stress: float = 0.2
    social: float = 0.5
    motivation: float = 0.5
    # economy
    income_year: float = 0.0
    income_last_year: float = 0.0
    work_ticks_year: int = 0
    absent_ticks_year: int = 0
    # learning bookkeeping (counts only; skill is measured by assessment)
    practice_counts: dict[str, int] = field(default_factory=dict)
    recent_reward: dict[str, float] = field(default_factory=dict)
    last_practice_tick: dict[str, int] = field(default_factory=dict)
    measured: dict[str, dict[str, Any]] = field(default_factory=dict)
    # mechanisms / interventions
    mentor_id: str | None = None
    mentor_until_tick: int = -1
    illness_until_tick: int = -1
    flags: dict[str, bool] = field(default_factory=dict)
    caregiver_household: str | None = None
    n_children_born: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentRec:
        return cls(**d)


@dataclass(slots=True)
class Household:
    id: str
    home_place_id: str
    members: list[str]
    resources: float
    formed_tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Household:
        return cls(**d)


@dataclass(slots=True)
class Job:
    id: str
    workplace_id: str
    family: str
    holder: str | None = None
    start_tick: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        return cls(**d)


@dataclass(slots=True)
class EdgeRec:
    kind: str
    trust: float
    interactions: int
    formed_tick: int

    def to_list(self) -> list[Any]:
        return [self.kind, self.trust, self.interactions, self.formed_tick]

    @classmethod
    def from_list(cls, v: list[Any]) -> EdgeRec:
        return cls(str(v[0]), float(v[1]), int(v[2]), int(v[3]))


# Strength order used when one pair has several relationship kinds.
EDGE_PRIORITY: dict[str, int] = {
    "parent": 9,
    "child": 9,
    "partner": 8,
    "sibling": 7,
    "friend": 6,
    "mentor": 5,
    "tutor": 4,
    "coworker": 3,
    "classmate": 2,
    "acquaintance": 1,
}
FAMILY_EDGE_KINDS = frozenset({"parent", "child", "partner", "sibling"})
