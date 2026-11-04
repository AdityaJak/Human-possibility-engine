"""HTTP API request/response contracts (contract v0.1). See docs/INTERFACES.md §4."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from hpe.contracts.schemas import (
    SCHEMA_VERSION,
    AgentFrame,
    AgentState,
    ExperienceEvent,
    ExperimentDefinition,
    ExperimentResult,
    JobStatus,
    OutcomeObservation,
    PolicyClass,
    RunManifest,
    SimulationConfig,
    SocialEdge,
    WorldMap,
    WorldTick,
    _Model,
)


class Health(_Model):
    status: Literal["ok", "degraded"]
    version: str
    contract_version: str
    llm_provider: str = "none"
    offline_demo_runs: int = 0


class RunCreate(_Model):
    config: SimulationConfig
    mode: Literal["background", "interactive"] = Field(
        "background",
        description="background: run to completion as a job; interactive: keep world in memory for step/pause",
    )


class RunInfo(_Model):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    status: JobStatus
    mode: Literal["background", "interactive", "stored"]
    source: Literal["live", "stored_real_output"] = Field(
        "live", description="stored_real_output = precomputed run loaded from reports/, produced by real code"
    )
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_tick: int = 0
    current_year: float = 0.0
    manifest: RunManifest | None = None
    latest_tick: WorldTick | None = None


class StepRequest(_Model):
    ticks: int = Field(1, ge=1, le=1200)
    include_frames: bool = True


class RunControl(_Model):
    action: Literal["pause", "resume", "reset", "cancel"]


class AgentSummary(_Model):
    agent_id: str
    household_id: str
    age_years: float
    stage: str
    alive: bool
    activity: str | None
    job_id: str | None
    education_years: float
    resources: float
    mean_measured_accuracy: float | None
    n_friends: int


class SkillPoint(_Model):
    tick: int
    age_years: float
    family: str
    accuracy: float
    ci_low: float
    ci_high: float
    log_loss: float | None = None
    n_items: int
    split: Literal["holdout", "transfer"] = "holdout"


class ParameterPoint(_Model):
    tick: int
    age_years: float
    summary: dict[str, Any]


class AgentDetail(_Model):
    state: AgentState
    timeline: list[ExperienceEvent]
    skill_curve: list[SkillPoint]
    parameter_history: list[ParameterPoint]
    relationships: list[SocialEdge]
    recent_memories: list[dict[str, Any]] = Field(default_factory=list)
    latest_decision_explanation: dict[str, Any] | None = None


class EventPage(_Model):
    items: list[ExperienceEvent]
    total: int
    offset: int
    limit: int


class AgentPage(_Model):
    items: list[AgentSummary]
    total: int
    offset: int
    limit: int


class EventQuery(_Model):
    agent_id: str | None = None
    types: list[str] | None = None
    tick_from: int | None = None
    tick_to: int | None = None
    offset: int = 0
    limit: int = Field(200, le=5000)


class FramesResponse(_Model):
    run_id: str
    map: WorldMap
    ticks: list[int]
    frames: list[list[AgentFrame]]


class TimeSeriesResponse(_Model):
    run_id: str
    ticks: list[WorldTick]


class ExperimentCreate(_Model):
    definition: ExperimentDefinition


class ExperimentInfo(_Model):
    experiment_id: str
    status: JobStatus
    source: Literal["live", "stored_real_output"] = "live"
    definition: ExperimentDefinition
    result: ExperimentResult | None = None
    outcomes_sample: list[OutcomeObservation] = Field(default_factory=list)


class BenchmarkRecord(_Model):
    benchmark_id: str
    n_agents: int
    years: float
    dt_days: int
    policy_class: PolicyClass
    wall_time_s: float
    peak_rss_mb: float | None
    agent_years_per_s: float
    ticks_per_s: float
    events: int
    hardware: str
    status: Literal["completed", "failed", "timeout"]
    notes: str = ""


class ClaimRecord(_Model):
    """One row on the investor Evidence page."""

    claim_id: str
    claim: str
    status: Literal["supported", "not_supported", "inconclusive", "not_tested", "blocked"]
    evidence_kind: Literal["synthetic_simulation", "human_dataset", "engineering", "none"]
    run_ids: list[str] = Field(default_factory=list)
    metric: str | None = None
    value: str | None = None
    config_ref: str | None = None
    limitations: list[str] = Field(default_factory=list)


class PrivacyRequest(_Model):
    user_id: str


class PrivacyExport(_Model):
    user_id: str
    exported_at: str
    records: dict[str, list[dict[str, Any]]]


class PrivacyDeleteResult(_Model):
    user_id: str
    deleted: dict[str, int]
