"""Canonical persisted / API schemas (contract v0.1).

Spec Part IX §47B requires: AgentState, CapabilityState, ExperienceEvent, LearningUpdate,
SocialEdge, WorldTick, ExperimentDefinition, RunManifest, OutcomeObservation, EvidenceRecord
and Scenario. JSON Schema and TypeScript types are generated from these models by
``scripts/export_contracts.py``; CI fails if the generated files are stale.

Changing any field requires a versioned ADR (docs/ADRS) and a CONTRACT_VERSION bump.
Additive optional fields are a MINOR bump; renames/removals are MAJOR.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpe.contracts.runtime import CONTRACT_VERSION

SCHEMA_VERSION = CONTRACT_VERSION


def utcnow() -> datetime:
    return datetime.now(UTC)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------------------


class PolicyClass(StrEnum):
    STATIC_PERSONA = "static_persona"
    MEMORY_ONLY = "memory_only"
    PLAIN_RL = "plain_rl"
    DEVELOPMENTAL = "developmental"
    RANDOM = "random"


class LifeStage(StrEnum):
    INFANT = "infant"  # 0-2
    CHILD = "child"  # 3-11
    ADOLESCENT = "adolescent"  # 12-17
    ADULT = "adult"  # 18-64
    ELDER = "elder"  # 65+


class EventType(StrEnum):
    BIRTH = "birth"
    DEATH = "death"
    STAGE_CHANGE = "stage_change"
    SCHOOL_ENROLL = "school_enroll"
    SCHOOL_GRADUATE = "school_graduate"
    TUTORING_SESSION = "tutoring_session"
    LEARNING_EPISODE = "learning_episode"  # aggregated per agent per tick
    POLICY_UPDATE = "policy_update"  # aggregated LearningUpdate summary
    ASSESSMENT = "assessment"  # held-out skill measurement (no learning)
    DECISION = "decision"  # sampled activity decisions with explanation
    JOB_START = "job_start"
    JOB_END = "job_end"
    WAGE = "wage"
    SOCIAL_INTERACTION = "social_interaction"
    FRIENDSHIP_FORMED = "friendship_formed"
    PARTNERSHIP_FORMED = "partnership_formed"
    MENTORSHIP = "mentorship"
    SHOCK = "shock"
    INTERVENTION = "intervention"
    INHERITANCE = "inheritance"
    CHECKPOINT = "checkpoint"
    RUN_START = "run_start"
    RUN_END = "run_end"


class EvidenceSourceType(StrEnum):
    USER_FACT = "user_fact"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"
    EXTERNAL_STUDY = "external_study"
    SIMULATED_OUTCOME = "simulated_outcome"
    ASSUMPTION = "assumption"
    NARRATIVE = "narrative"


class ValidationStatus(StrEnum):
    VALIDATED = "validated"  # passed a defined external validation in a defined domain
    UNVALIDATED = "unvalidated"  # real source, not validated for this use
    ILLUSTRATIVE = "illustrative"  # imaginative / narrative only
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # we explicitly cannot predict this


class DataNamespace(StrEnum):
    """Spec §22 data separation. Every stored row belongs to exactly one namespace."""

    SYNTHETIC_SIMULATION = "synthetic_simulation"
    EXTERNAL_DATASET = "external_dataset"
    USER_PRIVATE = "user_private"
    EXPERIMENT = "experiment"
    EVIDENCE = "evidence"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------------------


class InterventionKind(StrEnum):
    TUTORING_ACCESS = "tutoring_access"  # extra full-feedback instruction for targeted agents
    SCHOOL_ACCESS = "school_access"  # whether targeted children can enrol in school
    MIDLIFE_EDUCATION = "midlife_education"  # adult education opens for targeted adults
    SOCIAL_ISOLATION = "social_isolation"  # removes non-family social learning/interaction
    MISINFORMATION = "misinformation"  # seeded agents broadcast wrong demonstrations
    CASH_TRANSFER = "cash_transfer"  # one-off/periodic resource transfer to households
    CAREGIVING_SUPPORT = "caregiving_support"  # extra caregiver practice time for young children
    CROSS_FOSTER = "cross_foster"  # newborns swapped between households (Exp 5)


class InterventionTarget(_Model):
    age_min: float = 0.0
    age_max: float = 200.0
    fraction: float = Field(
        1.0, ge=0.0, le=1.0, description="fraction of eligible agents, assigned by seeded randomisation"
    )
    household_ids: list[str] | None = None
    agent_ids: list[str] | None = None


class Intervention(_Model):
    name: str
    kind: InterventionKind
    target: InterventionTarget = Field(default_factory=InterventionTarget)
    start_year: float = 0.0
    end_year: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class WorldParams(_Model):
    """Abstract world parameters. These are modelling assumptions, not empirical estimates."""

    n_households: int | None = Field(None, description="default: n_agents // 4")
    initial_age_distribution: Literal["uniform_0_60", "young_families", "newborn_cohort"] = "young_families"
    max_population: int | None = Field(None, description="soft carrying capacity; default 1.5 * n_agents")
    annual_birth_rate_per_couple: float = 0.18
    mortality_gompertz_a: float = 0.00005
    mortality_gompertz_b: float = 0.095
    max_age_years: float = 105.0
    school_start_age: float = 6.0
    school_end_age: float = 18.0
    school_episodes_per_tick: int = 6
    school_full_feedback_fraction: float = 0.25
    tutoring_episodes_per_tick: int = 4
    self_study_episodes_per_tick: int = 4
    work_episodes_per_tick: int = 4
    caregiver_episodes_per_tick: int = 2
    peer_demonstrations_per_social_tick: int = 2
    baseline_tutoring_fraction: float = Field(
        0.0, description="share of children with tutoring absent any intervention"
    )
    shock_rate_per_year: float = 0.15
    retirement_age: float = 67.0
    enable_age_gate: bool = True
    label_noise: float = Field(0.05, ge=0.0, le=0.5)
    n_jobs_per_100_agents: int = 45
    wage_base: float = 1.0
    consumption_per_member: float = 0.55


class LogConfig(_Model):
    frames_every_ticks: int = Field(1, ge=0, description="0 disables map frames")
    decision_sample_rate: float = Field(
        0.02, ge=0.0, le=1.0, description="share of activity decisions logged with explanations"
    )
    log_learning_episodes: bool = True
    assessment_interval_years: float = 1.0
    checkpoint_every_years: float | None = 10.0


class SimulationConfig(_Model):
    schema_version: str = SCHEMA_VERSION
    seed: int = 0
    population_seed: int | None = Field(
        None, description="initial population; defaults to seed (matched across twin arms)"
    )
    task_seed: int | None = Field(None, description="task-world rules; defaults to seed")
    n_agents: int = Field(100, ge=1, le=20000)
    years: float = Field(80.0, gt=0, le=150)
    dt_days: int = Field(30, ge=1, le=366)
    policy_class: PolicyClass = PolicyClass.DEVELOPMENTAL
    policy_params: dict[str, Any] = Field(default_factory=dict)
    world: WorldParams = Field(default_factory=WorldParams)
    interventions: list[Intervention] = Field(default_factory=list)
    log: LogConfig = Field(default_factory=LogConfig)
    max_events: int | None = 5_000_000
    label: str | None = None

    def resolved_population_seed(self) -> int:
        return self.seed if self.population_seed is None else self.population_seed

    def resolved_task_seed(self) -> int:
        return self.seed if self.task_seed is None else self.task_seed


# ---------------------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------------------


class CapabilityState(_Model):
    """Skill in one task family. ``measured_*`` comes ONLY from held-out assessment;
    ``self_estimate_*`` is the policy's internal belief (e.g. Beta mastery posterior) and is
    never treated as evidence of skill."""

    family: str
    measured_accuracy: float | None = None
    measured_ci_low: float | None = None
    measured_ci_high: float | None = None
    measured_log_loss: float | None = None
    n_assessment_items: int = 0
    last_assessed_tick: int | None = None
    self_estimate_mean: float | None = None
    self_estimate_sd: float | None = None
    practice_episodes: int = 0
    last_practice_tick: int | None = None


class PolicyDescriptor(_Model):
    policy_class: PolicyClass
    version: str
    parameter_summary: dict[str, Any] = Field(default_factory=dict)


class AffectProxies(_Model):
    """Modelled control signals. Labelled proxies; not experienced emotions."""

    stress: float = 0.0
    energy: float = 1.0
    social_connection: float = 0.5
    motivation: float = 0.5


class AgentState(_Model):
    schema_version: str = SCHEMA_VERSION
    agent_id: str
    run_id: str | None = None
    household_id: str
    birth_tick: int
    birth_cohort_year: int
    age_years: float
    stage: LifeStage
    alive: bool = True
    death_tick: int | None = None
    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    partner_id: str | None = None
    traits: dict[str, float] = Field(default_factory=dict)
    persona: str = Field(
        "", description="generated biography text; descriptive only, used by static baseline"
    )
    resources: float = 0.0
    household_resources: float = 0.0
    place_id: str | None = None
    x: float = 0.0
    y: float = 0.0
    activity: str | None = None
    job_id: str | None = None
    education_years: float = 0.0
    enrolled_school: bool = False
    tutored: bool = False
    affect: AffectProxies = Field(default_factory=AffectProxies)
    values: dict[str, float] = Field(default_factory=dict)
    capabilities: dict[str, CapabilityState] = Field(default_factory=dict)
    policy: PolicyDescriptor
    memory_items: int = 0
    n_friends: int = 0


# ---------------------------------------------------------------------------------------
# Events, learning, social graph, ticks
# ---------------------------------------------------------------------------------------


class Provenance(_Model):
    source: Literal["world", "policy", "evaluation", "intervention", "storage"]
    mechanism: str = Field(
        ..., description="name of the code mechanism that produced the event, e.g. 'world.school.enrol'"
    )
    rng_stream: str | None = None
    code_version: str | None = None


class ExperienceEvent(_Model):
    schema_version: str = SCHEMA_VERSION
    seq: int = Field(..., description="monotonic sequence number within a run (replay order)")
    run_id: str
    tick: int
    day: int
    type: EventType
    agent_id: str | None = None
    other_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    namespace: DataNamespace = DataNamespace.SYNTHETIC_SIMULATION


class LearningUpdate(_Model):
    schema_version: str = SCHEMA_VERSION
    run_id: str | None = None
    agent_id: str
    tick: int
    policy_class: PolicyClass
    family: str
    source: str
    feedback_mode: Literal["none", "bandit", "full"]
    n_samples: int
    loss: float
    param_delta_norm: float
    n_updates_total: int


class SocialEdgeKind(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    PARTNER = "partner"
    FRIEND = "friend"
    CLASSMATE = "classmate"
    COWORKER = "coworker"
    MENTOR = "mentor"
    TUTOR = "tutor"


class SocialEdge(_Model):
    src: str
    dst: str
    kind: SocialEdgeKind
    trust: float = Field(0.5, ge=0.0, le=1.0)
    interactions: int = 0
    formed_tick: int = 0


class AgentFrame(_Model):
    """Compact per-agent map state for live/replay rendering."""

    id: str
    x: float
    y: float
    place_id: str | None
    activity: str | None
    age: float
    alive: bool
    household_id: str


class WorldTick(_Model):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    tick: int
    day: int
    year: float
    population: int
    births: int = 0
    deaths: int = 0
    mean_measured_accuracy: dict[str, float] = Field(default_factory=dict)
    employment_rate: float = 0.0
    school_enrolment_rate: float = 0.0
    resource_gini: float = 0.0
    mean_resources: float = 0.0
    events_emitted: int = 0
    age_histogram: dict[str, int] = Field(default_factory=dict)
    wall_time_ms: float = 0.0
    frames: list[AgentFrame] | None = None


class Place(_Model):
    place_id: str
    kind: Literal["home", "school", "workplace", "gathering", "tutoring_center", "elder_center"]
    name: str
    x: float
    y: float
    capacity: int | None = None


class WorldMap(_Model):
    width: float
    height: float
    places: list[Place]


# ---------------------------------------------------------------------------------------
# Experiments, manifests, outcomes
# ---------------------------------------------------------------------------------------


class ArmDefinition(_Model):
    name: str
    policy_class: PolicyClass
    policy_params: dict[str, Any] = Field(default_factory=dict)
    interventions: list[Intervention] = Field(default_factory=list)
    world_overrides: dict[str, Any] = Field(default_factory=dict)


class AnalysisPlan(_Model):
    primary_metric: str
    primary_metric_definition: str
    comparison: str = Field(..., description="e.g. 'treatment - control, paired by seed'")
    test: Literal["paired_bootstrap", "unpaired_bootstrap", "permutation", "welch_t"] = "paired_bootstrap"
    alpha: float = 0.05
    n_bootstrap: int = 2000
    minimum_effect_of_interest: float | None = None
    secondary_metrics: list[str] = Field(default_factory=list)


class ExperimentDefinition(_Model):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    title: str
    hypothesis: str
    falsification_criterion: str
    arms: list[ArmDefinition]
    control_arm: str
    seeds: list[int]
    base_config: SimulationConfig
    analysis: AnalysisPlan
    ablations: list[str] = Field(default_factory=list)
    validity_notes: list[str] = Field(default_factory=list)
    preregistration_hash: str | None = Field(
        None, description="stable hash of this definition, frozen before any run"
    )
    created_at: datetime = Field(default_factory=utcnow)


class ComputeReport(_Model):
    wall_time_s: float
    cpu_time_s: float | None = None
    peak_rss_mb: float | None = None
    ticks: int = 0
    agent_years: float = 0.0
    events: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_assumptions: str = "local CPU only; no paid model calls"
    hardware: str | None = None


class RunManifest(_Model):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=utcnow)
    status: JobStatus = JobStatus.QUEUED
    code_version: str | None = Field(None, description="git commit sha (+dirty flag)")
    contract_version: str = CONTRACT_VERSION
    config: SimulationConfig
    config_hash: str
    experiment_id: str | None = None
    arm: str | None = None
    policy_class: PolicyClass
    policy_version: str | None = None
    task_world_hash: str | None = None
    dataset_hashes: dict[str, str] = Field(default_factory=dict)
    event_log_path: str | None = None
    event_log_sha256: str | None = None
    n_events: int = 0
    final_state_hash: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    compute: ComputeReport | None = None
    nondeterminism: list[str] = Field(
        default_factory=list, description="any non-seeded inputs, e.g. external LLM calls"
    )
    error: str | None = None


class OutcomeObservation(_Model):
    schema_version: str = SCHEMA_VERSION
    run_id: str | None = None
    experiment_id: str | None = None
    arm: str | None = None
    seed: int | None = None
    agent_id: str | None = None
    metric: str
    value: float
    unit: str = ""
    tick: int | None = None
    age_years: float | None = None
    split: Literal["train", "holdout", "transfer", "human_holdout", "n/a"] = "n/a"
    source: Literal["simulation", "human_dataset"] = "simulation"


class EffectEstimate(_Model):
    metric: str
    contrast: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    p_value: float | None = None
    standardized_effect: float | None = None
    n_units: int
    unit_of_analysis: str = "seed"
    method: str
    passed_minimum_effect: bool | None = None


class ArmSummary(_Model):
    arm: str
    n_runs: int
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="metric -> {mean, sd, ci_low, ci_high, n}"
    )


class ExperimentResult(_Model):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    preregistration_hash: str
    status: Literal["passed", "failed", "inconclusive", "error"]
    status_reason: str
    arms: list[ArmSummary]
    effects: list[EffectEstimate]
    run_ids: list[str]
    sensitivity: list[dict[str, Any]] = Field(default_factory=list)
    ablations: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    synthetic_disclaimer: str = (
        "Simulated result from a synthetic world. Effects may simply reflect assumptions built into the "
        "simulator and are not estimates of real-world causal effects."
    )
    generated_from: str = Field("raw event logs", description="what the analysis was regenerated from")
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------------------
# Evidence and Future Self
# ---------------------------------------------------------------------------------------


class EvidenceRecord(_Model):
    schema_version: str = SCHEMA_VERSION
    evidence_id: str
    domain: str
    source_type: EvidenceSourceType
    claim: str
    citation: str | None = None
    data_ref: str | None = None
    applicability: str = ""
    date: str | None = None
    uncertainty: str = ""
    validation_status: ValidationStatus
    run_id: str | None = None
    namespace: DataNamespace = DataNamespace.EVIDENCE


class ValueWeight(_Model):
    key: str = Field(
        ...,
        max_length=64,
        description="e.g. autonomy, income, relationships, stability, learning, health, time",
    )
    label: str = Field(..., max_length=200)
    weight: float = Field(..., ge=0.0, le=1.0)


class DecisionOption(_Model):
    option_id: str = Field(..., max_length=64)
    label: str = Field(..., max_length=200)
    description: str = Field("", max_length=2000)
    kind: Literal["current_path", "new_job", "learn_skill", "start_project", "education", "other"] = "other"
    inputs: dict[str, float] = Field(
        default_factory=dict, description="optional user-provided numbers, e.g. monthly_income, savings"
    )


class DecisionRequest(_Model):
    schema_version: str = SCHEMA_VERSION
    user_id: str = Field(..., max_length=128, description="pseudonymous demo user id")
    consent_given: bool
    question: str = Field(..., max_length=4000)
    options: list[DecisionOption] = Field(..., min_length=2, max_length=4)
    values: list[ValueWeight] = Field(..., max_length=16)
    constraints: dict[str, float | str] = Field(default_factory=dict)
    horizon_years: float = Field(3.0, gt=0, le=10)


class Calculation(_Model):
    name: str
    formula: str
    inputs: dict[str, float]
    result: float | None
    unit: str
    evidence_id: str
    note: str = ""


class Tradeoff(_Model):
    value_key: str
    direction: Literal["likely_better", "likely_worse", "mixed", "unclear"]
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReversibleExperiment(_Model):
    title: str
    description: str
    duration_days: int
    cost_note: str = ""
    reversibility: Literal["fully_reversible", "mostly_reversible"] = "fully_reversible"
    learns_about: list[str] = Field(default_factory=list)


class Scenario(_Model):
    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    decision_id: str
    option_id: str
    title: str
    label: Literal["illustrative scenario"] = "illustrative scenario"
    narrative: str
    narrative_source: Literal["template", "llm_external"] = "template"
    assumptions: list[EvidenceRecord] = Field(default_factory=list)
    tradeoffs: list[Tradeoff] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    cannot_predict: list[str] = Field(default_factory=list)
    reversible_experiments: list[ReversibleExperiment] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Illustrative scenario conditioned on the assumptions listed. It is not a prediction of your future "
        "and contains no individualized probability estimates."
    )


class SafetyNotice(_Model):
    category: Literal["health", "financial", "legal", "relationship", "crisis", "none"]
    message: str
    referral: str | None = None


class ScenarioResponse(_Model):
    schema_version: str = SCHEMA_VERSION
    decision_id: str
    scenarios: list[Scenario]
    evidence: list[EvidenceRecord]
    safety: SafetyNotice | None = None
    value_alignment_note: str
    simulation_separation_note: str = (
        "No result from the synthetic agent society is used as evidence about your life."
    )


class Reflection(_Model):
    user_id: str = Field(..., max_length=128)
    decision_id: str = Field(..., max_length=128)
    chosen_option_id: str | None = Field(None, max_length=64)
    what_happened: str = Field("", max_length=4000)
    values_changed: list[ValueWeight] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class ConsentRecord(_Model):
    user_id: str
    purpose: Literal["future_self_demo", "reflection_storage", "personalization"]
    granted: bool
    created_at: datetime = Field(default_factory=utcnow)
    namespace: DataNamespace = DataNamespace.USER_PRIVATE


ALL_SCHEMAS: list[type[BaseModel]] = [
    AgentState,
    CapabilityState,
    ExperienceEvent,
    LearningUpdate,
    SocialEdge,
    WorldTick,
    AgentFrame,
    WorldMap,
    Place,
    SimulationConfig,
    Intervention,
    ExperimentDefinition,
    ExperimentResult,
    EffectEstimate,
    ArmSummary,
    RunManifest,
    ComputeReport,
    OutcomeObservation,
    EvidenceRecord,
    Scenario,
    ScenarioResponse,
    DecisionRequest,
    Reflection,
    ConsentRecord,
]
