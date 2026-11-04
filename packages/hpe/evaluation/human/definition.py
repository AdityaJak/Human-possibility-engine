"""Preregistration schemas for human-data experiments.

``hpe.contracts.schemas.ExperimentDefinition`` is simulation-centric (arms are policy classes,
``base_config`` is a SimulationConfig), so human-data experiments use these local schemas.
The preregistration hash is computed with ``hpe.core.stable_hash`` over the definition with
``preregistration_hash`` removed, and written to disk BEFORE any data is analysed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hpe.core.hashing import stable_hash


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitPlan(_M):
    unit: Literal["participant"] = "participant"
    train_fraction: float = Field(0.7, gt=0, lt=1)
    stratify_by: Literal["experiment"] = "experiment"
    seed: int
    robustness_seeds: list[int] = Field(default_factory=list)


class HumanAnalysisPlan(_M):
    primary_metric: str
    primary_metric_definition: str
    test: Literal["paired_bootstrap"] = "paired_bootstrap"
    alpha: float = 0.05
    n_bootstrap: int = 2000
    minimum_effect_of_interest: float
    primary_contrast: str
    secondary_contrasts: list[str] = Field(default_factory=list)
    status_rule: str


class ParadigmPlan(_M):
    paradigm: Literal["mcpl", "igt"]
    role: Literal["primary", "secondary"]
    experiments_prefix: str


class HumanHeldoutDefinition(_M):
    experiment_id: str
    title: str
    hypothesis: str
    falsification_criterion: str
    dataset: str
    paradigms: list[ParadigmPlan]
    models: list[str]
    model_groups: dict[str, list[str]]
    grids: dict[str, dict[str, dict[str, list[float]]]] = Field(
        default_factory=dict, description="model -> paradigm -> param -> values"
    )
    lapse_grid: list[float]
    igt_reward_scale_grid: list[float]
    age_years: float = 25.0
    split: SplitPlan
    analysis: HumanAnalysisPlan
    validity_notes: list[str] = Field(default_factory=list)
    preregistration_hash: str | None = None


class PublishedValue(_M):
    term: str
    published_percent: float
    published_p: str
    source: str
    verified_text: str


class CausalGroundingDefinition(_M):
    experiment_id: str
    title: str
    hypothesis: str
    falsification_criterion: str
    dataset: str
    estimand: str
    outcome: str
    controls: list[str]
    centring: str
    sensitivity: list[str]
    n_bootstrap: int = 2000
    seed: int
    published: list[PublishedValue]
    comparison_rule: str
    identification_assumptions: list[str]
    validity_notes: list[str] = Field(default_factory=list)
    preregistration_hash: str | None = None


class V2Paradigm(_M):
    paradigm: Literal["mcpl", "igt", "bandit2", "srl3", "horizon", "casino", "wpt", "garcia", "described"]
    experiments: list[str] = Field(..., description="experiment files (must appear in the partitions file)")
    role: Literal["primary", "secondary", "exploratory"] = "exploratory"


class V2AnalysisPlan(_M):
    primary_metric: str = "report_nll_per_choice"
    primary_metric_definition: str
    n_bootstrap: int = 2000
    alpha: float = 0.05
    minimum_effect_of_interest: float = 0.02
    contrasts: list[str] = Field(default_factory=list)
    status_rule: str


class HumanV2Definition(_M):
    """Phase II human-validation protocol (partition-aware, equal tuning)."""

    experiment_id: str
    title: str
    hypothesis: str
    falsification_criterion: str
    dataset: str
    partitions_file: str = "reports/phase2/partitions/psych101.json"
    fit_partition: Literal["dev", "tune"] = "dev"
    select_partition: Literal["dev", "tune"] = "tune"
    report_partition: Literal["dev", "tune", "test"] = "tune"
    paradigms: list[V2Paradigm]
    models: list[str]
    reference_models: list[str]
    grids: dict[str, dict[str, dict[str, list[float]]]] = Field(
        default_factory=dict, description="model -> paradigm -> param -> values (<= 24 configurations each)"
    )
    max_grid_configurations: int = 24
    lapse_grid: list[float]
    igt_reward_scale_grid: list[float] = Field(default_factory=lambda: [250.0, 500.0])
    seed: int = 20261105
    analysis: V2AnalysisPlan
    validity_notes: list[str] = Field(default_factory=list)
    preregistration_hash: str | None = None


def preregistration_hash(defn: BaseModel) -> str:
    d: dict[str, Any] = defn.model_dump(mode="json")
    d.pop("preregistration_hash", None)
    return stable_hash(d)


AnyDefinition = HumanHeldoutDefinition | CausalGroundingDefinition | HumanV2Definition


def load_definition(path: Path) -> AnyDefinition:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    kind = raw.pop("kind")
    model: type[HumanHeldoutDefinition] | type[CausalGroundingDefinition] | type[HumanV2Definition]
    if kind == "human_heldout":
        model = HumanHeldoutDefinition
    elif kind == "human_v2":
        model = HumanV2Definition
    else:
        model = CausalGroundingDefinition
    defn = model.model_validate(raw)
    defn.preregistration_hash = preregistration_hash(defn)
    return defn
