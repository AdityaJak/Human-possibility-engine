"""Experiment specification files (YAML) and preregistration hashing.

An experiment file has two top-level keys:

``definition``  a frozen ``hpe.contracts.schemas.ExperimentDefinition`` (hypothesis, arms, seeds,
                analysis plan, falsification criterion).
``protocol``    how the arms are executed and judged: ``kind`` (``learning_lab`` | ``society``),
                protocol ``params``, preregistered ``contrasts`` (exactly one ``primary``) and
                optional ``sensitivity`` variants.

The preregistration hash covers both keys (minus the hash field and timestamp) and is written to
``definition.json`` *before* any run starts. Re-running into a directory whose stored hash differs
is refused, so a definition cannot be silently edited after results exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hpe.contracts.schemas import ExperimentDefinition
from hpe.core.hashing import stable_hash


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrongestOf(_M):
    strongest_of: list[str] = Field(..., min_length=1)


class ContrastSpec(_M):
    name: str
    treatment: str
    control: str | StrongestOf
    metric: str
    kind: Literal["superiority", "equivalence", "non_inferiority", "exploratory"] = "superiority"
    paired: bool = True
    direction: Literal["greater", "less"] = Field(
        "greater", description="which direction counts as better for the treatment"
    )
    min_effect: float | None = None
    min_effect_kind: Literal["absolute", "relative"] = "absolute"
    margin: float | None = Field(None, description="equivalence / non-inferiority margin (absolute)")
    primary: bool = False
    description: str = ""


class SensitivityVariant(_M):
    name: str
    params: dict[str, Any] = Field(
        default_factory=dict, description="protocol param overrides (lab) or world overrides (society)"
    )
    arms: list[str] | None = Field(None, description="subset of arms to run; default all")
    contrasts: list[str] | None = Field(
        None, description="contrast names to evaluate within this variant; default all"
    )


class ProtocolSpec(_M):
    kind: Literal["learning_lab", "society"]
    params: dict[str, Any] = Field(default_factory=dict)
    contrasts: list[ContrastSpec]
    sensitivity: list[SensitivityVariant] = Field(default_factory=list)
    pipeline_validation: bool = Field(
        False, description="True for scaled-down runs that validate the pipeline only"
    )

    @model_validator(mode="after")
    def _one_primary(self) -> ProtocolSpec:
        n = sum(c.primary for c in self.contrasts)
        if n != 1:
            raise ValueError(f"exactly one primary contrast required, found {n}")
        names = [c.name for c in self.contrasts]
        if len(set(names)) != len(names):
            raise ValueError("contrast names must be unique")
        return self

    @property
    def primary(self) -> ContrastSpec:
        return next(c for c in self.contrasts if c.primary)


class ExperimentSpec(_M):
    protocol: ProtocolSpec
    definition: ExperimentDefinition

    @model_validator(mode="after")
    def _arms_known(self) -> ExperimentSpec:
        arms = {a.name for a in self.definition.arms}
        if self.definition.control_arm not in arms:
            raise ValueError(f"control_arm {self.definition.control_arm!r} not among arms")
        for c in self.protocol.contrasts:
            refs = [c.treatment] + (
                c.control.strongest_of if isinstance(c.control, StrongestOf) else [c.control]
            )
            missing = [r for r in refs if r not in arms]
            if missing:
                raise ValueError(f"contrast {c.name!r} references unknown arms {missing}")
        return self

    def preregistration_payload(self) -> dict[str, Any]:
        d = self.definition.model_dump(mode="json", exclude={"preregistration_hash", "created_at"})
        return {"definition": d, "protocol": self.protocol.model_dump(mode="json")}

    def preregistration_hash(self) -> str:
        return stable_hash(self.preregistration_payload(), length=32)

    def frozen(self) -> ExperimentSpec:
        """Copy with ``definition.preregistration_hash`` set."""
        h = self.preregistration_hash()
        return self.model_copy(
            update={"definition": self.definition.model_copy(update={"preregistration_hash": h})}
        )


def is_simulation_spec_file(path: str | Path) -> bool:
    """True for simulation/learning-lab specs (top-level ``protocol``); False for human-dataset
    preregistrations (top-level ``kind``, run by ``hpe.evaluation.human``)."""
    data = yaml.safe_load(Path(path).read_text())
    return isinstance(data, dict) and "protocol" in data


def simulation_spec_files(directory: str | Path = "experiments") -> list[Path]:
    return [p for p in sorted(Path(directory).glob("*.yaml")) if is_simulation_spec_file(p)]


def load_spec(path: str | Path) -> ExperimentSpec:
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentSpec.model_validate(data)


def scaled_for_pipeline_validation(
    spec: ExperimentSpec, *, n_seeds: int = 3, scale: float = 0.25
) -> ExperimentSpec:
    """Scaled-down copy for pipeline validation. Its hash differs from the preregistered one, and its
    outputs are labelled PIPELINE VALIDATION — never reported as the preregistered result."""
    params = dict(spec.protocol.params)
    if "n_agents" in params:
        params["n_agents"] = max(2, int(round(params["n_agents"] * scale)))
    if "episodes_per_family" in params:
        block = int(params.get("block", 10))
        eps = max(2 * block, int(round(params["episodes_per_family"] * scale / block)) * block)
        params["episodes_per_family"] = eps
        params["checkpoint_every"] = max(block, int(round(eps / 4 / block)) * block)
    if "years" in params:
        params["years"] = max(2.0, float(params["years"]) * scale)
    proto = spec.protocol.model_copy(
        update={"params": params, "pipeline_validation": True, "sensitivity": spec.protocol.sensitivity[:1]}
    )
    d = spec.definition.model_copy(
        update={
            "experiment_id": spec.definition.experiment_id + "__pipeline_validation",
            "seeds": list(spec.definition.seeds[:n_seeds]),
            "title": "[PIPELINE VALIDATION] " + spec.definition.title,
            "preregistration_hash": None,
        }
    )
    return ExperimentSpec(protocol=proto, definition=d)
