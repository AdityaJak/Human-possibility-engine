from __future__ import annotations

from typing import Any

from hpe.evaluation.spec import ExperimentSpec

TINY_LAB = {
    "n_agents": 2,
    "episodes_per_family": 20,
    "block": 10,
    "checkpoint_every": 10,
    "transfer_shots": [0, 4],
    "transfer_modes": ["shared", "orthogonal"],
    "retention_gap_years": 0.5,
}


def lab_spec(
    exp_id: str = "tiny_lab",
    arms: list[dict[str, Any]] | None = None,
    seeds: list[int] | None = None,
    contrasts: list[dict[str, Any]] | None = None,
    sensitivity: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> ExperimentSpec:
    arms = arms or [
        {"name": "static_persona", "policy_class": "static_persona"},
        {"name": "plain_rl", "policy_class": "plain_rl"},
    ]
    contrasts = contrasts or [
        {
            "name": "rl_vs_static",
            "primary": True,
            "treatment": "plain_rl",
            "control": "static_persona",
            "metric": "final_heldout_accuracy",
            "kind": "superiority",
            "min_effect": 0.01,
        }
    ]
    return ExperimentSpec.model_validate(
        {
            "protocol": {
                "kind": "learning_lab",
                "params": {**TINY_LAB, **(params or {})},
                "contrasts": contrasts,
                "sensitivity": sensitivity or [],
            },
            "definition": {
                "experiment_id": exp_id,
                "title": "tiny",
                "hypothesis": "h",
                "falsification_criterion": "f",
                "arms": arms,
                "control_arm": arms[0]["name"],
                "seeds": seeds or [0, 1, 2],
                "base_config": {"seed": 0, "n_agents": 2, "years": 1},
                "analysis": {
                    "primary_metric": "final_heldout_accuracy",
                    "primary_metric_definition": "d",
                    "comparison": "c",
                    "n_bootstrap": 500,
                },
            },
        }
    )
