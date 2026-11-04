"""Runs only once the research/development lane's learner is importable (skipped otherwise)."""

import pytest
from eval_helpers import lab_spec

from hpe.contracts import registry
from hpe.evaluation.runner import run_experiment

ABLATION_FLAGS = [
    "use_memory",
    "use_feedback",
    "use_learning",
    "use_shared_representation",
    "use_plasticity_schedule",
    "use_social_learning",
    "use_curriculum",
    "use_forgetting",
]


def _dev_registered() -> bool:
    try:
        registry.get_factory("developmental")
    except (KeyError, ImportError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _dev_registered(), reason="developmental learner not integrated in this build"
)


@pytest.mark.parametrize("flag", ABLATION_FLAGS)
def test_ablation_flags_accepted(flag):
    from hpe.contracts.conformance import run_policy_conformance

    run_policy_conformance(registry.get_factory("developmental", {flag: False}))


def test_developmental_arm_runs_in_learning_lab(tmp_path):
    spec = lab_spec(
        exp_id="dev_integration",
        arms=[
            {"name": "plain_rl", "policy_class": "plain_rl"},
            {"name": "developmental", "policy_class": "developmental"},
            {"name": "frozen", "policy_class": "developmental", "policy_params": {"use_learning": False}},
        ],
        contrasts=[
            {
                "name": "dev_vs_rl",
                "primary": True,
                "treatment": "developmental",
                "control": "plain_rl",
                "metric": "final_heldout_accuracy",
            }
        ],
        params={"demonstrations_per_block": 1},
        seeds=[0, 1],
    )
    res = run_experiment(spec, tmp_path, make_figures=False)
    assert res.status in {"passed", "failed", "inconclusive"}
    assert {a.arm for a in res.arms} == {"plain_rl", "developmental", "frozen"}
