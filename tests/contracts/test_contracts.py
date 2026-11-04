import json

import numpy as np
import pytest

from hpe.contracts import registry
from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.contracts.runtime import CONTRACT_VERSION, RandomPolicy, masked_softmax, random_policy_factory
from hpe.contracts.schemas import ALL_SCHEMAS, ExperimentDefinition, Scenario, SimulationConfig
from hpe.tasks import SymbolWorld


def test_random_policy_conforms():
    run_policy_conformance(random_policy_factory)


def test_registry_resolves_random():
    f = registry.get_factory("random")
    pol = f(make_init(SymbolWorld(seed=1)))
    assert isinstance(pol, RandomPolicy)


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        registry.get_factory("does_not_exist")


def test_masked_softmax_zeroes_masked():
    p = masked_softmax(np.array([1.0, 5.0, 2.0]), np.array([True, False, True]))
    assert p[1] == 0 and abs(p.sum() - 1) < 1e-12


@pytest.mark.parametrize("model", ALL_SCHEMAS)
def test_json_schema_generates(model):
    schema = model.model_json_schema()
    assert schema["title"] == model.__name__
    json.dumps(schema)


def test_simulation_config_defaults_roundtrip():
    c = SimulationConfig(seed=5)
    assert c.resolved_population_seed() == 5 and c.resolved_task_seed() == 5
    c2 = SimulationConfig.model_validate_json(c.model_dump_json())
    assert c2 == c
    assert c.schema_version == CONTRACT_VERSION


def test_scenario_label_is_fixed():
    with pytest.raises(ValueError):
        Scenario(
            scenario_id="s", decision_id="d", option_id="o", title="t", label="prediction", narrative="n"
        )  # type: ignore[arg-type]


def test_extra_fields_forbidden():
    with pytest.raises(ValueError):
        SimulationConfig(seed=1, bogus=2)  # type: ignore[call-arg]


def test_experiment_definition_minimal():
    ExperimentDefinition.model_validate(
        {
            "experiment_id": "exp",
            "title": "t",
            "hypothesis": "h",
            "falsification_criterion": "f",
            "arms": [{"name": "control", "policy_class": "developmental"}],
            "control_arm": "control",
            "seeds": [0, 1],
            "base_config": {"seed": 0},
            "analysis": {"primary_metric": "m", "primary_metric_definition": "d", "comparison": "c"},
        }
    )
