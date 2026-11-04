import pytest

import hpe.development as dev
from hpe.contracts import registry
from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.contracts.runtime import Policy
from hpe.tasks import SymbolWorld

NON_LEARNING_ABLATIONS = {"use_learning", "use_feedback"}


def test_default_conforms_and_learns():
    run_policy_conformance(dev.build_factory(), check_learning=True)


@pytest.mark.parametrize("flag", dev.ABLATION_FLAGS)
def test_every_ablation_conforms(flag):
    run_policy_conformance(
        dev.build_factory({flag: False}), check_learning=flag not in NON_LEARNING_ABLATIONS
    )


def test_registered_in_registry():
    factory = registry.get_factory("developmental", {"use_memory": False})
    pol = factory(make_init(SymbolWorld(seed=1)))
    assert isinstance(pol, dev.DevelopmentalLearner)
    assert isinstance(pol, Policy)
    assert pol.policy_class == "developmental"
    assert pol.parameter_summary()["flags"]["use_memory"] is False


def test_unknown_params_raise():
    with pytest.raises(ValueError, match="unknown developmental policy params"):
        dev.build_factory({"use_memroy": False})


def test_flags_exposed_in_summary():
    pol = dev.build_factory()(make_init(SymbolWorld(seed=1)))
    flags = pol.parameter_summary()["flags"]
    assert set(flags) == set(dev.ABLATION_FLAGS) and all(flags.values())
