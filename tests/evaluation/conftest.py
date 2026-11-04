import pytest
from eval_helpers import lab_spec

from hpe.evaluation.spec import ExperimentSpec


@pytest.fixture
def tiny_spec() -> ExperimentSpec:
    return lab_spec()
