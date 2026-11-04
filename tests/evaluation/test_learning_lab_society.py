from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from eval_helpers import TINY_LAB

from hpe.contracts import registry
from hpe.contracts.schemas import AgentState, CapabilityState, PolicyDescriptor, SimulationConfig
from hpe.evaluation import society
from hpe.evaluation.contamination import InstrumentedSymbolWorld, run_contamination_suite
from hpe.evaluation.learning_lab import LabContext, agent_init, resolve_params, run_agent
from hpe.evaluation.runner import run_experiment
from hpe.evaluation.spec import ExperimentSpec
from hpe.tasks import SymbolWorld


class RecordingWorld(InstrumentedSymbolWorld):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.sequence: list[tuple[str, tuple[int, ...]]] = []

    def sample_item(self, family, rng, split="train", difficulty=None):
        item = super().sample_item(family, rng, split=split, difficulty=difficulty)
        if split == "train":
            self.sequence.append((family, item.symbol_ids))
        return item


def _ctx(arm: str, params: dict[str, Any] | None = None) -> LabContext:
    return LabContext(
        "t", arm, 4, f"t__{arm}", resolve_params({**TINY_LAB, **(params or {}), "skip_transfer": True})
    )


def test_common_random_numbers_identical_items_across_arms():
    seqs = {}
    for name in ("static_persona", "memory_only", "plain_rl"):
        w = RecordingWorld(4)
        run_agent(registry.get_factory(name), w, 0, _ctx(name, {"demonstrations_per_block": 1}))
        seqs[name] = w.sequence
    assert seqs["static_persona"] == seqs["memory_only"] == seqs["plain_rl"]
    # practice (20) + demonstrations (2) per family, plus the fixed 60-item train-split set used for the
    # generalisation-gap measurement (train symbols only)
    assert len(seqs["plain_rl"]) == 3 * (20 + 2) + 3 * 60


def test_agent_init_identical_across_arms_and_varies_by_index():
    w = SymbolWorld(seed=1)
    a, b = agent_init(1, 0, w), agent_init(1, 0, w)
    assert a.traits == b.traits and a.seed == b.seed
    assert agent_init(1, 1, w).traits != a.traits


def test_lab_outputs_expected_metrics_and_plain_rl_beats_static_with_budget():
    w = InstrumentedSymbolWorld(2)
    p = {**TINY_LAB, "episodes_per_family": 150, "checkpoint_every": 50, "skip_transfer": True}
    rl = run_agent(registry.get_factory("plain_rl"), w, 0, LabContext("t", "rl", 2, "r", resolve_params(p)))
    st = run_agent(
        registry.get_factory("static_persona"), w, 0, LabContext("t", "st", 2, "s", resolve_params(p))
    )
    get = lambda rows, m: next(r.value for r in rows if r.metric == m)  # noqa: E731
    assert get(rl, "final_heldout_accuracy") > get(st, "final_heldout_accuracy")
    names = {r.metric for r in rl}
    for m in (
        "final_log_loss",
        "heldout_ece",
        "generalization_gap",
        "retention_change",
        "zero_shot/barter",
        "curve/heldout_accuracy/sort",
        "episodes_to_criterion/sort",
        "compute/decisions",
    ):
        assert m in names, m
    assert w.leakage_counts()["holdout_symbols_in_practice"] == 0


def test_transfer_metrics_present():
    w = InstrumentedSymbolWorld(3)
    rows = run_agent(
        registry.get_factory("plain_rl"), w, 0, LabContext("t", "rl", 3, "r", resolve_params({**TINY_LAB}))
    )
    names = {r.metric for r in rows}
    for m in (
        "transfer_auc/shared/trained",
        "transfer_auc/shared/fresh",
        "transfer_gain/shared",
        "transfer_auc/orthogonal/trained",
        "transfer_curve/shared/trained",
    ):
        assert m in names
    gain = next(r.value for r in rows if r.metric == "transfer_gain/shared")
    assert gain == pytest.approx(0.0), "independent specialists cannot transfer"


def test_unknown_lab_param_rejected():
    with pytest.raises(ValueError):
        resolve_params({"nonsense": 1})


def test_contamination_suite_clean():
    rep = run_contamination_suite(seed=1, episodes=30)
    assert rep["clean"], rep
    assert {"static_persona", "memory_only", "plain_rl"} <= set(rep["policies"])


# ------------------------------------------------------------------------------------------
# Society protocol with a fake WorldRunner (the real hpe.world is built by the sim/world lane)
# ------------------------------------------------------------------------------------------


class FakeWorld:
    """Minimal WorldRunner surface used by hpe.evaluation.society. Tutoring 'works' only for learners,
    so the analysis pipeline (primary contrast, mechanism check) can be exercised end to end."""

    def __init__(self, config: SimulationConfig, factory, run_id: str) -> None:
        self.config, self.run_id, self.tick, self._year = config, run_id, 0, 0.0
        self.rng = np.random.default_rng(config.seed)
        self.base = self.rng.uniform(0.55, 0.65, size=40)

    @property
    def year(self) -> float:
        return self._year

    def run(self, *, until_year=None, progress=None):
        self._year = float(until_year if until_year is not None else self.config.years)
        self.tick = int(self._year * 12)

    def agents(self, *, include_dead: bool = True):
        tutored = any(i.kind.value == "tutoring_access" for i in self.config.interventions)
        learns = self.config.policy_class.value != "static_persona"
        out = []
        for i, b in enumerate(self.base):
            acc = float(b + (0.05 if (tutored and learns) else 0.0))
            out.append(
                AgentState(
                    agent_id=f"a{i:04d}",
                    household_id=f"h{i // 4:03d}",
                    birth_tick=0,
                    birth_cohort_year=0,
                    age_years=float(5 + i * 0.5),
                    stage="child",
                    tutored=tutored,
                    parents=[f"a{(i + 1) % 40:04d}"],
                    capabilities={
                        f: CapabilityState(family=f, measured_accuracy=acc)
                        for f in ("sort", "compare", "signal")
                    },
                    policy=PolicyDescriptor(policy_class=self.config.policy_class, version="fake"),
                )
            )
        return out

    def manifest(self):
        raise NotImplementedError("fake")

    def event_stream_hash(self) -> str:
        return "fake"


def _society_spec() -> ExperimentSpec:
    tut = [{"name": "t", "kind": "tutoring_access", "target": {"age_min": 5, "age_max": 15}}]
    return ExperimentSpec.model_validate(
        {
            "protocol": {
                "kind": "society",
                "params": {
                    "metrics": [
                        {
                            "name": "acc_10_20",
                            "fn": "cohort_mean_accuracy",
                            "args": {"age_min": 10, "age_max": 20},
                        },
                        {
                            "name": "acc_mid",
                            "fn": "cohort_mean_accuracy",
                            "args": {"age_min": 0, "age_max": 99},
                            "at_year": 1,
                        },
                    ]
                },
                "contrasts": [
                    {
                        "name": "tut",
                        "primary": True,
                        "treatment": "dev_t",
                        "control": "dev_c",
                        "metric": "acc_10_20",
                        "min_effect": 0.02,
                    },
                    {
                        "name": "mech",
                        "treatment": "st_t",
                        "control": "st_c",
                        "metric": "acc_10_20",
                        "kind": "equivalence",
                        "margin": 0.01,
                    },
                ],
            },
            "definition": {
                "experiment_id": "fake_society",
                "title": "t",
                "hypothesis": "h",
                "falsification_criterion": "f",
                "arms": [
                    {"name": "dev_c", "policy_class": "plain_rl"},
                    {"name": "dev_t", "policy_class": "plain_rl", "interventions": tut},
                    {"name": "st_c", "policy_class": "static_persona"},
                    {"name": "st_t", "policy_class": "static_persona", "interventions": tut},
                ],
                "control_arm": "dev_c",
                "seeds": [0, 1, 2, 3, 4],
                "base_config": {"seed": 0, "n_agents": 10, "years": 2},
                "analysis": {
                    "primary_metric": "acc_10_20",
                    "primary_metric_definition": "d",
                    "comparison": "c",
                    "n_bootstrap": 500,
                },
            },
        }
    )


def test_society_protocol_end_to_end_with_fake_world(tmp_path, monkeypatch):
    monkeypatch.setattr(society, "default_world_builder", lambda c, f, r: FakeWorld(c, f, r))
    monkeypatch.setattr(society, "world_available", lambda: True)
    res = run_experiment(_society_spec(), tmp_path, make_figures=False)
    assert res.status == "passed", res.status_reason
    mech = next(e for e in res.effects if e.contrast.startswith("mech"))
    assert abs(mech.estimate) < 1e-9


def test_society_arms_skipped_when_world_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(society, "world_available", lambda: False)
    res = run_experiment(_society_spec(), tmp_path, make_figures=False)
    assert res.status == "inconclusive"
    assert any("hpe.world.World not available" in lim for lim in res.limitations)


def test_build_config_matches_seeds_across_arms():
    base = SimulationConfig(seed=0, n_agents=50, years=3)
    a = society.build_config(
        base,
        seed=7,
        policy_class="plain_rl",
        policy_params={},
        interventions=[],
        world_overrides={},
        params={},
        label="a",
    )
    b = society.build_config(
        base,
        seed=7,
        policy_class="static_persona",
        policy_params={},
        interventions=[],
        world_overrides={"label_noise": 0.2},
        params={"world_overrides": {"tutoring_episodes_per_tick": 9}},
        label="b",
    )
    assert (a.seed, a.population_seed, a.task_seed) == (b.seed, b.population_seed, b.task_seed) == (7, 7, 7)
    assert (
        b.world.label_noise == 0.2 and b.world.tutoring_episodes_per_tick == 9 and a.world.label_noise == 0.05
    )


def test_metric_functions_on_agent_states():
    fw = FakeWorld(SimulationConfig(seed=1, n_agents=5, years=1), None, "r")
    agents = fw.agents()
    assert 0.5 < society.cohort_mean_accuracy(agents, age_min=0, age_max=100) < 0.7
    assert society.cohort_size(agents, age_min=5, age_max=6) == 3
    assert np.isnan(society.cohort_mean_accuracy(agents, age_min=90, age_max=100))
    assert -1 <= society.biological_parent_corr(agents, age_min=0, age_max=100) <= 1
    assert 0 <= society.resource_gini(agents) or np.isnan(society.resource_gini(agents))
