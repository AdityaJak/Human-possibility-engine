"""Society kernel tests: determinism, replay, accounting, mechanisms, interventions."""

from __future__ import annotations

import ast
import json
import pathlib
import time

import numpy as np
import pytest
from world_test_learner import linear_factory

from hpe.contracts.protocols import ListSink, WorldRunner
from hpe.contracts.runtime import random_policy_factory
from hpe.contracts.schemas import (
    AgentState,
    EventType,
    Intervention,
    InterventionKind,
    InterventionTarget,
    LogConfig,
    RunManifest,
    SimulationConfig,
    WorldParams,
)
from hpe.tasks import TRAINING_FAMILIES
from hpe.world import World

WORLD_DIR = pathlib.Path(__file__).resolve().parents[2] / "packages" / "hpe" / "world"


# Demography-focused tests do not need many learning episodes: shrink per-tick episode counts.
LIGHT = WorldParams(
    school_episodes_per_tick=1,
    tutoring_episodes_per_tick=1,
    self_study_episodes_per_tick=1,
    work_episodes_per_tick=1,
    caregiver_episodes_per_tick=1,
    peer_demonstrations_per_social_tick=1,
)
SPARSE = LogConfig(assessment_interval_years=5.0)


def cfg(**kw) -> SimulationConfig:
    base = {"seed": 3, "n_agents": 20, "years": 2.0, "dt_days": 60}
    base.update(kw)
    return SimulationConfig(**base)


def event_types(sink: ListSink) -> set[str]:
    return {e.type.value for e in sink.events}


# ------------------------------------------------------------------ determinism & replay


def test_world_satisfies_protocol():
    assert isinstance(World(cfg(), random_policy_factory), WorldRunner)


def test_same_config_same_event_hash():
    a, b = World(cfg(), linear_factory), World(cfg(), linear_factory)
    a.run()
    b.run()
    assert a.event_stream_hash() == b.event_stream_hash()
    assert a.state_hash() == b.state_hash()
    assert a.tick == b.tick and a.done


def test_different_seeds_differ():
    a, b = World(cfg(seed=1), random_policy_factory), World(cfg(seed=2), random_policy_factory)
    a.run()
    b.run()
    assert a.event_stream_hash() != b.event_stream_hash()


def test_snapshot_resume_is_bit_identical():
    c = cfg(years=4.0)
    full = World(c, linear_factory)
    full.run()
    part = World(c, linear_factory)
    part.run(until_year=2.0)
    snap = json.loads(json.dumps(part.snapshot()))  # must survive JSON
    resumed = World(c, linear_factory)
    resumed.load_snapshot(snap)
    assert resumed.tick == part.tick
    resumed.run()
    assert resumed.event_stream_hash() == full.event_stream_hash()
    assert resumed.state_hash() == full.state_hash()
    assert len(resumed.outcomes()) == len(full.outcomes())


def test_snapshot_rejects_other_config():
    w = World(cfg(), random_policy_factory)
    w.step(2)
    other = World(cfg(seed=99), random_policy_factory)
    with pytest.raises(ValueError):
        other.load_snapshot(w.snapshot())


def test_run_id_does_not_change_event_hash():
    a = World(cfg(), random_policy_factory, run_id="x")
    b = World(cfg(), random_policy_factory, run_id="y")
    a.run()
    b.run()
    assert a.event_stream_hash() == b.event_stream_hash()


def test_sink_receives_every_event_and_tick():
    sink = ListSink()
    w = World(cfg(log=LogConfig(frames_every_ticks=6)), random_policy_factory, sink=sink)
    w.run()
    assert len(sink.events) == w.manifest().n_events
    assert [e.seq for e in sink.events] == list(range(len(sink.events)))
    assert len(sink.ticks) == w.tick + 1
    assert any(t.frames for t in sink.ticks) and not all(t.frames for t in sink.ticks)
    assert sink.events[0].type == EventType.RUN_START and sink.events[-1].type == EventType.RUN_END


def test_max_events_truncates_run():
    w = World(cfg(max_events=200), random_policy_factory)
    w.run()
    assert w.done and w.manifest().n_events == 200
    assert "truncated" in " ".join(w.manifest().nondeterminism)


# ------------------------------------------------------------------ accounting & lifecycle


def _check_accounting(w: World) -> None:
    alive = set(w.agent_ids(include_dead=False))
    all_ids = w.agent_ids()
    births = w._counters["births"]
    deaths = w._counters["deaths"]
    assert len(all_ids) == w.config.n_agents + births
    assert len(alive) == len(all_ids) - deaths
    members = [m for hh in w._households.values() for m in hh.members]
    assert len(members) == len(set(members)), "agent in two households"
    assert set(members) == alive, "household membership must equal the alive set"
    for hh in w._households.values():
        assert hh.resources >= 0.0
    for aid in alive:
        a = w._agents[aid]
        assert w._households[a.household_id] is not None
        if a.partner_id:
            assert w._agents[a.partner_id].partner_id == aid


def test_population_accounting_conserved():
    w = World(cfg(n_agents=40, years=30.0, dt_days=122, world=LIGHT, log=SPARSE), random_policy_factory)
    for y in (5.0, 15.0, 30.0):
        w.run(until_year=y)
        _check_accounting(w)
    assert w.ticks_history[-1].population == len(w.agent_ids(include_dead=False))


@pytest.mark.slow
def test_lifecycle_events_occur():
    sink = ListSink()
    w = World(
        cfg(n_agents=40, years=30.0, dt_days=122, world=LIGHT, log=SPARSE), random_policy_factory, sink=sink
    )
    w.run()
    types = event_types(sink)
    for t in (
        "birth",
        "death",
        "stage_change",
        "school_enroll",
        "school_graduate",
        "job_start",
        "partnership_formed",
        "assessment",
        "policy_update",
        "learning_episode",
        "shock",
        "wage",
    ):
        assert t in types, f"missing {t}"
    assert w._counters["births"] > 0 and w._counters["deaths"] > 0
    newborn = next(e for e in sink.events if e.type == EventType.BIRTH and not e.payload["initial"])
    assert newborn.payload["inherited"]["knowledge_inherited"] is False
    assert len(newborn.other_ids) == 2


def test_assessments_measure_random_policy_at_chance():
    w = World(cfg(), random_policy_factory)
    w.run()
    states = [s for s in w.agents(include_dead=False) if s.age_years >= 2]
    assert states
    for s in states:
        assert abs(s.capabilities["sort"].measured_accuracy - 1 / 3) < 1e-9
        assert abs(s.capabilities["barter"].measured_accuracy - 0.5) < 1e-9


def test_agent_state_and_manifest_are_valid_contract_objects():
    w = World(cfg(), linear_factory)
    w.run()
    for s in w.agents():
        AgentState.model_validate(s.model_dump(mode="json"))
    m = w.manifest()
    RunManifest.model_validate(m.model_dump(mode="json"))
    assert m.compute is not None and m.compute.llm_calls == 0 and m.compute.agent_years > 0
    assert m.status.value == "completed" and m.task_world_hash
    aid = w.agent_ids()[3]
    assert w.timeline(aid) and w.skill_curve(aid) and w.parameter_history(aid)
    assert w.edges(aid)


def test_frames_and_map_are_valid():
    w = World(cfg(), random_policy_factory)
    tick = w.step(3, include_frames=True)
    mp = w.world_map()
    ids = {p.place_id for p in mp.places}
    assert {"plaza", "school_north", "work_market"} <= ids
    assert tick.frames is not None and len(tick.frames) == tick.population
    for f in tick.frames:
        assert 0 <= f.x <= mp.width and 0 <= f.y <= mp.height
        assert f.place_id in ids
    assert w.ticks_history[-1].frames is None  # history stores no frames


def test_learning_changes_measured_skill_and_parameters():
    w = World(cfg(n_agents=24, years=5.0), linear_factory)
    w.run()
    kids = [s for s in w.agents(include_dead=False) if 8 <= s.age_years < 18]
    assert kids
    mean = np.mean([s.capabilities["compare"].measured_accuracy for s in kids])
    assert mean > 0.65, mean  # held-out, chance = 0.5
    hist = w.parameter_history(kids[0].agent_id)
    assert hist[0]["summary"]["n_updates"] < hist[-1]["summary"]["n_updates"]


# ------------------------------------------------------------------ interventions


def _tutoring(fraction: float = 1.0, **params) -> Intervention:
    return Intervention(
        name="tutoring",
        kind=InterventionKind.TUTORING_ACCESS,
        target=InterventionTarget(age_min=5, age_max=17.99, fraction=fraction),
        params=params,
    )


def _learning_events(sink: ListSink, source: str) -> int:
    return sum(
        e.payload.get("by_source", {}).get(source, {}).get("n", 0)
        for e in sink.events
        if e.type == EventType.LEARNING_EPISODE
    )


def test_tutoring_changes_inputs_only_for_nonlearner():
    """With a non-learning policy, tutoring adds episodes but cannot change measured skill or anything else."""
    c = cfg(n_agents=20, years=3.0)
    sc, st = ListSink(), ListSink()
    ctrl = World(c, random_policy_factory, sink=sc)
    treat = World(c.model_copy(update={"interventions": [_tutoring()]}), random_policy_factory, sink=st)
    ctrl.run()
    treat.run()
    assert _learning_events(sc, "tutor") == 0 and _learning_events(st, "tutor") > 0
    acc_c = [
        s.capabilities[f].measured_accuracy
        for s in ctrl.agents()
        for f in TRAINING_FAMILIES
        if s.capabilities[f].measured_accuracy is not None
    ]
    acc_t = [
        s.capabilities[f].measured_accuracy
        for s in treat.agents()
        for f in TRAINING_FAMILIES
        if s.capabilities[f].measured_accuracy is not None
    ]
    assert acc_c == acc_t  # random policy: zero skill effect, exactly
    # common random numbers: untouched mechanisms give identical lives
    assert [(s.agent_id, s.death_tick, s.job_id) for s in ctrl.agents()] == [
        (s.agent_id, s.death_tick, s.job_id) for s in treat.agents()
    ]


@pytest.mark.slow
def test_tutoring_raises_measured_skill_through_learning():
    c = cfg(n_agents=24, years=8.0, dt_days=91, seed=5)
    ctrl = World(c, linear_factory)
    treat = World(c.model_copy(update={"interventions": [_tutoring()]}), linear_factory)
    ctrl.run()
    treat.run()

    def mean_skill(w: World) -> float:
        vals = [
            s.capabilities[f].measured_accuracy
            for s in w.agents(include_dead=False)
            if 7 <= s.age_years < 18
            for f in TRAINING_FAMILIES
        ]
        return float(np.mean(vals))

    mc, mt = mean_skill(ctrl), mean_skill(treat)
    print(f"tutoring effect (test learner, 1 seed): control={mc:.3f} treatment={mt:.3f}")
    assert mt > mc


def test_common_random_numbers_for_untargeted_agents():
    """Partial tutoring: untargeted agents' mortality streams advance identically in both arms."""
    c = cfg(n_agents=20, years=3.0)
    ctrl = World(c, linear_factory)
    treat = World(c.model_copy(update={"interventions": [_tutoring(fraction=0.5)]}), linear_factory)
    ctrl.run()
    treat.run()
    untargeted = [
        aid
        for aid in treat.agent_ids()
        if not treat._agents[aid].flags.get("iv:tutoring:assigned", False) and aid in ctrl._agents
    ]
    assert untargeted
    same = 0
    for aid in untargeted:
        a_c, a_t = ctrl._agents[aid], treat._agents[aid]
        if a_c.alive == a_t.alive and a_c.death_tick == a_t.death_tick:
            sc = ctrl.streams.get("agent", aid, "life").bit_generator.state
            st = treat.streams.get("agent", aid, "life").bit_generator.state
            assert sc == st
            same += 1
    assert same >= len(untargeted) * 0.9


def test_school_access_block_prevents_enrolment():
    sink = ListSink()
    iv = Intervention(name="no_school", kind=InterventionKind.SCHOOL_ACCESS, params={"access": False})
    w = World(cfg(interventions=[iv]), random_policy_factory, sink=sink)
    w.run()
    assert "school_enroll" not in event_types(sink)
    assert all(not s.enrolled_school for s in w.agents())


def test_misinformation_and_isolation_run():
    mis = Intervention(
        name="rumour",
        kind=InterventionKind.MISINFORMATION,
        target=InterventionTarget(age_min=18, fraction=0.3),
    )
    iso = Intervention(
        name="iso", kind=InterventionKind.SOCIAL_ISOLATION, target=InterventionTarget(fraction=1.0)
    )
    w1 = World(cfg(interventions=[mis]), linear_factory)
    w1.run()
    assert any(a.flags.get("misinformer") for a in w1._agents.values())
    w2 = World(cfg(interventions=[iso]), linear_factory)
    w2.run()
    for src, adj in w2._edges.items():
        for dst, e in adj.items():
            if e.kind in ("classmate", "coworker", "acquaintance", "friend"):
                pytest.fail(f"isolated society formed non-family tie {src}->{dst} ({e.kind})")


def test_cash_transfer_and_midlife_education():
    cash = Intervention(
        name="cash", kind=InterventionKind.CASH_TRANSFER, params={"amount": 50.0, "period": "once"}
    )
    a = World(cfg(), random_policy_factory)
    b = World(cfg(interventions=[cash]), random_policy_factory)
    a.step(1)
    b.step(1)
    assert (
        sum(h.resources for h in b._households.values())
        > sum(h.resources for h in a._households.values()) + 100
    )
    mid = Intervention(
        name="mid", kind=InterventionKind.MIDLIFE_EDUCATION, target=InterventionTarget(age_min=30, age_max=60)
    )
    c = World(cfg(), random_policy_factory)
    d = World(cfg(interventions=[mid]), random_policy_factory)
    c.run()
    d.run()
    adult_ed = {s.agent_id: s.education_years for s in c.agents() if 30 <= s.age_years <= 60}
    treated = {s.agent_id: s.education_years for s in d.agents() if s.agent_id in adult_ed}
    assert sum(treated.values()) > sum(adult_ed.values())


@pytest.mark.slow
def test_cross_foster_moves_newborns():
    iv = Intervention(
        name="xf", kind=InterventionKind.CROSS_FOSTER, target=InterventionTarget(age_min=0, fraction=1.0)
    )
    sink = ListSink()
    w = World(
        cfg(n_agents=30, years=12.0, dt_days=122, world=LIGHT, log=SPARSE, interventions=[iv]),
        random_policy_factory,
        sink=sink,
    )
    w.run()
    births = [e for e in sink.events if e.type == EventType.BIRTH and not e.payload["initial"]]
    assert births
    moved = [e for e in births if e.payload["inherited"].get("cross_fostered")]
    assert moved
    for e in moved:
        inh = e.payload["inherited"]
        assert inh["rearing_household"] != inh["biological_household"]


def test_caregiving_support_adds_demonstrations():
    iv = Intervention(
        name="care",
        kind=InterventionKind.CAREGIVING_SUPPORT,
        target=InterventionTarget(age_min=1, age_max=5.99),
    )
    s0, s1 = ListSink(), ListSink()
    World(cfg(), random_policy_factory, sink=s0).run()
    World(cfg(interventions=[iv]), random_policy_factory, sink=s1).run()
    assert _learning_events(s1, "caregiver") > _learning_events(s0, "caregiver") > 0


# ------------------------------------------------------------------ boundaries & scale


def test_world_package_never_imports_learners_or_evaluation():
    banned = ("hpe.development", "hpe.evaluation", "hpe_api", "hpe.storage", "hpe.futures")
    for path in WORLD_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(banned), f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith(banned), f"{path.name} imports {n.name}"


def test_coarse_time_step_runs():
    w = World(
        cfg(n_agents=20, years=6.0, dt_days=182, world=WorldParams(enable_age_gate=False)),
        random_policy_factory,
    )
    w.run()
    assert w.done and w.tick == 13


@pytest.mark.slow
def test_full_lifespan_100_agents_80_years():
    t0 = time.perf_counter()
    w = World(SimulationConfig(seed=11, n_agents=100, years=80.0), random_policy_factory)
    w.run()
    wall = time.perf_counter() - t0
    m = w.manifest()
    print(
        f"100 agents x 80 years: {wall:.1f}s wall, {m.n_events} events, births={w._counters['births']}, "
        f"deaths={w._counters['deaths']}, final population={w.ticks_history[-1].population}"
    )
    assert w.done and w.tick == 974
    assert w._counters["births"] > 30 and w._counters["deaths"] > 30
    _check_accounting(w)
    # at least one agent has a recorded lifespan of >= 60 simulated years inside the run
    spans = []
    for aid in w.agent_ids():
        curve = w.skill_curve(aid)
        if curve:
            spans.append(curve[-1]["age_years"] - curve[0]["age_years"])
    assert max(spans) >= 60


@pytest.mark.slow
def test_1000_agents_2_years():
    t0 = time.perf_counter()
    w = World(SimulationConfig(seed=2, n_agents=1000, years=2.0), random_policy_factory)
    w.run()
    wall = time.perf_counter() - t0
    print(f"1000 agents x 2 years: {wall:.1f}s wall, {w.manifest().n_events} events")
    assert w.done and w.ticks_history[-1].population > 900
