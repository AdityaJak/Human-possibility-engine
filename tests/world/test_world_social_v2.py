"""Social mechanism v2 (Phase II): flag-guarded, instrumented, deterministic, snapshot-safe."""

from __future__ import annotations

import pytest

from hpe.contracts.protocols import ListSink
from hpe.contracts.registry import get_factory
from hpe.contracts.schemas import (
    Intervention,
    InterventionKind,
    InterventionTarget,
    LogConfig,
    SimulationConfig,
)
from hpe.world import World
from hpe.world.social import STATS_KEYS, SocialV2Config

pytest.importorskip("hpe.development.social")

LOG = LogConfig(
    frames_every_ticks=0, decision_sample_rate=0.0, log_learning_episodes=False, checkpoint_every_years=None
)


def _cfg(
    seed: int = 1000,
    n: int = 25,
    years: float = 3.0,
    params: dict | None = None,
    interventions: list | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        seed=seed,
        n_agents=n,
        years=years,
        policy_class="developmental",
        policy_params=dict(params or {}),
        interventions=list(interventions or []),
        log=LOG,
    )


def _world(cfg: SimulationConfig, sink: ListSink | None = None) -> World:
    return World(cfg, get_factory("developmental_social", dict(cfg.policy_params)), sink=sink)


def test_flag_absent_never_touches_v2():
    sink = ListSink()
    w = _world(_cfg(params={}), sink)
    assert w._sv2 is None
    w.run()
    assert all(v == 0.0 for v in w._sv2_state.stats.values())
    assert not any(e.provenance.mechanism.startswith("world.social.v2") for e in sink.events)


def test_v2_creates_repeat_contacts_ties_and_informative_demonstrations():
    sink = ListSink()
    w = _world(_cfg(params={"social_v2": {}}), sink)
    assert w._sv2 is not None
    w.run()
    st = w._sv2_state.stats
    assert st["meetings"] > 0 and st["repeat_meetings"] > 0
    assert st["ties_formed"] > 0
    assert st["friendships"] >= 1
    assert st["demos"] > 0 and st["demos_full"] > 0 and st["info_gain_n"] == st["demos"]
    assert st["cooperation_items"] > 0
    yearly = [e for e in sink.events if e.provenance.mechanism == "world.social.v2.stats"]
    assert len(yearly) >= 2 and set(yearly[-1].payload["cumulative"]) == set(STATS_KEYS)
    assert w._sv2_state.stats["demos_correct"] / w._sv2_state.stats["demos"] > 0.5


def test_social_reward_only_control_carries_no_information():
    w = _world(_cfg(params={"social_v2": {"information": False}}))
    w.run()
    st = w._sv2_state.stats
    assert st["meetings"] > 0 and st["ties_formed"] > 0
    assert st["demos"] == 0 and st["cooperation_items"] == 0 and st["info_gain_n"] == 0


def test_v2_is_deterministic():
    a = _world(_cfg(params={"social_v2": {}}))
    b = _world(_cfg(params={"social_v2": {}}))
    a.run()
    b.run()
    assert a.event_stream_hash() == b.event_stream_hash()
    assert a._sv2_state.stats == b._sv2_state.stats


def test_v2_snapshot_resume_is_bit_identical():
    full = _world(_cfg(params={"social_v2": {}}))
    full.run()
    part = _world(_cfg(params={"social_v2": {}}))
    part.run(until_year=1.5)
    snap = part.snapshot()
    assert snap["social_v2"] is not None
    resumed = _world(_cfg(params={"social_v2": {}}))
    resumed.load_snapshot(snap)
    resumed.run()
    assert resumed.event_stream_hash() == full.event_stream_hash()
    assert resumed._sv2_state.stats == full._sv2_state.stats


def test_misinformers_lose_trust_through_observed_reliability():
    rumours = Intervention(
        name="rumours",
        kind=InterventionKind.MISINFORMATION,
        target=InterventionTarget(age_min=18, age_max=80, fraction=0.3),
        start_year=0.0,
    )
    w = _world(_cfg(n=40, years=4.0, params={"social_v2": {}}, interventions=[rumours]))
    w.run()
    st = w._sv2_state
    assert st.stats["demos_from_misinformers"] > 0
    assert st.stats["demos_from_misinformers_correct"] == 0
    assert st.trust_misinformer[-1] < st.trust_honest[-1]


def test_isolation_still_blocks_non_family_contact_under_v2():
    isolation = Intervention(
        name="iso",
        kind=InterventionKind.SOCIAL_ISOLATION,
        target=InterventionTarget(fraction=1.0),
        start_year=0.0,
    )
    w = _world(_cfg(params={"social_v2": {}}, interventions=[isolation]))
    w.run()
    assert w._sv2_state.stats["friendships"] == 0
    assert w._sv2_state.stats["cooperation_items"] == 0


def test_unknown_social_v2_param_raises():
    with pytest.raises(ValueError):
        SocialV2Config.from_params({"social_v2": {"bogus": 1}})
    assert SocialV2Config.from_params({"social_v2": True}) == SocialV2Config()
    assert SocialV2Config.from_params({}) is None
