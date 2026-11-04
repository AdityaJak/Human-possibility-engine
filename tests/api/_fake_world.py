"""A small deterministic WorldRunner used to test the API without the real society engine.

It implements ``hpe.contracts.protocols.WorldRunner`` and emits the event shapes the API relies
on (BIRTH, LEARNING_EPISODE, DECISION, ASSESSMENT, POLICY_UPDATE, FRIENDSHIP_FORMED). It is not a
simulation of anything and must never be used outside tests.
"""

from __future__ import annotations

import copy
from typing import Any

from hpe.contracts.protocols import EventSink, ListSink
from hpe.contracts.runtime import AgentInit, Policy, PolicyFactory
from hpe.contracts.schemas import (
    AgentFrame,
    AgentState,
    CapabilityState,
    EventType,
    ExperienceEvent,
    LifeStage,
    OutcomeObservation,
    Place,
    PolicyClass,
    PolicyDescriptor,
    Provenance,
    RunManifest,
    SimulationConfig,
    SocialEdge,
    SocialEdgeKind,
    WorldMap,
    WorldTick,
)
from hpe.core.hashing import stable_hash
from hpe.tasks import ALL_FAMILIES


class FakeWorld:
    def __init__(
        self,
        config: SimulationConfig,
        policy_factory: PolicyFactory,
        sink: EventSink | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id or "fake-run"
        self.sink = sink or ListSink()
        self.tick = 0
        self._seq = 0
        self._hash: list[str] = []
        self.total_ticks = max(1, round(config.years * 365 / config.dt_days))
        self._ids = [f"a{i:04d}" for i in range(config.n_agents)]
        self._acc: dict[str, dict[str, float]] = {a: dict.fromkeys(ALL_FAMILIES, 0.4) for a in self._ids}
        self._edges: list[SocialEdge] = []
        self.policies: dict[str, Policy] = {
            a: policy_factory(
                AgentInit(
                    a, 0, config.seed, {"curiosity": 0.5}, {}, {}, ["rest", "study", "work", "socialize"]
                )
            )
            for a in self._ids
        }
        for a in self._ids:
            self._emit(EventType.BIRTH, a, {"household_id": self._household(a)})

    # ---------------------------------------------------------------- helpers
    def _household(self, a: str) -> str:
        return f"h{int(a[1:]) // 4:03d}"

    def _emit(
        self,
        type_: EventType,
        agent_id: str | None,
        payload: dict[str, Any],
        other_ids: list[str] | None = None,
    ) -> None:
        ev = ExperienceEvent(
            seq=self._seq,
            run_id=self.run_id,
            tick=self.tick,
            day=self.tick * self.config.dt_days,
            type=type_,
            agent_id=agent_id,
            other_ids=other_ids or [],
            payload=payload,
            provenance=Provenance(source="world", mechanism="fake_world"),
        )
        self._seq += 1
        self._hash.append(stable_hash(ev.model_dump(mode="json")))
        self.sink.emit(ev)

    @property
    def done(self) -> bool:
        return self.tick >= self.total_ticks

    @property
    def year(self) -> float:
        return self.tick * self.config.dt_days / 365.0

    # ---------------------------------------------------------------- protocol
    def step(self, n_ticks: int = 1, *, include_frames: bool = False) -> WorldTick:
        last: WorldTick | None = None
        for _ in range(n_ticks):
            if self.done:
                break
            self.tick += 1
            for a in self._ids:
                self._emit(EventType.LEARNING_EPISODE, a, {"family": "sort", "n": 3, "mean_reward": 0.5})
                if self.tick % 3 == 0:
                    self._emit(
                        EventType.DECISION,
                        a,
                        {"explanation": {"action": "study", "values": [0.1, 0.4, 0.2, 0.3]}},
                    )
                if self.tick % 12 == 0:
                    for fam in ALL_FAMILIES:
                        self._acc[a][fam] = min(0.95, self._acc[a][fam] + 0.05)
                    self._emit(
                        EventType.ASSESSMENT,
                        a,
                        {
                            "age_years": self.year,
                            "results": {
                                fam: {
                                    "accuracy": v,
                                    "ci_low": v - 0.1,
                                    "ci_high": v + 0.1,
                                    "n_items": 60,
                                    "log_loss": 0.9,
                                }
                                for fam, v in self._acc[a].items()
                            },
                        },
                    )
                    self._emit(
                        EventType.POLICY_UPDATE,
                        a,
                        {
                            "parameter_summary": {
                                "n_updates": self.tick,
                                "recent_memories": [{"family": "sort", "tick": self.tick}],
                            }
                        },
                    )
            if self.tick == 6 and len(self._ids) > 1:
                self._edges.append(
                    SocialEdge(
                        src=self._ids[0],
                        dst=self._ids[1],
                        kind=SocialEdgeKind.FRIEND,
                        trust=0.7,
                        formed_tick=6,
                    )
                )
                self._emit(EventType.FRIENDSHIP_FORMED, self._ids[0], {}, other_ids=[self._ids[1]])
            every = self.config.log.frames_every_ticks
            frames = self.frames() if (every and self.tick % every == 0) else None
            last = WorldTick(
                run_id=self.run_id,
                tick=self.tick,
                day=self.tick * self.config.dt_days,
                year=self.year,
                population=len(self._ids),
                frames=frames,
            )
            self.sink.emit_tick(last)
        if last is None:
            last = WorldTick(
                run_id=self.run_id,
                tick=self.tick,
                day=self.tick * self.config.dt_days,
                year=self.year,
                population=len(self._ids),
            )
        if include_frames:
            last = last.model_copy(update={"frames": self.frames()})
        elif last.frames is not None:
            last = last.model_copy(update={"frames": None})
        return last

    def run(self, *, until_year: float | None = None, progress: Any = None) -> WorldTick:
        tick = self.step(0)
        while not self.done and (until_year is None or self.year < until_year):
            tick = self.step(1)
        return tick

    def agents(self, *, include_dead: bool = True) -> list[AgentState]:
        return [self.agent(a) for a in self._ids]

    def agent(self, agent_id: str) -> AgentState:
        if agent_id not in self._acc:
            raise KeyError(agent_id)
        caps = {
            f: CapabilityState(family=f, measured_accuracy=v if self.tick >= 12 else None)
            for f, v in self._acc[agent_id].items()
        }
        return AgentState(
            agent_id=agent_id,
            run_id=self.run_id,
            household_id=self._household(agent_id),
            birth_tick=0,
            birth_cohort_year=0,
            age_years=self.year,
            stage=LifeStage.CHILD,
            capabilities=caps,
            policy=PolicyDescriptor(policy_class=PolicyClass.RANDOM, version="1.0.0"),
            n_friends=1 if agent_id in ("a0000", "a0001") else 0,
        )

    def policy_of(self, agent_id: str) -> Policy:
        return self.policies[agent_id]

    def edges(self, agent_id: str | None = None) -> list[SocialEdge]:
        return [e for e in self._edges if agent_id is None or agent_id in (e.src, e.dst)]

    def frames(self) -> list[AgentFrame]:
        return [
            AgentFrame(
                id=a,
                x=float(i),
                y=float(self.tick % 10),
                place_id="p0",
                activity="study",
                age=self.year,
                alive=True,
                household_id=self._household(a),
            )
            for i, a in enumerate(self._ids)
        ]

    def world_map(self) -> WorldMap:
        return WorldMap(
            width=100, height=60, places=[Place(place_id="p0", kind="school", name="School", x=50, y=30)]
        )

    def outcomes(self) -> list[OutcomeObservation]:
        return [
            OutcomeObservation(
                run_id=self.run_id, agent_id=a, metric="accuracy_sort", value=v["sort"], split="holdout"
            )
            for a, v in self._acc.items()
        ]

    def manifest(self) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            config=self.config,
            config_hash=stable_hash(self.config.model_dump(mode="json")),
            policy_class=self.config.policy_class,
            n_events=self._seq,
        )

    def event_stream_hash(self) -> str:
        return stable_hash(self._hash)

    def snapshot(self) -> dict[str, Any]:
        return {"tick": self.tick, "seq": self._seq, "acc": copy.deepcopy(self._acc)}

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.tick = snapshot["tick"]
        self._seq = snapshot["seq"]
        self._acc = copy.deepcopy(snapshot["acc"])


def fake_world_factory(
    config: SimulationConfig, policy_factory: PolicyFactory, sink: EventSink, run_id: str
) -> FakeWorld:
    return FakeWorld(config, policy_factory, sink=sink, run_id=run_id)
