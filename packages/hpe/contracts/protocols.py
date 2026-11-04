"""Cross-lane integration protocols (contract v0.1).

World lane implements ``WorldRunner``; storage lane implements durable ``EventSink``s;
evaluation lane drives ``WorldRunner`` through ``hpe.contracts.registry`` policy factories;
backend lane wraps all of it behind HTTP.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from hpe.contracts.runtime import AgentInit, Policy, PolicyFactory
from hpe.contracts.schemas import (
    AgentFrame,
    AgentState,
    ExperienceEvent,
    OutcomeObservation,
    RunManifest,
    SimulationConfig,
    SocialEdge,
    WorldMap,
    WorldTick,
)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: ExperienceEvent) -> None: ...

    def emit_tick(self, tick: WorldTick) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class ListSink:
    """In-memory sink for tests and small interactive runs."""

    def __init__(self) -> None:
        self.events: list[ExperienceEvent] = []
        self.ticks: list[WorldTick] = []

    def emit(self, event: ExperienceEvent) -> None:
        self.events.append(event)

    def emit_tick(self, tick: WorldTick) -> None:
        self.ticks.append(tick)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


@runtime_checkable
class WorldRunner(Protocol):
    """A deterministic, checkpointable society simulation.

    Construction: ``World(config: SimulationConfig, policy_factory: PolicyFactory,
    sink: EventSink | None = None, run_id: str | None = None)``.

    Determinism contract: two worlds built with equal ``config`` and the same policy
    factory produce identical event streams (``event_stream_hash``). ``load_snapshot`` of a
    snapshot taken at tick T followed by ``run`` produces the same remaining events as the
    uninterrupted run.
    """

    run_id: str
    config: SimulationConfig
    tick: int

    def step(self, n_ticks: int = 1, *, include_frames: bool = False) -> WorldTick: ...

    def run(
        self, *, until_year: float | None = None, progress: Callable[[float], None] | None = None
    ) -> WorldTick: ...

    @property
    def done(self) -> bool: ...

    @property
    def year(self) -> float: ...

    def agents(self, *, include_dead: bool = True) -> list[AgentState]: ...

    def agent(self, agent_id: str) -> AgentState: ...

    def policy_of(self, agent_id: str) -> Policy: ...

    def edges(self, agent_id: str | None = None) -> list[SocialEdge]: ...

    def frames(self) -> list[AgentFrame]: ...

    def world_map(self) -> WorldMap: ...

    def outcomes(self) -> list[OutcomeObservation]: ...

    def manifest(self) -> RunManifest: ...

    def event_stream_hash(self) -> str: ...

    def snapshot(self) -> dict[str, Any]: ...

    def load_snapshot(self, snapshot: dict[str, Any]) -> None: ...


PolicyFactoryBuilder = Callable[[dict[str, Any]], PolicyFactory]
"""Given ``policy_params``, return a factory that builds one Policy per ``AgentInit``."""


def iter_factories(builders: dict[str, PolicyFactoryBuilder]) -> Iterable[tuple[str, PolicyFactoryBuilder]]:
    return sorted(builders.items())


__all__ = [
    "AgentInit",
    "EventSink",
    "ListSink",
    "PolicyFactoryBuilder",
    "WorldRunner",
    "iter_factories",
]
